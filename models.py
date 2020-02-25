"""Model functions.

"""
import numpy as np
import pandas as pd
import pymc3 as pm
import theano.tensor as tt
import matplotlib.pyplot as plt

from os.path import exists

from patsy import dmatrix
from scipy.special import expit as logistic


def fit_model(name, func):
    """Fits a model (in a Bayesian sense) to the data.

    This was written as a function so that some of the code can be re-used for the
    secondary model.

    Args:
        name (str): Descriptive name of the model. Posterior samples, statistics, and
            figures are generated and saved in a subdirectory with this name.
        func (:obj:`<class 'function'>): Function for model construction. Should
            return a formatted copy of the data.

    Returns:
        data (pandas.DataFrame): The formatted copy of the data augmented with the
            results from the model fitting.

    """
    with pm.Model() as m:

        # construct model and load data
        data = func()

        if exists(f"{name}") is False:
            # sample posterior
            trace = pm.sample(10000, tune=1000, chains=2)
            pm.save_trace(trace, f"{name}")
        else:
            # load samples
            trace = pm.load_trace(f"{name}")

        if exists(f"{name}/ppc.npz") is False:
            # perform ppc
            ppc = pm.sample_posterior_predictive(trace, samples=10000)["y"]
            np.savez_compressed(f"{name}/ppc.npz", ppc)
        else:
            # load pp samples
            ppc = np.load(f"{name}/ppc.npz")["arr_0"]

        if exists(f"{name}/summary.csv") is False:
            # make a summary csv
            summary = pm.summary(trace, var_names=m.free_RVs)
            summary.to_csv(f"{name}/summary.csv")
        else:
            summary = pd.read_csv(f"{name}/summary.csv")

        if exists(f"{name}/details.txt") is False:

            details = f"Minimum Rhat = {summary.Rhat.min()}\n"
            details += f"Minimum Neff = {summary.n_eff.min()}\n"
            n = data.trials.mean()
            r2 = pm.stats.r2_score(data.num.values, trace["p"] * n, 3)
            details += f"Bayesian median R2 = {r2[0]}\n"
            try:
                details += f"BFMI = {pm.stats.bfmi(trace)}\n"
            except KeyError:
                details += f"No BFMI generated!\n"

            open(f"{name}/details.txt", "w").write(details)

        if exists(f"{name}/traceplot.png") is False:

            pm.traceplot(trace, compact=True)
            plt.savefig(f"{name}/traceplot.png")

        if exists(f"{name}/data.csv") is False:

            data["a"] = logistic(trace[r"$\alpha$"].mean(axis=0))
            data["b"] = trace[r"$\beta$"].mean(axis=0)
            data["l"] = logistic(trace[r"$\lambda$"].mean(axis=0))
            data["s"] = np.exp(trace[r"$\varsigma$"].mean(axis=0))
            data["d"] = np.exp(trace[r"$\delta$"].mean(axis=0))

            ppq = pd.DataFrame(ppc).quantile([0.025, 0.975]).T
            ppq.columns = ["ppc_lo", "ppc_hi"]
            data = pd.concat([data, ppq], axis=1)
            data["ppc_lo"] /= data.trials
            data["ppc_hi"] /= data.trials

            ppe = np.abs(ppc - np.tile(data.num.values, (10000, 1))).mean(axis=0)
            data["pro_pp_errors"] = ppe / data.trials
            data.to_csv(f"{name}/data.csv", index=False)

        else:

            data = pd.read_csv(f"{name}/data.csv")

    return data


def modela():
    """Constructs the primary model used in the manuscript.

    Returns:
        data (pandas.DataFrame): A formatted copy of the data.

    """
    # load the data
    _data = pd.read_csv("data.csv")
    kwargs = dict(values="response", index=("listener", "condition", "isi", "delta"))
    data = pd.pivot_table(_data, aggfunc=len, **kwargs).reset_index()
    cols = data.columns.tolist()
    cols[4] = "trials"
    data.columns = cols
    _s = pd.pivot_table(_data, aggfunc=sum, **kwargs).reset_index().response
    data["num"] = _s
    data["prop"] = data.num / data.trials

    # design matrix for fixed effects
    biggamma = dmatrix("0 + C(condition)", data)
    feffs = biggamma.design_info.column_names
    nf = len(feffs)

    # design matrix for random effects
    bigtheta = dmatrix("0 + C(listener)", data)
    reffs = bigtheta.design_info.column_names
    nr = len(reffs)

    # two effects per row; this ensures induced s.d. = 1
    sd = 1 / np.sqrt(2)

    # preference during lapse, a
    za = pm.Normal(name=r"$\zeta_{\alpha}$", mu=0, sd=sd, shape=nf)
    xa = pm.Normal(name=r"$\xi_{\alpha}$", mu=0, sd=sd, shape=nr)
    alpha = pm.Deterministic(
        r"$\alpha$", tt.dot(np.asarray(biggamma), za) + tt.dot(np.asarray(bigtheta), xa)
    )
    a = pm.Deterministic("$a$", pm.math.sigmoid(alpha))
    pm.Deterministic(r"$\Lambda_{\alpha}$", za[1] - za[0])

    # bias, b
    zb = pm.Normal(name=r"$\zeta_{\beta}$", mu=0, sd=sd, shape=nf)
    xb = pm.Normal(name=r"$\xi_{\beta}$", mu=0, sd=sd, shape=nr)
    beta = pm.Deterministic(
        r"$\beta$", tt.dot(np.asarray(biggamma), zb) + tt.dot(np.asarray(bigtheta), xb)
    )
    b = pm.Deterministic("$b$", beta)
    pm.Deterministic(r"$\Lambda_{\beta}$", zb[1] - zb[0])

    # lapse rate, l
    zl = pm.Normal(name=r"$\zeta_{\lambda}$", mu=0, sd=sd, shape=nf)
    xl = pm.Normal(name=r"$\xi_{\lambda}$", mu=0, sd=sd, shape=nr)
    lambda_ = pm.Deterministic(
        r"$\lambda$",
        tt.dot(np.asarray(biggamma), zl) + tt.dot(np.asarray(bigtheta), xl),
    )
    l = pm.Deterministic("$l$", pm.math.sigmoid(lambda_))
    pm.Deterministic(r"$\Lambda_{\lambda}$", zl[1] - zl[0])

    # s.d. of sensory noise, s
    zs = pm.Normal(name=r"$\zeta_{\varsigma}$", mu=0, sd=sd, shape=nf)
    xs = pm.Normal(name=r"$\xi_{\varsigma}$", mu=0, sd=sd, shape=nr)
    varsigma = pm.Deterministic(
        r"$\varsigma$",
        tt.dot(np.asarray(biggamma), zs) + tt.dot(np.asarray(bigtheta), xs),
    )
    s = pm.Deterministic("$s$", tt.exp(varsigma))
    pm.Deterministic(r"$\Lambda_{\varsigma}$", zs[1] - zs[0])

    # rate of increase of memory-noise variance, d
    zd = pm.Normal(name=r"$\zeta_{\delta}$", mu=0, sd=sd, shape=nf)
    xd = pm.Normal(name=r"$\xi_{\delta}$", mu=0, sd=sd, shape=nr)
    delta = pm.Deterministic(
        r"$\delta$", tt.dot(np.asarray(biggamma), zd) + tt.dot(np.asarray(bigtheta), xd)
    )
    d = pm.Deterministic("$d$", tt.exp(delta))
    pm.Deterministic(r"$\Lambda_{\delta}$", zd[1] - zd[0])

    # predicted response probabilities
    num = data.delta.values - b
    den = tt.sqrt((2 * s ** 2) + (d * (data.isi + 0.1)))
    phi = pm.invprobit(num / den)
    p = pm.Deterministic("p", l * a + (1 - l) * phi)

    # prior on data
    pm.Binomial(name="y", p=p, n=data.trials.values, observed=data.num.values)

    return data


def modelb():
    """Constructs the first secondary model. This model is identical to the first except
    that trials where Delta=0 are removed and treated as missing.

    Returns:
        data (pandas.DataFrame): A formatted copy of the data.

    """
    # load the data
    _data = pd.read_csv("data.csv")
    kwargs = dict(values="response", index=("listener", "condition", "isi", "delta"))
    data = pd.pivot_table(_data, aggfunc=len, **kwargs).reset_index()
    cols = data.columns.tolist()
    cols[4] = "trials"
    data.columns = cols
    _s = pd.pivot_table(_data, aggfunc=sum, **kwargs).reset_index().response
    data["num"] = _s
    data["prop"] = data.num / data.trials

    # remove trials where Delta=0
    data.loc[data[data.delta == 0].index, ["num", "prop"]] = np.nan

    # design matrix for fixed effects
    biggamma = dmatrix("0 + C(condition)", data)
    feffs = biggamma.design_info.column_names
    nf = len(feffs)

    # design matrix for random effects
    bigtheta = dmatrix("0 + C(listener)", data)
    reffs = bigtheta.design_info.column_names
    nr = len(reffs)

    # two effects per row; this ensures induced s.d. = 1
    sd = 1 / np.sqrt(2)

    # preference during lapse, a
    za = pm.Normal(name=r"$\zeta_{\alpha}$", mu=0, sd=sd, shape=nf)
    xa = pm.Normal(name=r"$\xi_{\alpha}$", mu=0, sd=sd, shape=nr)
    alpha = pm.Deterministic(
        r"$\alpha$", tt.dot(np.asarray(biggamma), za) + tt.dot(np.asarray(bigtheta), xa)
    )
    a = pm.Deterministic("$a$", pm.math.sigmoid(alpha))
    pm.Deterministic(r"$\Lambda_{\alpha}$", za[1] - za[0])

    # bias, b
    zb = pm.Normal(name=r"$\zeta_{\beta}$", mu=0, sd=sd, shape=nf)
    xb = pm.Normal(name=r"$\xi_{\beta}$", mu=0, sd=sd, shape=nr)
    beta = pm.Deterministic(
        r"$\beta$", tt.dot(np.asarray(biggamma), zb) + tt.dot(np.asarray(bigtheta), xb)
    )
    b = pm.Deterministic("$b$", beta)
    pm.Deterministic(r"$\Lambda_{\beta}$", zb[1] - zb[0])

    # lapse rate, l
    zl = pm.Normal(name=r"$\zeta_{\lambda}$", mu=0, sd=sd, shape=nf)
    xl = pm.Normal(name=r"$\xi_{\lambda}$", mu=0, sd=sd, shape=nr)
    lambda_ = pm.Deterministic(
        r"$\lambda$",
        tt.dot(np.asarray(biggamma), zl) + tt.dot(np.asarray(bigtheta), xl),
    )
    l = pm.Deterministic("$l$", pm.math.sigmoid(lambda_))
    pm.Deterministic(r"$\Lambda_{\lambda}$", zl[1] - zl[0])

    # s.d. of sensory noise, s
    zs = pm.Normal(name=r"$\zeta_{\varsigma}$", mu=0, sd=sd, shape=nf)
    xs = pm.Normal(name=r"$\xi_{\varsigma}$", mu=0, sd=sd, shape=nr)
    varsigma = pm.Deterministic(
        r"$\varsigma$",
        tt.dot(np.asarray(biggamma), zs) + tt.dot(np.asarray(bigtheta), xs),
    )
    s = pm.Deterministic("$s$", tt.exp(varsigma))
    pm.Deterministic(r"$\Lambda_{\varsigma}$", zs[1] - zs[0])

    # rate of increase of memory-noise variance, d
    zd = pm.Normal(name=r"$\zeta_{\delta}$", mu=0, sd=sd, shape=nf)
    xd = pm.Normal(name=r"$\xi_{\delta}$", mu=0, sd=sd, shape=nr)
    delta = pm.Deterministic(
        r"$\delta$", tt.dot(np.asarray(biggamma), zd) + tt.dot(np.asarray(bigtheta), xd)
    )
    d = pm.Deterministic("$d$", tt.exp(delta))
    pm.Deterministic(r"$\Lambda_{\delta}$", zd[1] - zd[0])

    # predicted response probabilities
    num = data.delta.values - b
    den = tt.sqrt((2 * s ** 2) + (d * (data.isi + 0.1)))
    phi = pm.invprobit(num / den)
    p = pm.Deterministic("p", l * a + (1 - l) * phi)

    # prior on data
    pm.Binomial(name="y", p=p, n=data.trials.values, observed=data.num)

    return data

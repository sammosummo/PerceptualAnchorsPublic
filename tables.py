"""Makes tables.

"""
import pandas as pd
import pymc3 as pm

from scipy.stats import norm, skewnorm

from models import fit_model

interps = [
    "-",
    "Listeners were not more biased",
    "Greater rate of increase in memory noise per s",
    "Listeners were more likely to lapse",
    "Less sensory noise"
]


def latexify(x, p=3):
    """Convert a float to a LaTeX-formatted string.

    Values are formatted to p significant digits and in standard form if very large or
    snall.

    Args:
        x (float): Value.
        p (:obj:`int`, optional): Number of significant digits. Default is 3.

    Return:
        s (str): Formatted value.

    """

    def _f(x, p):
        s = "{:g}".format(float("{:.{p}g}".format(x, p=3)))
        n = p - len(s.replace(".", "").replace("-", "").lstrip("3"))
        s += "3" * n
        return s

    if abs(x) < 10 ** -p or abs(x) > 10 ** (p + 1):
        a, b = str("%e" % x).split("e")
        return "$%s$" % r"%s \times 10^{%i}" % (_f(float(a), p), int(b))

    return "$%s$" % _f(x, p)


def interpret(bf):
    """Interpret a Bayes factor.

    Args:
        bf (float): The Bayes factor.

    Returns:
        interpretation (str): Verbal label given to the Baves factor according to Lee
        and Wagenmakers (2009; Table 7.1).

    """
    if bf > 100:
        return "Extreme"
    elif 30 < bf <= 100:
        return "Very strong"
    elif 10 < bf <= 30:
        return "Strong"
    elif 3 < bf <= 10:
        return "Moderate"
    elif 1 < bf <= 3:
        return "Anecdotal"
    elif bf == 1:
        return "No evidence"
    elif 1/3 < bf < 1:
        return "Anecdotal"
    elif 1/10 < bf <= 1/3:
        return "Moderate"
    elif 1/30 < bf <= 1/10:
        return "Strong"
    elif 1/100 < bf <= 1/30:
        return "Very strong"
    else:
        return "Extreme"


def table2(name, func, tex):
    """Makes table 2.

    Args:
        name (str): Descriptive name of the model. Posterior samples, statistics, and
            figures are generated and saved in a subdirectory with this name.
        func (:obj:`<class 'function'>): Function for model construction. Should
            return a formatted copy of the data.
        tex (bool): If True, saves the table to the manuscript subdirectory.

    """

    with pm.Model() as m:
        fit_model(name, func)
        trace = pm.load_trace(name)
        params = sorted([p.name for p in m.deterministics if "Lambda" in p.name])
        df = pm.summary(trace, var_names=params)

    table = []
    for p, i in zip(params, interps):

            theta = skewnorm.fit(trace[p])
            p0 = norm.pdf(0)
            p1 = skewnorm.pdf(0, *theta)
            bf = p0 / p1
            a, b, c = df.loc[p, ["mean", "hpd_2.5", "hpd_97.5"]]

            dic = {
                "Variable": p,
                "Posterior mean (95% HPD)": "%s (%s, %s)" % (
                    latexify(a), latexify(b), latexify(c)),
                "During roved-frequency trials ...": i,
                "BF": latexify(bf),
                "Evidence": interpret(bf),
            }
            table.append(dic)
            # print(p, bf)

    df = pd.DataFrame(table)[dic.keys()]
    df.to_latex(f"{name}/table2.tex", escape=False, index=False)

    if tex is True:
        df.to_latex("manuscript/table2.tex", escape=False, index=False)


def table3(name, func, tex):
    """Makes table 3.

    Args:
        name (str): Descriptive name of the model. Posterior samples, statistics, and
            figures are generated and saved in a subdirectory with this name.
        func (:obj:`<class 'function'>): Function for model construction. Should
            return a formatted copy of the data.
        tex (bool): If True, saves the table to the manuscript subdirectory.

    """
    with pm.Model():
        data = fit_model(name, func)

    df = data.groupby(["listener", "condition"])[list("abdls")].mean().reset_index()
    df = df.pivot(
        index="listener", columns="condition", values=list("abdls")
    ).reset_index()
    # df = df.T.sort_values(["condition"]).T.set_index("listener")
    df = df.set_index("listener")
    df.loc["Group mean"] = df.mean(axis=0)

    df = df.applymap(latexify)
    df.to_latex(f"{name}/table3.tex", escape=False)

    if tex is True:
        df.to_latex("manuscript/table3.tex", escape=False)

"""Figure functions.

"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pymc3 as pm

from scipy.stats import norm, skewnorm
from matplotlib import rcParams
from matplotlib.patches import ArrowStyle

from models import fit_model


def set_fig_defaults():
    """Make figures look nice for JASA.

    """
    rcParams["figure.figsize"] = (3, 3 * 0.6)
    rcParams["font.size"] = 10
    rcParams["font.sans-serif"] = "Arial"
    rcParams["font.family"] = "sans-serif"
    rcParams["xtick.direction"] = "in"
    rcParams["xtick.top"] = True
    rcParams["ytick.direction"] = "in"
    rcParams["ytick.right"] = True
    rcParams["figure.subplot.wspace"] = 0.025
    rcParams["figure.subplot.hspace"] = 0.025 * 2
    rcParams["legend.borderpad"] = 0
    rcParams["legend.handletextpad"] = 0.2
    rcParams["legend.columnspacing"] = 0.5


def fig12(data, name, eps):
    """Make figures 1 and 2. This can be done separately for different models.

    Args:
        data (pandas.DataFrame): Data returned from a specific model after fitting.
        name (str): Name of the model. Versions of the figures in .png format are saved
            in this subdirectory.
        eps (bool): If True, saves the figures to the manuscript subdirectory in .eps
            format.
    """
    n = data.listener.nunique()
    m = data.isi.nunique()

    set_fig_defaults()
    rcParams["figure.figsize"] = (7, 7.7)

    figures = [plt.figure(), plt.figure()]

    for i, (ix, df) in enumerate(data.groupby(["listener", "isi"]), 1):

        listener, isi = ix
        isi = int(round(isi))
        soa = isi + 0.1

        for (ix_, sdf), fig in zip(df.groupby("condition"), figures):

            ax = fig.add_subplot(n, m, i)
            ax.set_ylim(-0.1, 1.1)
            if i <= 3:
                ax.set_title(
                    f"ISI = {isi} s; SOA = {soa} s", fontsize=rcParams["font.size"]
                )
            if i != 1:
                plt.setp(ax.get_yticklabels(), visible=False)
            else:
                ax.set_ylabel("Prop. 2nd")
            if i != 30:
                plt.setp(ax.get_xticklabels(), visible=False)
            else:
                ax.set_xlabel(r"$\Delta$ (semitones)")
            if i % 3 == 1:
                ax.text(-1, 1, listener, verticalalignment="top")

            ax.plot(sdf.delta, sdf.prop, "ko", fillstyle="none")
            x = np.linspace(sdf.delta.min(), sdf.delta.max())
            row = sdf.iloc[0]
            l, a, b, s, d = row[list("labsd")]
            num = x - b
            den = np.sqrt((2 * s ** 2) + (d * (isi + 0.1)))
            phi = norm.cdf(num / den)
            p = l * a + (1 - l) * phi
            ax.plot(x, p, "k")
            ax.fill_between(sdf.delta, sdf.ppc_lo, sdf.ppc_hi, fc="#D3D3D3", zorder=-10)

    for i, fig in enumerate(figures, 1):

        f = f"{name}/fig{i}.png"
        fig.savefig(f, bbox_inches="tight")

        if eps is True:
            f = f"manuscript/fig{i}.eps"
            fig.savefig(f, bbox_inches="tight")


def fig3(data, name, eps):
    """Make figure 3.

    Args:
        data (pandas.DataFrame): Data returned from a specific model after fitting.
        name (str): Name of the model. Versions of the figures in .png format are saved
            in this subdirectory.
        eps (bool): If True, saves the figures to the manuscript subdirectory in .eps
            format.
    """

    n = data.listener.nunique()
    data["soa"] = data.isi + 0.1
    data["m2"] = data.d * data.soa
    data["s2"] = data.s ** 2
    data["sigma2"] = 2 * data.s2 + data.m2
    isi = data.isi.unique()
    lsd = {"fixed": "--", "roved": "-"}
    cd = {"fixed": "grey", "roved": "lightgrey"}

    set_fig_defaults()
    rcParams["figure.figsize"] = (3, 3 * 2)

    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, constrained_layout=True)
    groupy = data.groupby(["condition", "isi"]).mean()

    ax0.plot(isi, groupy.loc["fixed", "s2"], "ko", ls="--", fillstyle="none")
    ax0.plot(isi, groupy.loc["roved", "s2"], "ko", ls="-", fillstyle="full")
    plt.setp(ax0.get_xticklabels(), visible=False)
    ax0.set_ylabel(r"$s^2$ (semitones$^2$)")

    ax1.plot(isi, groupy.loc["fixed", "m2"], "ko", ls="--", fillstyle="none")
    ax1.plot(isi, groupy.loc["roved", "m2"], "ko", ls="-", fillstyle="full")
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.set_ylabel(r"$m^2$ (semitones$^2$)")

    ax2.plot(
        isi,
        groupy.loc["fixed", "sigma2"],
        "ko",
        ls="--",
        fillstyle="none",
        label="Fixed",
    )
    ax2.plot(
        isi,
        groupy.loc["roved", "sigma2"],
        "ko",
        ls="-",
        fillstyle="full",
        label="Roved",
    )
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax2.set_ylabel(r"$\sigma^2$ (semitones$^2$)")
    ax2.set_xlabel(r"ISI (s)")
    ax2.legend(frameon=False)

    for (listener, condition), df in data.groupby(["listener", "condition"]):

        y = df.groupby(["isi"]).mean()
        z = -10 if condition == "roved" else -9
        ax0.plot(isi, y.s2, c=cd[condition], ls=lsd[condition], zorder=z)
        ax1.plot(isi, y.m2, c=cd[condition], ls=lsd[condition], zorder=z)
        ax2.plot(isi, y.sigma2, c=cd[condition], ls=lsd[condition], zorder=z)

    fig.savefig(f"{name}/fig5.png")

    if eps is True:
        fig.savefig("manuscript/fig3.eps")


def fig4(name, func, eps):
    """Makes figure 4.

    Args:
        name (str): Descriptive name of the model. Posterior samples, statistics, and
            figures are generated and saved in a subdirectory with this name.
        func (:obj:`<class 'function'>): Function for model construction. Should
            return a formatted copy of the data.
        eps (bool): If True, saves the figures to the manuscript subdirectory in .eps
            format.

    """

    with pm.Model() as m:

        fit_model(name, func)
        trace = pm.load_trace(name)
        params = sorted([p.name for p in m.deterministics if "Lambda" in p.name])

    set_fig_defaults()
    rcParams["figure.figsize"] = (3, 3 * 2)
    fig, axes = plt.subplots(5, 1, constrained_layout=True)

    for p, ax in zip(params, axes):

        vals, bins, _ = ax.hist(
            trace[p], bins=50, density=True, histtype="step", color="lightgray"
        )
        ax.set_xlabel(p)
        if ax == axes[0]:
            ax.set_ylabel("Posterior density")

        start, stop = pm.stats.hpd(trace[p])
        for n, l, r in zip(vals, bins, bins[1:]):

            if l > start:
                if r < stop:
                    ax.fill_between([l, r], 0, [n, n], color="lightgray")
                elif l < stop < r:
                    ax.fill_between([l, stop], 0, [n, n], color="lightgray")
            elif l < start < r:
                ax.fill_between([start, r], 0, [n, n], color="lightgray")

        x = np.linspace(min([bins[0], 0]), max([0, bins[-1]]))
        theta = skewnorm.fit(trace[p])
        ax.plot(x, skewnorm.pdf(x, *theta), "k", label="Normal approx.")
        ax.plot(x, norm.pdf(x), "k--", label="Prior")
        ax.plot([0, 0], [skewnorm.pdf(0, *theta), norm.pdf(0)], "ko")

    fig.savefig(f"{name}/fig4.png")

    if eps is True:
        fig.savefig("manuscript/fig4.eps")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.optimize import minimize, approx_fprime
from scipy.special import voigt_profile
from math import lgamma
import re

from glob import glob

from matplotlib import rcParams
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from XRD_fits import compute_ratio_with_error




##### Note, I asked claude to format these plots, need to go through and make sure it is all correct....



# ── colour palette  ────────────
_C1 = "#CC6677"  # rose – Peak 1 / A₁
_C2 = "#332288"  # indigo – Peak 2 / A₂
_C3 = "#117733"  # green – ratio / delta (single-series panels)

# ── rc context applied to every figure we create ────────────────────
_RC = {
    "font.family":        "serif",
    "mathtext.fontset":   "cm",
    "font.size":          9,
    "axes.labelsize":     10,
    "axes.titlesize":     10,
    "axes.linewidth":     0.6,
    "axes.edgecolor":     "#333333",
    "xtick.labelsize":    8.5,
    "ytick.labelsize":    8.5,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "xtick.top":          True,
    "ytick.right":        True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size":   4,
    "ytick.major.size":   4,
    "xtick.minor.size":   2,
    "ytick.minor.size":   2,
    "xtick.major.width":  0.6,
    "ytick.major.width":  0.6,
    "xtick.minor.width":  0.4,
    "ytick.minor.width":  0.4,
    "legend.fontsize":    8,
    "legend.frameon":     False,
    "legend.handlelength": 1.4,
    "grid.linewidth":     0.4,
    "grid.alpha":         0.30,
    "lines.linewidth":    1.2,
    "lines.markersize":   5,
    "errorbar.capsize":   2.5,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
    "figure.dpi":         150,
}

# Common errorbar styling
_ERR_KW = dict(lw=1.0, capthick=0.7, elinewidth=0.7)

# ── helpers ──────────────────────────────────────────────────────────

def _prep_summary_df(summary_fit_df, stress_col="stress", sort=True):
    d = summary_fit_df.copy()
    d = d.dropna(subset=[stress_col])
    if sort:
        d = d.sort_values(stress_col)
    return d


def _label_panel(ax, label, x=-0.12, y=1.05):
    """Place a bold (a)/(b)/… label in axes coordinates."""
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=11, fontweight="bold",
        va="bottom", ha="right",
    )


# ── individual panels ───────────────────────────────────────────────

def plot_peak_locations_vs_stress(
    summary_fit_df,
    stress_col="stress",
    use_profile_errors=True,
    ax=None,
):
    d = _prep_summary_df(summary_fit_df, stress_col)

    if ax is None:
        fig, ax = plt.subplots(figsize=(3.4, 2.8))

    if use_profile_errors:
        yerr1 = np.vstack([d["x0_errm_prof"], d["x0_errp_prof"]])
        yerr2 = np.vstack([d["x2_errm_prof"], d["x2_errp_prof"]])
    else:
        yerr1 = d["x0_err_hess"]
        yerr2 = d["x2_err_hess"]

    ax.errorbar(
        d[stress_col], d["x0"], yerr=yerr1,
        fmt="o", color=_C1, mec=_C1, mfc="white",
        label="Peak 1", zorder=3, **_ERR_KW,
    )
    ax.errorbar(
        d[stress_col], d["x2"], yerr=yerr2,
        fmt="s", color=_C2, mec=_C2, mfc="white",
        label="Peak 2", zorder=3, **_ERR_KW,
    )

    ax.set_xlabel("Film stress (MPa)")
    ax.set_ylabel(r"Peak position, $2\theta$ (°)")
    ax.legend(loc="best")
    ax.grid(True, which="major", ls="--")
    return ax


def plot_amplitude_ratio_vs_stress(
    summary_fit_df,
    stress_col="stress",
    ax=None,
    logy=False,
):
    d = _prep_summary_df(summary_fit_df, stress_col)

    if ax is None:
        fig, ax = plt.subplots(figsize=(3.4, 2.8))

    R, dR = compute_ratio_with_error(d)

    ax.errorbar(
        d[stress_col], R, yerr=dR,
        fmt="o", color=_C3, mec=_C3, mfc="white",
        zorder=3, **_ERR_KW,
    )

    ax.set_xlabel("Film stress (MPa)")
    ax.set_ylabel(r"$A_2\,/\,A_1$")
    if logy:
        ax.set_yscale("log")
    ax.grid(True, which="major", ls="--")
    return ax


def plot_amplitudes_vs_stress(
    summary_fit_df,
    stress_col="stress",
    ax=None,
    logy=True,
):
    d = _prep_summary_df(summary_fit_df, stress_col)

    if ax is None:
        fig, ax = plt.subplots(figsize=(3.4, 2.8))

    ax.plot(
        d[stress_col], d["A1"],
        "o", color=_C1, mec=_C1, mfc="white",
        label=r"$A_1$", zorder=3,
    )
    ax.plot(
        d[stress_col], d["A2"],
        "s", color=_C2, mec=_C2, mfc="white",
        label=r"$A_2$", zorder=3,
    )

    ax.set_xlabel("Film stress (MPa)")
    ax.set_ylabel("Peak amplitude (arb. u.)")
    if logy:
        ax.set_yscale("log")
    ax.grid(True, which="both", ls="--")
    ax.legend(loc="best")
    return ax


def plot_delta_vs_stress(
    summary_fit_df,
    stress_col="stress",
    ax=None,
    use_profile_errors=True,
):
    d = _prep_summary_df(summary_fit_df, stress_col)

    if ax is None:
        fig, ax = plt.subplots(figsize=(3.4, 2.8))

    if use_profile_errors:
        yerr = np.vstack([d["delta_errm_prof"], d["delta_errp_prof"]])
    else:
        yerr = d["delta_err_hess"]

    ax.errorbar(
        d[stress_col], d["delta"], yerr=yerr,
        fmt="o", color=_C3, mec=_C3, mfc="white",
        zorder=3, **_ERR_KW,
    )

    ax.set_xlabel("Film stress (MPa)")
    ax.set_ylabel(r"Peak separation $\Delta 2\theta$ (°)")
    ax.grid(True, which="major", ls="--")
    return ax


# ── composite figure ────────────────────────────────────────────────

def plot_xrd_fit_summary(
    summary_fit_df,
    stress_col="stress",
    use_profile_errors=True,
):
    """Four-panel summary figure, publication-ready at 180 mm wide."""

    with plt.rc_context(_RC):
        fig, axs = plt.subplots(
            1, 4,
            figsize=(7.08, 2.4),      # ~180 mm wide (common journal width)
            constrained_layout=True,
        )

        plot_peak_locations_vs_stress(
            summary_fit_df, stress_col=stress_col,
            use_profile_errors=use_profile_errors, ax=axs[0],
        )
        plot_amplitude_ratio_vs_stress(
            summary_fit_df, stress_col=stress_col, ax=axs[1],
        )
        plot_amplitudes_vs_stress(
            summary_fit_df, stress_col=stress_col, ax=axs[2],
        )
        plot_delta_vs_stress(
            summary_fit_df, stress_col=stress_col,
            use_profile_errors=use_profile_errors, ax=axs[3],
        )

        # Panel labels
        for ax, lbl in zip(axs, ["(a)", "(b)", "(c)", "(d)"]):
            _label_panel(ax, lbl)

    return fig, axs


# ── weighted linear fit utility ─────────────────────────────────────

def weighted_linear_fit(x, y, yerr):
    """Weighted least-squares straight line.  Returns slope, intercept and
    their 1-σ uncertainties."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)

    w = 1.0 / yerr**2
    X = np.vstack([x, np.ones_like(x)]).T
    W = np.diag(w)

    cov = np.linalg.inv(X.T @ W @ X)
    beta = cov @ (X.T @ W @ y)
    slope, intercept = beta
    slope_err = np.sqrt(cov[0, 0])
    intercept_err = np.sqrt(cov[1, 1])

    return slope, intercept, slope_err, intercept_err


# ── area-fraction plot ──────────────────────────────────────────────

_C4 = "#882255"  # wine – area fraction (single-series, distinct from _C3)

def plot_area_fraction_vs_stress(
    df,
    use_profile_errors=True,
    ax=None,
    label=None,
    show_fit=True,
):
    """
    Plot ω-phase area fraction vs film stress with optional weighted
    linear fit overlay.

    Parameters
    ----------
    df : DataFrame
        Must contain A1, A2, stress, and error columns (see code).
    use_profile_errors : bool
        Prefer profile-likelihood errors over Hessian.
    ax : matplotlib Axes, optional
    label : str, optional
    show_fit : bool
        Overlay a weighted linear fit and print results.

    Returns
    -------
    ax : matplotlib Axes
    slope: slope of fit
    intercept: intercept of fit
    """
    d = _prep_summary_df(df, stress_col="stress")

    A1 = d["A1"].values
    A2 = d["A2"].values
    frac = A2 / (A1 + A2)

    # ── error propagation ───────────────────────────────────────────
    if "A2_over_A1_err" in d.columns:
        R  = d["A2_over_A1"].values
        dR = d["A2_over_A1_err"].values
        frac_err = np.abs(1.0 / (1.0 + R) ** 2) * dR

    elif {"A1_err_hess", "A2_err_hess", "A1_A2_cov"}.issubset(d.columns):
        var = np.empty(len(d))
        for i, (_, row) in enumerate(d.iterrows()):
            a1, a2 = row["A1"], row["A2"]
            s1, s2, cov12 = (
                row["A1_err_hess"],
                row["A2_err_hess"],
                row["A1_A2_cov"],
            )
            denom = (a1 + a2) ** 2
            df_dA1 = -a2 / denom
            df_dA2 =  a1 / denom
            var[i] = (
                df_dA1**2 * s1**2
                + df_dA2**2 * s2**2
                + 2 * df_dA1 * df_dA2 * cov12
            )
        frac_err = np.sqrt(np.abs(var))
    else:
        frac_err = np.full_like(frac, np.nan)

    # ── weighted fit ────────────────────────────────────────────────
    if show_fit:
        slope, intercept, slope_err, intercept_err = weighted_linear_fit(
            d["stress"].values, frac, frac_err,
        )
        #print(f"slope     = {slope:.4e} ± {slope_err:.4e}")
        #print(f"intercept = {intercept:.4f} ± {intercept_err:.4f}")

    # ── plot ────────────────────────────────────────────────────────
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.4, 2.8))

    ax.errorbar(
        d["stress"], frac, yerr=frac_err,
        fmt="o", color=_C4, mec=_C4, mfc="white",
        zorder=3, label=label, **_ERR_KW,
    )

    if show_fit:
        xfit = np.linspace(d["stress"].min(), d["stress"].max(), 200)
        ax.plot(
            xfit, slope * xfit + intercept,
            color="k", ls="--", lw=0.9, zorder=2,
            label=(
                f"Fit: slope = {slope:.2e}"
                + r"$\,\pm\,$"
                + f"{slope_err:.2e}"
            ),
        )

    ax.set_xlabel("Film stress (MPa)")
    ax.set_ylabel(r"Area fraction ($\omega$ phase)")
    ax.grid(True, which="major", ls="--")
    if label is not None or show_fit:
        ax.legend(loc="best")
    return ax, slope, intercept


# ── delta vs stress with weighted fit ───────────────────────────────

def plot_delta_vs_stress_weighted(
    df,
    ax=None,
    use_profile_errors=False,
):
    """
    Peak separation vs stress with a weighted linear fit overlay.

    Parameters
    ----------
    df : DataFrame
    ax : matplotlib Axes, optional
    use_profile_errors : bool
        If True, uses delta_errm_prof / delta_errp_prof (symmetrised).
        Default False → delta_err_hess.
    """
    d = _prep_summary_df(df, stress_col="stress")

    x = d["stress"].values
    y = d["delta"].values

    if use_profile_errors:
        yerr = 0.5 * (d["delta_errm_prof"].values + d["delta_errp_prof"].values)
    else:
        yerr = d["delta_err_hess"].values

    slope, intercept, slope_err, intercept_err = weighted_linear_fit(
        x, y, yerr,
    )

    xfit = np.linspace(x.min(), x.max(), 200)
    yfit = slope * xfit + intercept

    # ── plot ────────────────────────────────────────────────────────
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.4, 2.8))

    ax.errorbar(
        x, y, yerr=yerr,
        fmt="o", color=_C3, mec=_C3, mfc="white",
        zorder=3, label="Data", **_ERR_KW,
    )
    ax.plot(
        xfit, yfit,
        color="k", ls="--", lw=0.9, zorder=2,
        label=(
            f"Fit: slope = {slope:.2e}"
            + r"$\,\pm\,$"
            + f"{slope_err:.2e}"
        ),
    )

    ax.set_xlabel("Film stress (MPa)")
    ax.set_ylabel(r"Peak separation $\Delta 2\theta$ (°)")
    ax.grid(True, which="major", ls="--")
    ax.legend(loc="best")

    print("Weighted fit results:")
    print(f"  slope     = {slope:.4e} ± {slope_err:.4e}")
    print(f"  intercept = {intercept:.4f} ± {intercept_err:.4f}")

    return ax, (slope, intercept, slope_err, intercept_err)



    

def plot_area_fraction_vs_stress(
    df,
    use_profile_errors=True,
    ax=None,
    label=None,
):
    """
    Plot omega phase area fraction vs stress.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain:
            A1, A2
            sigma (optional but used if widths differ later)
            x0_err*, delta_err* (for error propagation)
            stress

    use_profile_errors : bool
        Use profile errors (recommended) instead of Hessian

    ax : matplotlib axis (optional)

    label : str (optional)
    """

    import numpy as np
    import matplotlib.pyplot as plt

    d = df.copy().sort_values("stress")

    # ---------------------------
    # Area fraction
    # ---------------------------
    # Since widths are shared, area ∝ amplitude → simplifies nicely
    A1 = d["A1"].values
    A2 = d["A2"].values

    frac = A2 / (A1 + A2)

    # ---------------------------
    # Error propagation
    # ---------------------------
    if "A2_over_A1_err" in d.columns:
        # Convert ratio error → fraction error
        R = d["A2_over_A1"].values
        dR = d["A2_over_A1_err"].values

        # f = R / (1+R)
        # df/dR = 1 / (1+R)^2
        frac_err = np.abs(1 / (1 + R)**2) * dR

    elif {"A1_err_hess", "A2_err_hess", "A1_A2_cov"}.issubset(d.columns):
        # full propagation from amplitudes

        var = []

        for _, row in d.iterrows():
            A1 = row["A1"]
            A2 = row["A2"]

            s1 = row["A1_err_hess"]
            s2 = row["A2_err_hess"]
            cov = row["A1_A2_cov"]

            denom = (A1 + A2)**2

            df_dA1 = -A2 / denom
            df_dA2 = A1 / denom

            var_f = (
                (df_dA1**2) * s1**2 +
                (df_dA2**2) * s2**2 +
                2 * df_dA1 * df_dA2 * cov
            )

            var.append(var_f)

        frac_err = np.sqrt(np.abs(np.array(var)))

    else:
        frac_err = np.full_like(frac, np.nan)

    slope, intercept, slope_err, intercept_err = weighted_linear_fit(d['stress'], frac, frac_err)

    print(f'slope={slope}+/-{slope_err}')
    print(f'slope={intercept}+/-{intercept_err}')
    # ---------------------------
    # Plot
    # ---------------------------
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))

    ax.errorbar(
        d["stress"],
        frac,
        yerr=frac_err,
        fmt="o",
        ms=7,
        mec="k",
        mew=0.8,
        capsize=4,
        elinewidth=1.5,
        label=label,
    )

    plt.plot(d['stress'], slope*d['stress']+intercept, color='k', linestyle='--') 

    ax.set_xlabel("Film Stress (MPa)", fontsize=14)
    ax.set_ylabel("Area Fraction (ω phase)", fontsize=14)
    ax.set_title("ω-phase Fraction vs Film Stress", fontsize=13)

    ax.grid(True, alpha=0.25)

    if label is not None:
        ax.legend()

    return ax
def weighted_linear_fit(x, y, yerr):
    w = 1.0 / (yerr**2)

    # Weighted least squares
    X = np.vstack([x, np.ones_like(x)]).T
    W = np.diag(w)

    beta = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ y)
    slope, intercept = beta

    # uncertainties
    cov = np.linalg.inv(X.T @ W @ X)
    slope_err = np.sqrt(cov[0,0])
    intercept_err = np.sqrt(cov[1,1])

    return slope, intercept, slope_err, intercept_err




def plot_delta_vs_stress_weighted(
    df,
    ax=None,
    use_profile_errors=False,
):
    d = _prep_summary_df(df, stress_col="stress")

    x = d["stress"].values
    y = d["delta"].values

    if use_profile_errors:
        yerr = 0.5 * (d["delta_errm_prof"].values + d["delta_errp_prof"].values)
    else:
        yerr = d["delta_err_hess"].values

    slope, intercept, slope_err, intercept_err = weighted_linear_fit(x, y, yerr)

    xfit = np.linspace(x.min(), x.max(), 200)
    yfit = slope * xfit + intercept

    if ax is None:
        fig, ax = plt.subplots(figsize=(3.4, 2.8))

    ax.errorbar(
        x, y, yerr=yerr,
        fmt="o", color=_C3, mec=_C3, mfc="white",
        zorder=3, label="Data", **_ERR_KW,
    )
    ax.plot(
        xfit, yfit,
        color="k", ls="--", lw=0.9, zorder=2,
        label=(
            f"Fit: slope = {slope:.2e}"
            + r"$\,\pm\,$"
            + f"{slope_err:.2e}"
        ),
    )

    ax.set_xlabel("Film stress (MPa)")
    ax.set_ylabel(r"Peak separation $\Delta 2\theta$ (°)")
    ax.grid(True, which="major", ls="--")
    ax.legend(loc="best")

    return ax, (slope, intercept, slope_err, intercept_err)
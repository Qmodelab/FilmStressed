import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, approx_fprime
from scipy.special import voigt_profile
from math import lgamma
import re

from glob import glob

from matplotlib import rcParams
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_ratio_with_error(df):
    d = df.copy()

    if "A2_over_A1" in d.columns:
        R = d["A2_over_A1"]
    else:
        R = d["A2"] / d["A1"]

    if "A2_over_A1_err" in d.columns:
        dR = d["A2_over_A1_err"]
    elif {"A1_err_hess", "A2_err_hess", "A1_A2_cov"}.issubset(d.columns):
        A1 = d["A1"]
        A2 = d["A2"]

        var_R = (
            (A2 / A1**2)**2 * d["A1_err_hess"]**2
            + (1 / A1)**2 * d["A2_err_hess"]**2
            - 2 * (A2 / A1**3) * d["A1_A2_cov"]
        )

        dR = np.sqrt(np.abs(var_R))
    else:
        dR = np.full(len(d), np.nan)

    return R, dR
    
def fits_to_dataframe(fits, pressures=None):
    """
    Convert a list of PeakFitXRD objects into a tidy summary dataframe.

    Parameters
    ----------
    fits : list[PeakFitXRD]
    pressures : dict or None
        Mapping like {'JT1': -724, 'JT2': -158, ...}

    Returns
    -------
    pd.DataFrame
    """
    rows = []

    for fit in fits:
        row = {
            "label": fit.label,
            "A1": fit.A1,
            "A2": fit.A2,
            "x0": fit.x0,
            "x2": fit.x2,
            "sigma": fit.sigma,
            "gamma": fit.gamma,
            "delta": fit.delta,
            "c0": fit.c0,
            "delta_AIC": fit.delta_AIC,
            "delta_BIC": fit.delta_BIC,
            "fit_success_one": None if fit.fit1 is None else bool(fit.fit1["res"].success),
            "fit_success_two": None if fit.fit2 is None else bool(fit.fit2["res"].success),
        }

        # Hessian errors
        if fit.hess_errs is not None:
            row.update({
                "x0_err1_hess": fit.hess_errs.get("x0_err1_hess", np.nan),
                "x0_err2_hess": fit.hess_errs.get("x0_err2_hess", np.nan),
                "delta_err_hess": fit.hess_errs.get("delta_err_hess", np.nan),
                "x2_err_hess": fit.hess_errs.get("x2_err_hess", np.nan),
            })
        else:
            row.update({
                "x0_err1_hess": np.nan,
                "x0_err2_hess": np.nan,
                "delta_err_hess": np.nan,
                "x2_err_hess": np.nan,
            })
        
        # Amplitude errors and ratio errors
        if getattr(fit, "amp_errs", None) is not None:
            row.update({
                "A1_err_hess": fit.amp_errs.get("A1_err_hess", np.nan),
                "A2_err_hess": fit.amp_errs.get("A2_err_hess", np.nan),
                "A1_A2_cov": fit.amp_errs.get("A1_A2_cov", np.nan),
                "A2_over_A1": fit.amp_errs.get("A2_over_A1", np.nan),
                "A2_over_A1_err": fit.amp_errs.get("A2_over_A1_err", np.nan),
            })
        else:
            row.update({
                "A1_err_hess": np.nan,
                "A2_err_hess": np.nan,
                "A1_A2_cov": np.nan,
                "A2_over_A1": np.nan if (fit.A1 is None or fit.A1 == 0) else fit.A2 / fit.A1,
                "A2_over_A1_err": np.nan,
            })
        # Profile errors
        if fit.prof_x0 is not None:
            row.update({
                "x0_errm_prof": fit.prof_x0.get("errm", np.nan),
                "x0_errp_prof": fit.prof_x0.get("errp", np.nan),
            })
        else:
            row.update({
                "x0_errm_prof": np.nan,
                "x0_errp_prof": np.nan,
            })

        if fit.prof_delta is not None:
            row.update({
                "delta_errm_prof": fit.prof_delta.get("errm", np.nan),
                "delta_errp_prof": fit.prof_delta.get("errp", np.nan),
            })
        else:
            row.update({
                "delta_errm_prof": np.nan,
                "delta_errp_prof": np.nan,
            })

        row.update({
            "x2_errm_prof": getattr(fit, "x2_errm_prof", np.nan),
            "x2_errp_prof": getattr(fit, "x2_errp_prof", np.nan),
        })

        # # Derived quantities
        # row["A2_over_A1"] = np.nan if (fit.A1 is None or fit.A1 == 0) else fit.A2 / fit.A1 #no longer using, getting from full covariance now

        rows.append(row)

    df = pd.DataFrame(rows)

    if pressures is not None:
        df["stress"] = df["label"].map(pressures)

    return df

def filter_good_fits(df, require_two_peak_success=True, require_negative_delta_aic=False):
    """
    Return a filtered copy of the summary dataframe.
    """
    out = df.copy()

    if require_two_peak_success and "fit_success_two" in out.columns:
        out = out[out["fit_success_two"] == True]

    if require_negative_delta_aic and "delta_AIC" in out.columns:
        out = out[out["delta_AIC"] < 0]

    if "stress" in out.columns:
        out = out.dropna(subset=["stress"])

    return out


class PeakFitXRD:
    """
    Constrained XRD peak fitter:
      - one-peak Voigt + constant background
      - two-peak Voigt model with shared widths and x2 = x0 + delta
      - Poisson likelihood
      - Hessian covariance
      - profile likelihoods for A2, delta, x0
    """

    # ============================================================
    # Construction
    # ============================================================

    def __init__(
        self,
        df=None,
        mask=None,
        x=None,
        counts=None,
        label=None,
        delta_grid=(0.005, 0.01, 0.02, 0.03, 0.05),
        A2_fracs=(0.01, 0.03, 0.05, 0.1, 0.2),
        hessian_eps=1e-4,
        profile_npts=120,
        profile_span=0.05,
    ):
        if x is None or counts is None:
            if df is None or mask is None:
                raise ValueError("Provide either (df, mask) or (x, counts).")
            self.x = np.asarray(df.x[mask], dtype=float)
            self.counts = np.asarray(df.y[mask], dtype=float)
        else:
            self.x = np.asarray(x, dtype=float)
            self.counts = np.asarray(counts, dtype=float)

        self.label = label
        self.delta_grid = tuple(delta_grid)
        self.A2_fracs = tuple(A2_fracs)
        self.hessian_eps = hessian_eps
        self.profile_npts = profile_npts
        self.profile_span = profile_span

        self.bounds1, self.bounds2 = self._get_bounds()

        # storage
        self.fit1 = None
        self.fit2 = None
        self.cov1 = None
        self.cov2 = None
        self.hess_errs = None
        self.amp_errs = None
        self.prof_A2 = None
        self.prof_delta = None
        self.prof_x0 = None
        self.components = None
        self.residuals = None
        self.x2_errm_prof = np.nan
        self.x2_errp_prof = np.nan

    # ============================================================
    # Static helpers
    # ============================================================

    @staticmethod
    def poisson_nll_from_mu(counts, mu):
        mu = np.clip(mu, 1e-12, None)
        return np.sum(mu - counts * np.log(mu) + np.array([lgamma(int(n) + 1) for n in counts]))

    @staticmethod
    def background_const(x, c0):
        return np.full_like(x, c0)

    @staticmethod
    def voigt_peak(x, A, x0, sigma, gamma):
        return A * voigt_profile(x - x0, sigma, gamma)

    @staticmethod
    def numerical_hessian(fun, p, eps=1e-4):
        p = np.asarray(p, dtype=float)
        n = len(p)
        H = np.zeros((n, n))
        for i in range(n):
            dp = np.zeros(n)
            dp[i] = eps
            g1 = approx_fprime(p + dp, fun, eps)
            g2 = approx_fprime(p - dp, fun, eps)
            H[:, i] = (g1 - g2) / (2 * eps)
        return 0.5 * (H + H.T)

    @staticmethod
    def safe_covariance_from_hessian(H):
        try:
            cov = np.linalg.inv(H)
            method = "inv"
        except np.linalg.LinAlgError:
            cov = np.linalg.pinv(H)
            method = "pinv"
        return cov, method

    @staticmethod
    def profile_interval(xgrid, dprof, level=1.0):
        imin = np.argmin(dprof)
        xhat = xgrid[imin]

        left = np.nan
        for i in range(imin - 1, -1, -1):
            if dprof[i] > level and dprof[i + 1] <= level:
                x1, x2 = xgrid[i], xgrid[i + 1]
                y1, y2 = dprof[i], dprof[i + 1]
                left = x1 + (level - y1) * (x2 - x1) / (y2 - y1)
                break

        right = np.nan
        for i in range(imin, len(dprof) - 1):
            if dprof[i] <= level and dprof[i + 1] > level:
                x1, x2 = xgrid[i], xgrid[i + 1]
                y1, y2 = dprof[i], dprof[i + 1]
                right = x1 + (level - y1) * (x2 - x1) / (y2 - y1)
                break

        return xhat, left, right, xhat - left, right - xhat

    @staticmethod
    def aic_bic(nll, k, n):
        AIC = 2 * k + 2 * nll
        BIC = k * np.log(n) + 2 * nll
        return AIC, BIC

    # ============================================================
    # Models
    # ============================================================

    def model_one_peak(self, x, A1, x0, sigma, gamma, c0):
        return self.voigt_peak(x, A1, x0, sigma, gamma) + self.background_const(x, c0)

    def nll_one_peak(self, p, x=None, counts=None):
        if x is None:
            x = self.x
        if counts is None:
            counts = self.counts
        A1, x0, sigma, gamma, c0 = p
        mu = self.model_one_peak(x, A1, x0, sigma, gamma, c0)
        return self.poisson_nll_from_mu(counts, mu)

    def model_two_peak_stable(self, x, A1, x0, sigma, gamma, A2, delta, c0):
        return (
            self.voigt_peak(x, A1, x0, sigma, gamma)
            + self.voigt_peak(x, A2, x0 + delta, sigma, gamma)
            + self.background_const(x, c0)
        )

    def nll_two_peak_stable(self, p, x=None, counts=None):
        if x is None:
            x = self.x
        if counts is None:
            counts = self.counts
        A1, x0, sigma, gamma, A2, delta, c0 = p
        mu = self.model_two_peak_stable(x, A1, x0, sigma, gamma, A2, delta, c0)
        return self.poisson_nll_from_mu(counts, mu)

    # ============================================================
    # Core setup helpers
    # ============================================================

    def _get_bounds(self):
        xmin, xmax = np.min(self.x), np.max(self.x)
        xrange = xmax - xmin

        bounds1 = [
            (1e-8, None),      # A1
            (xmin, xmax),      # x0
            (1e-4, 1.0),       # sigma
            (1e-4, 1.0),       # gamma
            (0.0, None),       # c0
        ]

        bounds2 = [
            (1e-8, None),            # A1
            (xmin, xmax),            # x0
            (1e-4, 1.0),             # sigma
            (1e-4, 1.0),             # gamma
            (0.0, None),             # A2
            (1e-4, 0.25 * xrange),   # delta > 0
            (0.0, None),             # c0
        ]
        return bounds1, bounds2

    def _get_one_peak_init(self):
        baseline_guess = max(np.percentile(self.counts, 10), 1e-6)
        imax = np.argmax(self.counts)
        xpeak = self.x[imax]

        return np.array([
            max(self.counts.max() - baseline_guess, 1.0),
            xpeak,
            0.05,
            0.05,
            baseline_guess,
        ])

    # ============================================================
    # Fit methods
    # ============================================================

    def fit_one_peak(self):
        p0_1 = self._get_one_peak_init()
        res1 = minimize(self.nll_one_peak, p0_1, bounds=self.bounds1, method="L-BFGS-B")
        popt1 = res1.x
        mu1 = self.model_one_peak(self.x, *popt1)
        AIC1, BIC1 = self.aic_bic(res1.fun, len(popt1), len(self.counts))

        self.fit1 = {
            "res": res1,
            "popt": popt1,
            "mu": mu1,
            "AIC": AIC1,
            "BIC": BIC1,
            "p0": p0_1,
        }
        return self.fit1

    def fit_two_peak_multistart(self):
        if self.fit1 is None:
            self.fit_one_peak()

        A1_hat, x0_hat, sigma_hat, gamma_hat, c0_hat = self.fit1["popt"]

        best_res2 = None
        best_nll2 = np.inf
        best_p02 = None

        for d0 in self.delta_grid:
            for frac in self.A2_fracs:
                p0_2 = np.array([
                    max(A1_hat, 1e-6),
                    x0_hat,
                    sigma_hat,
                    gamma_hat,
                    frac * A1_hat,
                    d0,
                    c0_hat,
                ])

                try:
                    res2_try = minimize(
                        self.nll_two_peak_stable,
                        p0_2,
                        bounds=self.bounds2,
                        method="L-BFGS-B"
                    )
                except Exception:
                    continue

                if res2_try.success and res2_try.fun < best_nll2:
                    best_nll2 = res2_try.fun
                    best_res2 = res2_try
                    best_p02 = p0_2.copy()

        if best_res2 is None:
            raise RuntimeError(f"{self.label}: No successful two-peak fit found.")

        popt2 = best_res2.x
        mu2 = self.model_two_peak_stable(self.x, *popt2)
        AIC2, BIC2 = self.aic_bic(best_res2.fun, len(popt2), len(self.counts))

        self.fit2 = {
            "res": best_res2,
            "popt": popt2,
            "mu": mu2,
            "AIC": AIC2,
            "BIC": BIC2,
            "best_init": best_p02,
        }
        return self.fit2

    def compute_covariances(self):
        if self.fit1 is None:
            self.fit_one_peak()
        if self.fit2 is None:
            self.fit_two_peak_multistart()

        H1 = self.numerical_hessian(self.nll_one_peak, self.fit1["popt"], eps=self.hessian_eps)
        cov1, method1 = self.safe_covariance_from_hessian(H1)

        H2 = self.numerical_hessian(self.nll_two_peak_stable, self.fit2["popt"], eps=self.hessian_eps)
        cov2, method2 = self.safe_covariance_from_hessian(H2)

        self.cov1 = {"H": H1, "cov": cov1, "method": method1}
        self.cov2 = {"H": H2, "cov": cov2, "method": method2}
        return self.cov1, self.cov2

    def compute_hessian_errors(self):
        if self.cov1 is None or self.cov2 is None:
            self.compute_covariances()

        cov1 = self.cov1["cov"]
        cov2 = self.cov2["cov"]

        x0_err1_hess = np.sqrt(np.abs(cov1[1, 1]))
        x0_err2_hess = np.sqrt(np.abs(cov2[1, 1]))
        delta_err_hess = np.sqrt(np.abs(cov2[5, 5]))

        cov_x0_delta = cov2[1, 5]
        x2_var_hess = cov2[1, 1] + cov2[5, 5] + 2.0 * cov_x0_delta
        x2_err_hess = np.sqrt(np.abs(x2_var_hess))

        self.hess_errs = {
            "x0_err1_hess": x0_err1_hess,
            "x0_err2_hess": x0_err2_hess,
            "delta_err_hess": delta_err_hess,
            "x2_err_hess": x2_err_hess,
        }
        return self.hess_errs
    
    
    def compute_amplitude_errors(self):
        if self.cov2 is None:
            self.compute_covariances()

        cov = self.cov2["cov"]

        A1 = self.A1
        A2 = self.A2

        A1_err = np.sqrt(np.abs(cov[0, 0]))
        A2_err = np.sqrt(np.abs(cov[4, 4]))
        A1_A2_cov = cov[0, 4]

        if A1 is None or A1 == 0:
            R = np.nan
            R_err = np.nan
        else:
            R = A2 / A1

            var_R = (
                (A2 / A1**2)**2 * A1_err**2
                + (1 / A1)**2 * A2_err**2
                - 2 * (A2 / A1**3) * A1_A2_cov
            )

            R_err = np.sqrt(np.abs(var_R))

        self.amp_errs = {
            "A1_err_hess": A1_err,
            "A2_err_hess": A2_err,
            "A1_A2_cov": A1_A2_cov,
            "A2_over_A1": R,
            "A2_over_A1_err": R_err,
        }

        return self.amp_errs

    # ============================================================
    # Profile methods
    # ============================================================

    def _run_profile_scan(self, grid, make_objective, q0, bounds):
        prof = []
        for fixed_val in grid:
            obj_red = make_objective(fixed_val)
            r = minimize(obj_red, q0, bounds=bounds, method="L-BFGS-B")
            prof.append(r.fun)
        prof = np.asarray(prof)
        dprof = 2 * (prof - prof.min())
        return prof, dprof

    def compute_profile_A2(self):
        if self.fit2 is None:
            self.fit_two_peak_multistart()

        popt2 = self.fit2["popt"]
        A2_hat = popt2[4]
        A2_grid = np.linspace(0.0, max(2.5 * A2_hat, 1e-6), self.profile_npts)

        q0 = [popt2[0], popt2[1], popt2[2], popt2[3], popt2[5], popt2[6]]
        bnds = [self.bounds2[0], self.bounds2[1], self.bounds2[2], self.bounds2[3], self.bounds2[5], self.bounds2[6]]

        def make_objective(A2_fix):
            def obj_red(q):
                p = np.array([q[0], q[1], q[2], q[3], A2_fix, q[4], q[5]])
                return self.nll_two_peak_stable(p)
            return obj_red

        prof, dprof = self._run_profile_scan(A2_grid, make_objective, q0, bnds)

        self.prof_A2 = {
            "grid": A2_grid,
            "prof": prof,
            "dprof": dprof,
            "hat": A2_hat,
        }
        return self.prof_A2

    def compute_profile_delta(self):
        if self.fit2 is None:
            self.fit_two_peak_multistart()

        popt2 = self.fit2["popt"]
        delta_hat = popt2[5]
        delta_grid = np.linspace(
            max(self.bounds2[5][0], delta_hat - self.profile_span),
            min(self.bounds2[5][1], delta_hat + self.profile_span),
            self.profile_npts
        )

        q0 = [popt2[0], popt2[1], popt2[2], popt2[3], popt2[4], popt2[6]]
        bnds = [self.bounds2[0], self.bounds2[1], self.bounds2[2], self.bounds2[3], self.bounds2[4], self.bounds2[6]]

        def make_objective(delta_fix):
            def obj_red(q):
                p = np.array([q[0], q[1], q[2], q[3], q[4], delta_fix, q[5]])
                return self.nll_two_peak_stable(p)
            return obj_red

        prof, dprof = self._run_profile_scan(delta_grid, make_objective, q0, bnds)
        hat, left, right, errm, errp = self.profile_interval(delta_grid, dprof, level=1.0)

        self.prof_delta = {
            "grid": delta_grid,
            "prof": prof,
            "dprof": dprof,
            "hat": hat,
            "left": left,
            "right": right,
            "errm": errm,
            "errp": errp,
        }
        return self.prof_delta

    def compute_profile_x0(self):
        if self.fit2 is None:
            self.fit_two_peak_multistart()

        popt2 = self.fit2["popt"]
        x0_hat = popt2[1]
        x0_grid = np.linspace(
            max(self.bounds2[1][0], x0_hat - self.profile_span),
            min(self.bounds2[1][1], x0_hat + self.profile_span),
            self.profile_npts
        )

        q0 = [popt2[0], popt2[2], popt2[3], popt2[4], popt2[5], popt2[6]]
        bnds = [self.bounds2[0], self.bounds2[2], self.bounds2[3], self.bounds2[4], self.bounds2[5], self.bounds2[6]]

        def make_objective(x0_fix):
            def obj_red(q):
                p = np.array([q[0], x0_fix, q[1], q[2], q[3], q[4], q[5]])
                return self.nll_two_peak_stable(p)
            return obj_red

        prof, dprof = self._run_profile_scan(x0_grid, make_objective, q0, bnds)
        hat, left, right, errm, errp = self.profile_interval(x0_grid, dprof, level=1.0)

        self.prof_x0 = {
            "grid": x0_grid,
            "prof": prof,
            "dprof": dprof,
            "hat": hat,
            "left": left,
            "right": right,
            "errm": errm,
            "errp": errp,
        }
        return self.prof_x0

    def compute_all_profiles(self):
        self.compute_profile_A2()
        self.compute_profile_delta()
        self.compute_profile_x0()

        if np.isfinite(self.prof_x0["errm"]) and np.isfinite(self.prof_delta["errm"]):
            self.x2_errm_prof = np.sqrt(self.prof_x0["errm"]**2 + self.prof_delta["errm"]**2)
        else:
            self.x2_errm_prof = np.nan

        if np.isfinite(self.prof_x0["errp"]) and np.isfinite(self.prof_delta["errp"]):
            self.x2_errp_prof = np.sqrt(self.prof_x0["errp"]**2 + self.prof_delta["errp"]**2)
        else:
            self.x2_errp_prof = np.nan

        return self.prof_A2, self.prof_delta, self.prof_x0

    # ============================================================
    # Derived quantities
    # ============================================================

    def compute_components(self):
        if self.fit1 is None:
            self.fit_one_peak()
        if self.fit2 is None:
            self.fit_two_peak_multistart()

        popt1 = self.fit1["popt"]
        popt2 = self.fit2["popt"]

        comp1 = self.voigt_peak(self.x, popt1[0], popt1[1], popt1[2], popt1[3])
        bkg1 = self.background_const(self.x, popt1[4])

        A1, x0, sigma, gamma, A2, delta, c0 = popt2
        comp2_1 = self.voigt_peak(self.x, A1, x0, sigma, gamma)
        comp2_2 = self.voigt_peak(self.x, A2, x0 + delta, sigma, gamma)
        bkg2 = self.background_const(self.x, c0)

        self.components = {
            "comp1": comp1,
            "bkg1": bkg1,
            "comp2_1": comp2_1,
            "comp2_2": comp2_2,
            "bkg2": bkg2,
        }
        return self.components

    def compute_residuals(self):
        if self.fit1 is None:
            self.fit_one_peak()
        if self.fit2 is None:
            self.fit_two_peak_multistart()

        r1 = (self.counts - self.fit1["mu"]) / np.sqrt(np.clip(self.fit1["mu"], 1e-12, None))
        r2 = (self.counts - self.fit2["mu"]) / np.sqrt(np.clip(self.fit2["mu"], 1e-12, None))

        self.residuals = {"r1": r1, "r2": r2}
        return self.residuals

    # ============================================================
    # Workflow
    # ============================================================

    def run_all(self, verbose=False, plot=False):
        self.fit_one_peak()
        self.fit_two_peak_multistart()
        self.compute_covariances()
        self.compute_hessian_errors()
        self.compute_amplitude_errors()
        self.compute_all_profiles()
        self.compute_components()
        self.compute_residuals()
        

        if verbose:
            self.print_summary()

        if plot:
            self.plot_diagnostics()

        return self

    # ============================================================
    # Accessors
    # ============================================================

    @property
    def A1(self):
        return None if self.fit2 is None else self.fit2["popt"][0]

    @property
    def x0(self):
        return None if self.fit2 is None else self.fit2["popt"][1]

    @property
    def sigma(self):
        return None if self.fit2 is None else self.fit2["popt"][2]

    @property
    def gamma(self):
        return None if self.fit2 is None else self.fit2["popt"][3]

    @property
    def A2(self):
        return None if self.fit2 is None else self.fit2["popt"][4]

    @property
    def delta(self):
        return None if self.fit2 is None else self.fit2["popt"][5]

    @property
    def c0(self):
        return None if self.fit2 is None else self.fit2["popt"][6]

    @property
    def x2(self):
        return None if self.fit2 is None else self.x0 + self.delta

    @property
    def delta_AIC(self):
        return None if (self.fit1 is None or self.fit2 is None) else self.fit2["AIC"] - self.fit1["AIC"]

    @property
    def delta_BIC(self):
        return None if (self.fit1 is None or self.fit2 is None) else self.fit2["BIC"] - self.fit1["BIC"]

    # ============================================================
    # Reporting / plotting
    # ============================================================

    def print_summary(self):
        if self.fit1 is None or self.fit2 is None:
            print("Fit has not been run yet.")
            return

        print(f"\n===== {self.label}: one-peak model =====")
        print("  success =", self.fit1["res"].success)
        print("  NLL     =", self.fit1["res"].fun)
        print("  AIC     =", self.fit1["AIC"])
        print("  BIC     =", self.fit1["BIC"])
        print("  params  =", self.fit1["popt"])

        print(f"\n===== {self.label}: stable two-peak model =====")
        print("  success   =", self.fit2["res"].success)
        print("  best init =", self.fit2["best_init"])
        print("  NLL       =", self.fit2["res"].fun)
        print("  AIC       =", self.fit2["AIC"])
        print("  BIC       =", self.fit2["BIC"])
        print("  params    =", self.fit2["popt"])
        print(f"  ΔAIC(two-one) = {self.delta_AIC:.2f}")
        print(f"  ΔBIC(two-one) = {self.delta_BIC:.2f}")

        if self.cov1 is not None and self.cov2 is not None:
            print("\n  covariance methods:", self.cov1["method"], self.cov2["method"])

    def plot_diagnostics(self):
        if self.components is None:
            self.compute_components()
        if self.residuals is None:
            self.compute_residuals()
        if self.prof_A2 is None or self.prof_delta is None:
            self.compute_all_profiles()

        fig, axs = plt.subplots(2, 3, figsize=(16, 8))

        axs[0,0].step(self.x, self.counts, color='k', lw=1, label='data')
        axs[0,0].plot(self.x, self.fit1["mu"], '--', lw=2, color='b', label='one-peak fit')
        axs[0,0].plot(self.x, self.fit2["mu"], '-', lw=2, color='r', label='stable two-peak fit')
        axs[0,0].set_yscale('log')
        axs[0,0].set_xlabel(r'2$\theta$')
        axs[0,0].set_ylabel('Counts')
        axs[0,0].set_title(f'{self.label}: one-peak vs constrained two-peak')
        axs[0,0].legend()

        axs[0,1].step(self.x, self.counts, color='k', lw=1, label='data')
        axs[0,1].plot(self.x, self.fit2["mu"], lw=2, color='r', label='total fit')
        axs[0,1].plot(self.x, self.components["comp2_1"], '--', lw=2, color='purple', label='peak 1')
        axs[0,1].plot(self.x, self.components["comp2_2"], ':', lw=2, color='green', label='peak 2')
        axs[0,1].plot(self.x, self.components["bkg2"], '-.', lw=2, color='salmon', label='background')
        axs[0,1].set_yscale('log')
        axs[0,1].set_xlabel(r'2$\theta$')
        axs[0,1].set_ylabel('Counts')
        axs[0,1].set_title('Constrained two-peak components')
        axs[0,1].legend()

        axs[0,2].axhline(0, color='k', lw=1)
        axs[0,2].plot(self.x, self.residuals["r1"], 'x', ms=4, color='b', label='one-peak residuals')
        axs[0,2].plot(self.x, self.residuals["r2"], 'o', ms=4, color='r', label='two-peak residuals')
        axs[0,2].set_xlabel(r'2$\theta$')
        axs[0,2].set_ylabel(r'$(n_i-\mu_i)/\sqrt{\mu_i}$')
        axs[0,2].set_title('Poisson standardized residuals')
        axs[0,2].legend()

        axs[1,0].plot(self.prof_A2["grid"], self.prof_A2["dprof"], color='k', lw=2)
        axs[1,0].axhline(1, color='0.6', ls=':', label=r'$1\sigma$')
        axs[1,0].axvline(self.prof_A2["hat"], color='C1', ls='--', alpha=0.7, label=r'$\hat A_2$')
        axs[1,0].set_xlabel(r'second peak amplitude $A_2$')
        axs[1,0].set_ylabel(r'$-2\Delta\log L$')
        axs[1,0].set_title('Profile likelihood for $A_2$')
        axs[1,0].legend()

        axs[1,1].plot(self.prof_delta["grid"], self.prof_delta["dprof"], color='k', lw=2)
        axs[1,1].axhline(1, color='0.6', ls=':', label=r'$1\sigma$')
        axs[1,1].axvline(self.delta, color='C1', ls='--', alpha=0.7, label=r'$\hat\Delta$')
        axs[1,1].set_xlabel(r'peak separation $\Delta$')
        axs[1,1].set_ylabel(r'$-2\Delta\log L$')
        axs[1,1].set_title('Profile likelihood for separation')
        axs[1,1].legend()

        axs[1,2].axis('off')
        txt = (
            f"One-peak AIC = {self.fit1['AIC']:.2f}\n"
            f"Two-peak AIC = {self.fit2['AIC']:.2f}\n"
            f"ΔAIC = {self.delta_AIC:.2f}\n\n"
            f"One-peak BIC = {self.fit1['BIC']:.2f}\n"
            f"Two-peak BIC = {self.fit2['BIC']:.2f}\n"
            f"ΔBIC = {self.delta_BIC:.2f}\n\n"
            f"Best-fit parameters:\n"
            f"x0 = {self.x0:.5f}\n"
            f"Δ  = {self.delta:.5f}\n"
            f"A2 = {self.A2:.3g}\n"
        )
        axs[1,2].text(0.03, 0.97, txt, va='top', fontsize=11)

        plt.tight_layout()
        plt.show()

    def summary_dict(self):
        return {
            "label": self.label,
            "x": self.x,
            "counts": self.counts,
            "fit1": self.fit1,
            "fit2": self.fit2,
            "cov1": self.cov1,
            "cov2": self.cov2,
            "hess_errs": self.hess_errs,
            "amp_errs": self.amp_errs,
            "prof_A2": self.prof_A2,
            "prof_delta": self.prof_delta,
            "prof_x0": self.prof_x0,
            "A1": self.A1,
            "x0": self.x0,
            "sigma": self.sigma,
            "gamma": self.gamma,
            "A2": self.A2,
            "delta": self.delta,
            "c0": self.c0,
            "x2": self.x2,
            "x2_errm_prof": self.x2_errm_prof,
            "x2_errp_prof": self.x2_errp_prof,
            "components": self.components,
            "residuals": self.residuals,
            "delta_AIC": self.delta_AIC,
            "delta_BIC": self.delta_BIC,
        }
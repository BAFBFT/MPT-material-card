import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Callable, Dict, Any

from scipy.interpolate import interp1d
from scipy.optimize import least_squares


# -----------------------------
# Utilities
# -----------------------------

def macaulay(x: float) -> float:
    """<x> = max(x, 0)."""
    return x if x > 0.0 else 0.0


@dataclass
class Params:
    E: float
    B: float
    A: float
    C: float
    k: float
    K: float
    n1: float
    n2: float
    rho0: float
    R0: float


# -----------------------------
# Constitutive model RHS
# -----------------------------

def rhs(
    t: float,
    state: np.ndarray,
    epsT_of_t: Callable[[float], float],
    p: Params,
    rho_floor: float = 1e-12,
) -> np.ndarray:
    """
    state = [eps_p, rho, R]
    returns d/dt [eps_p, rho, R] using equations (1)-(4)
    """
    eps_p, rho, R = state
    rho = max(rho, rho_floor)

    epsT = float(epsT_of_t(t))
    sigma = p.E * (epsT - eps_p)

    # Eq (1): viscoplastic flow rule with yield gate
    drive = (sigma - R - p.k) / p.K
    eps_p_dot = macaulay(drive) ** p.n1

    # Eq (3): dislocation density evolution
    rho_dot = p.A * (1.0 - rho) * eps_p_dot - p.C * (rho ** p.n2)

    # Eq (2): isotropic hardening evolution
    R_dot = 0.5 * p.B * (rho ** -0.5) * rho_dot

    return np.array([eps_p_dot, rho_dot, R_dot], dtype=float)


# -----------------------------
# RK4 Integrator (vector state)
# -----------------------------

def rk4_step(
    t: float,
    y: np.ndarray,
    dt: float,
    f: Callable[[float, np.ndarray], np.ndarray],
) -> np.ndarray:
    k1 = f(t, y)
    k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


# -----------------------------
# Forward simulation
# -----------------------------

def simulate_stress(
    t: np.ndarray,
    epsT: np.ndarray,
    p: Params,
    rho_floor: float = 1e-12,
    clamp_rho: bool = True,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Simulate sigma(t) using RK4 with input epsT(t).
    Returns:
      sigma_model (N,)
      debug dict with eps_p, rho, R histories
    """
    t = np.asarray(t, dtype=float)
    epsT = np.asarray(epsT, dtype=float)

    if t.ndim != 1 or epsT.ndim != 1 or len(t) != len(epsT):
        raise ValueError("t and epsT must be 1D arrays of the same length.")

    # Interpolant for epsT(t) at RK4 stage times
    epsT_of_t = interp1d(t, epsT, kind="linear", fill_value="extrapolate", assume_sorted=True)

    # State histories
    N = len(t)
    eps_p_hist = np.zeros(N, dtype=float)
    rho_hist = np.zeros(N, dtype=float)
    R_hist = np.zeros(N, dtype=float)
    sigma_hist = np.zeros(N, dtype=float)

    # initial state
    y = np.array([0.0, p.rho0, p.R0], dtype=float)

    # function handle for RHS with parameters captured
    def f(tt, yy):
        return rhs(tt, yy, epsT_of_t, p, rho_floor=rho_floor)

    # time stepping
    for i in range(N):
        eps_p, rho, R = y
        eps_p_hist[i] = eps_p
        rho_hist[i] = rho
        R_hist[i] = R
        sigma_hist[i] = p.E * (epsT[i] - eps_p)

        if i == N - 1:
            break

        dt = t[i+1] - t[i]
        if dt <= 0:
            raise ValueError("t must be strictly increasing.")

        y = rk4_step(t[i], y, dt, f)

        if clamp_rho:
            # enforce physical bounds if rho is normalised
            y[1] = float(np.clip(y[1], 0.0, 1.0))
        # also guard against negative rho causing issues next step
        y[1] = max(y[1], rho_floor)

    debug = {"eps_p": eps_p_hist, "rho": rho_hist, "R": R_hist}
    return sigma_hist, debug


# -----------------------------
# Parameter fitting
# -----------------------------

def pack_params(x: np.ndarray) -> Params:
    """
    x ordering: [E, B, A, C, k, K, n1, n2, rho0, R0]
    """
    return Params(
        E=float(x[0]),
        B=float(x[1]),
        A=float(x[2]),
        C=float(x[3]),
        k=float(x[4]),
        K=float(x[5]),
        n1=float(x[6]),
        n2=float(x[7]),
        rho0=float(x[8]),
        R0=float(x[9]),
    )


def residuals(
    x: np.ndarray,
    t: np.ndarray,
    epsT: np.ndarray,
    sigma_exp: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    p = pack_params(x)
    sigma_model, _ = simulate_stress(t, epsT, p)

    r = sigma_model - sigma_exp
    if weights is not None:
        r = r * weights
    return r


def fit_params(
    t: np.ndarray,
    epsT: np.ndarray,
    sigma_exp: np.ndarray,
    x0: np.ndarray,
    bounds: Tuple[np.ndarray, np.ndarray],
    weights: np.ndarray | None = None,
) -> Any:
    """
    Bounded least squares fit.
    """
    res = least_squares(
        residuals,
        x0=x0,
        bounds=bounds,
        args=(t, epsT, sigma_exp, weights),
        method="trf",
        max_nfev=200,
        verbose=2,
    )
    return res


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # Replace this with your real data loading.
    # Expect arrays: t (s), epsT (true strain), sigma_exp (true stress)
    # ------------------------------------------------------------
    # Example dummy data (NOT meaningful)
    full_data = pd.read_csv("Constant Strain Rate Data.csv")
    t = np.linspace(0, 10, 501)
    epsT = 0.02 * (t / t.max())  # ramp to 2% strain
    sigma_exp = 200e9 * epsT  # purely elastic dummy (Pa)
    # ------------------------------------------------------------

    # Optional weighting (e.g., relative)
    sigma_min = np.percentile(np.abs(sigma_exp), 10) + 1e-9
    weights = 1.0 / (np.abs(sigma_exp) + sigma_min)

    # Initial guess x0 = [E, B, A, C, k, K, n1, n2, rho0, R0]
    x0 = np.array([
        200e9,   # E (Pa)
        1e9,     # B
        10.0,    # A
        0.1,     # C
        200e6,   # k (Pa)
        50e6,    # K (Pa)
        10.0,    # n1
        1.5,     # n2
        1e-4,    # rho0 (normalised)
        0.0,     # R0 (Pa)
    ], dtype=float)

    # Bounds (tune for your material / units!)
    lb = np.array([
        1e9,     # E
        1e3,     # B
        1e-6,    # A
        1e-8,    # C
        0.0,     # k
        1e-3,    # K
        1.0,     # n1
        0.1,     # n2
        0.0,     # rho0
        0.0,     # R0
    ], dtype=float)

    ub = np.array([
        400e9,   # E
        1e12,    # B
        1e6,     # A
        1e6,     # C
        2e9,     # k
        2e9,     # K
        100.0,   # n1
        10.0,    # n2
        1.0,     # rho0 (normalised)
        2e9,     # R0
    ], dtype=float)

    res = fit_params(t, epsT, sigma_exp, x0, (lb, ub), weights=weights)

    p_fit = pack_params(res.x)
    print("\nFitted params:", p_fit)

    sigma_fit, dbg = simulate_stress(t, epsT, p_fit)
    print("Final RMSE:", np.sqrt(np.mean((sigma_fit - sigma_exp)**2)))
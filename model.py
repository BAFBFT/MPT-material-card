import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# dataset and model params:
full_data = pd.read_csv("Constant Strain Rate Data.csv")
epsT = np.array(full_data["Strain"], dtype=float)

params_by_temp = {
    "200": dict(E=30000.0, k=50.0, K=200.0, n1=1.0, A=5.0, C=0.5, n2=1.5, B=200.0),
    "250": dict(E=28000.0, k=45.0, K=210.0, n1=1.0, A=4.5, C=0.6, n2=1.5, B=190.0),
    "300": dict(E=26000.0, k=40.0, K=220.0, n1=1.0, A=4.0, C=0.7, n2=1.5, B=180.0),
    "350": dict(E=26000.0, k=40.0, K=220.0, n1=1.0, A=4.0, C=0.7, n2=1.5, B=180.0),
    "400": dict(E=26000.0, k=40.0, K=220.0, n1=1.0, A=4.0, C=0.7, n2=1.5, B=180.0),
    "450": dict(E=26000.0, k=40.0, K=220.0, n1=1.0, A=4.0, C=0.7, n2=1.5, B=180.0),
    "500": dict(E=26000.0, k=40.0, K=220.0, n1=1.0, A=4.0, C=0.7, n2=1.5, B=180.0),
    "535": dict(E=26000.0, k=40.0, K=220.0, n1=1.0, A=4.0, C=0.7, n2=1.5, B=180.0)
}

def simulate_sigma_euler(
    epsT: np.ndarray,          # true strain array (monotonic increasing)
    SR: float,                 # strain rate (1/s)
    E: float, k: float, K: float, n1: float,
    A: float, C: float, n2: float,
    B: float,
    eps_p0: float = 0.0,
    rho0: float = 1e-4,
    rho_floor: float = 1e-12,
    n_sub: int = 1,            # set to e.g. 10, 50 if it jumps
):
    """
    Forward Euler on eps_p and rho, with algebraic R = B*sqrt(rho).
    Returns sigma_model, eps_p_hist, rho_hist, R_hist
    All stress-like quantities must be in consistent units (e.g., MPa everywhere).
    """

    # ensure epsT is np.array
    epsT = np.asarray(epsT, dtype=float)

    # number of sigma values to generate
    N = len(epsT)

    sigma = np.zeros(N)
    eps_p = np.zeros(N)
    rho = np.zeros(N)
    R = np.zeros(N)

    # intialise plastic strain as zero
    eps_p[0] = eps_p0

    # why is rho bounded?
    rho[0] = max(rho0, rho_floor)
    R[0] = B * np.sqrt(rho[0])
    sigma[0] = E * (epsT[0] - eps_p[0])

    # sub-stepping routine for stability 
    for i in range(1, N):
        # compute strain step size
        d_epsT = epsT[i] - epsT[i-1]
        if d_epsT <= 0:
            raise ValueError("epsT must be strictly increasing.")
        # compute time step size
        dt = d_epsT / SR

        # substepping (optional but very effective)
        dt_sub = dt / n_sub
        epsT_sub0 = epsT[i-1]

        epp = eps_p[i-1]
        r = rho[i-1]
        
        # if we have instability we can reduce the time step
        for s in range(n_sub):
            # linearly ramp epsT within this interval
            epsT_s = epsT_sub0 + (s + 1) * (d_epsT / n_sub)

            r = max(r, rho_floor)
            R_s = B * np.sqrt(r)
            sigma_s = E * (epsT_s - epp)

            drive = (sigma_s - R_s - k) / K
            epp_dot = max(drive, 0) ** n1

            r_dot = A * (1.0 - r) * epp_dot - C * (r ** n2)

            # Euler update
            epp = epp + epp_dot * dt_sub
            r = r + r_dot * dt_sub

            r = max(r, rho_floor)

        eps_p[i] = epp
        rho[i] = r
        R[i] = B * np.sqrt(rho[i])
        sigma[i] = E * (epsT[i] - eps_p[i])

    return sigma, eps_p, rho, R

## TODO: need to see the results of the parameters 
# can we do a dictionary of dictionaries, each mapping temperatures to params
# would that be too complicated?

if __name__ == "__main__":

    SR = 1.0  # set strain rate
    temps = [c for c in full_data.columns if c != "Strain"]

    for col in temps:
        sigma_exp = full_data[col].to_numpy()

        # mask to ensure we are working with full arrays
        mask = ~np.isnan(sigma_exp)
        epsT = full_data["Strain"].to_numpy()[mask]
        sigma_exp = sigma_exp[mask]

        params_guess = params_by_temp[col]

        sigma_model, eps_p_hist, rho_hist, R_hist = simulate_sigma_euler(
            epsT=epsT,
            SR=SR,
            **params_guess,
            rho0=1e-4,
            n_sub=1   
        )

        plt.title(f"Simulation and experiment plot for {col}°C")
        plt.plot(epsT, sigma_model)
        plt.scatter(epsT, sigma_exp)
        plt.show()

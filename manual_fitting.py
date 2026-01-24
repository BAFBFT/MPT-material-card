import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ipywidgets import interact, FloatSlider, Dropdown, Output
from IPython.display import display
from sklearn.linear_model import LinearRegression

# Load data
full_data = pd.read_csv("Constant Strain Rate Data.csv")
epsT_full = np.array(full_data["Strain"], dtype=float)
SR = 1.0  # this 
temps = [c for c in full_data.columns if c != "Strain"]

param_names = ["E", "k", "K", "n1", "A", "C", "n2", "B"]
temps_sorted = np.array(sorted([float(t) for t in temps]))
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

# numerical scheme 
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
            # linearl
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

# linear regression for fitting
def linear_regression(x, y):
    model = LinearRegression()
    model.fit(x, y)
    intercept = model.intercept_
    grad = model.coef_
    intercept = np.ravel(intercept).item()
    grad = np.ravel(grad).item()
    return float(intercept), float(grad)

# Create an Output widget for the plot
output_model = Output()
output_params = Output()

# Interactive plotting function
def plot_simulation(temp, E, k, K, n1, A, C, n2, B):
    with output_model:
        output_model.clear_output(wait=True)  # Clear previous plot
        sigma_exp = full_data[temp].to_numpy()
        mask = ~np.isnan(sigma_exp)
        epsT = epsT_full[mask]
        sigma_exp = sigma_exp[mask]
        
        sigma_model, _, _, _ = simulate_sigma_euler(
            epsT=epsT,
            SR=SR,
            E=E, k=k, K=K, n1=n1, A=A, C=C, n2=n2, B=B,
            rho0=1e-4,
            n_sub=1
        )
        
        plt.figure(figsize=(5, 3))
        plt.title(f"Simulation and experiment plot for {temp}°C")
        plt.plot(epsT, sigma_model, label='Model', color='blue')
        plt.scatter(epsT, sigma_exp, label='Experiment', color='red', alpha=0.7)
        plt.xlabel('Strain')
        plt.ylabel('Stress (MPa)')
        plt.legend()
        plt.grid(True)
        plt.show()

def plot_regression(temp, E, k, K, n1, A, C, n2, B):
    with output_params:
        output_params.clear_output(wait=True)
        
        
        params_by_temp[temp] = {
            "E": E, "k": k, "K": K, "n1": n1,
            "A": A, "C": C, "n2": n2, "B": B
        }

        # param vs temperature plots
        fig, axes = plt.subplots(4, 2, figsize=(8, 10), sharex=True)
        axes = axes.ravel()
        for ax, name in zip(axes, param_names):
            # select the variable of interest
            y = np.array([params_by_temp[temp][name] for temp in temps])
            # fitting in logspace against, 1/T (Kelvin)
            log_y = np.log(y).reshape(-1, 1)
            T_inv = (1 / (temps_sorted + 273.15)).reshape(-1,1)
            intercept, grad = linear_regression(T_inv, log_y)
            fit = lambda x : grad * x + intercept
            ax.plot(T_inv, fit(T_inv), )
            ax.scatter(T_inv, log_y, c='r', marker="x")
            ax.text(0.1, 0.9, 
                    f"ln({name}) = {intercept:.2f} + {grad:.2f} (1 / t)", 
                    transform=ax.transAxes)
            ax.set_title(f"{name} vs 1 / T (K)")
            ax.set_xlabel("Temperature (C)")
            ax.set_ylabel(name)
            ax.grid(True)

        plt.tight_layout()
        plt.show()

def full_plot(temp, E, k, K, n1, A, C, n2, B):
    plot_simulation(temp, E, k, K, n1, A, C, n2, B)
    plot_regression(temp, E, k, K, n1, A, C, n2, B)

# Create interactive widgets
temp_dropdown = Dropdown(options=temps, value=temps[0], description='Temperature:')
E_slider = FloatSlider(min=1000, max=80000, step=500, value=30000, description='E (MPa):')
k_slider = FloatSlider(min=0, max=200, step=1, value=50, description='k (MPa):')
K_slider = FloatSlider(min=10, max=500, step=1, value=200, description='K (MPa):')
n1_slider = FloatSlider(min=0.01, max=2, step=0.05, value=1.0, description='n1:')
A_slider = FloatSlider(min=0.01, max=5, step=0.05, value=5.0, description='A:')
C_slider = FloatSlider(min=1, max=200, step=1, value=0.5, description='C:')
n2_slider = FloatSlider(min=0.1, max=5, step=0.1, value=1.5, description='n2:')
B_slider = FloatSlider(min=10, max=500, step=1, value=200, description='B (MPa):')

# Display the output widget
display(output_model, output_params)

# Use interact to create the interactive plot
interact(full_plot,
            temp=temp_dropdown,
            E=E_slider, 
            k=k_slider, 
            K=K_slider, 
            n1=n1_slider, 
            A=A_slider, 
            C=C_slider, 
            n2=n2_slider, 
            B=B_slider)
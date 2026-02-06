import numpy as np
from pathlib import Path
from scipy.optimize import least_squares

from dynamin.utils import load_yaml

cwd = Path.cwd()
params_path = cwd / "dynamin" / "config"

params = load_yaml(params_path / "parameters.yaml")

# --- Parameters ---
g = params['firm']['growth']
g_P = 0.02
g_N = g + g_P
delta = params['firm']['depreciation']
r_M = params['bank']['deposit_interest']
r_L = params['bank']['min_capital_ratio'] * params['bank']['interest_risk']
phi = 1 / (params['bank']['loan_years'] + params['simulation']['steps'])
zeta = params['firm']['wage_buffer']
xi = params['kfirm']['excess_output']
nu = params['cfirm']['acceleration']
d0, d1, d2 = params['cfirm']['d0'], params['cfirm']['d1'], params['cfirm']['d2']
kappa = params['bank']['min_capital_ratio']
N_K = params['simulation']['num_kfirms']
N_C = params['simulation']['num_cfirms']

s_K = N_K / (N_K + N_C)
s_C = N_C / (N_K + N_C)

# --- 2. C-firms, Banks, Households (shares) ---
def residuals(x):
    # x = [omega_K, pi_K, e_K, m_K, rho_K, omega_C, pi_C, e_C, m_C, d_C, ell_C, rho_C, m_H, e_B, pi_B, rho_B]
    omega_K, pi_K, e_K, m_K, rho_K, omega_C, pi_C, e_C, m_C, d_C, ell_C, rho_C, m_H, e_B, pi_B, rho_B = x

    res = []
    
    # --- K-firms ---
    res.append(pi_K - (s_K * (1 - delta * xi) + r_M * m_K - omega_K))
    res.append(g_N * e_K - pi_K * (1 - rho_K))
    res.append(m_K - e_K)
    res.append(m_K - zeta * omega_K)
    res.append(g_N * m_K - pi_K * (1 - rho_K))
    
    # --- C-firms ---
    i_C = nu * (g + delta)
    res.append(pi_C - (s_C + r_M * m_C - omega_C - r_L * d_C - delta * nu * s_C))  # profits
    res.append(d_C - (d0 + d1 * g + d2 * (pi_C / s_C)))
    res.append(ell_C - d_C * (phi + g))
    res.append(g_N * e_C - pi_C * (1 - rho_C))  # equity growth
    res.append(m_C + s_C * nu - (d_C + e_C))
    res.append(m_C - zeta * omega_C)           # precautionary deposits
    res.append(g_N * m_C - (pi_C * (1 - rho_C) + ell_C - phi * d_C - i_C + delta * nu * s_C))  # deposit growth

    # --- Banks ---
    m_total = m_K + m_C + m_H
    d_total = d_C 
    res.append(pi_B - (r_L * d_total - r_M * m_total))
    res.append(g_N * e_B - pi_B * (1 - rho_B))
    res.append(e_B - (kappa * d_total))

    # --- Households ---
    y_H = omega_K + omega_C + rho_K * pi_K + rho_C * pi_C + rho_B * pi_B + r_M * m_H

    # Goods market equilibrium: household spending = C-firm output
    e_H = s_C
    res.append(g_N * m_H - (y_H - e_H))

    return sum(np.square(res))

import numpy as np
from scipy.optimize import least_squares

# --- same parameters and residuals function as you already have ---

# initial guess
x0 = np.array([0.4, 0.1, 0.5, 0.5, 0.6, 0.6, 0.2, 0.5, 0.5, 0.9, 0.1, 0.6, 0.5, 0.5, 0.2, 0.6])

lower_bounds = [0]*16
upper_bounds = [1,1,np.inf,np.inf,1,1,1,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,1,1]

tol = 1e-8   # convergence tolerance
max_iter = 100

x_old = x0.copy()
for iteration in range(max_iter):
    sol = least_squares(residuals, x_old, bounds=(lower_bounds, upper_bounds))
    x_new = sol.x

    # check convergence
    if np.max(np.abs(x_new - x_old)) < tol:
        print(f"Converged after {iteration+1} iterations")
        break

    x_old = x_new

else:
    print("Warning: did not converge after max iterations")

# unpack
omega_K, pi_K, e_K, m_K, rho_K, \
omega_C, pi_C, e_C, m_C, d_C, ell_C, rho_C, \
m_H, e_B, pi_B, rho_B = x_new

print("\nK-firms BGP:")
print(f"omega_K: {omega_K:.4f}, pi_K: {pi_K:.4f}, e_K: {e_K:.4f}, m_K: {m_K:.4f}, rho_K: {rho_K:.4f}")

print("\nC-firms BGP:")
print(f"omega_C: {omega_C:.4f}, pi_C: {pi_C:.4f}, e_C: {e_C:.4f}, m_C: {m_C:.4f}, d_C: {d_C:.4f}, ell_C: {ell_C:.4f}, rho_C: {rho_C:.4f}")

print("\nHouseholds BGP")
print(f"m_H: {m_H:.4f}")

print("\nBanks BGP")
print(f"e_B: {e_B:.4f}, pi_B: {pi_B:.4f}, rho_B: {rho_B:.4f}")
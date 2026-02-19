### import libraries
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import fsolve

### import local ###
from dynamin.utils import load_config

def balanced_growth_solution(params: dict) -> NDArray:
    ### growth rates ###
    # real growth rate 
    g = params["firm"]["growth"]
    # inflation
    gP = params["firm"]["inflation"]
    # nominal growth rate 
    gN = g + gP
    
    ### bank parameters ###
    # deposit interest rate 
    rM = params["bank"]["deposit_interest"]
    # loan interest rate 
    rL = params["bank"]["loan_interest"] + gP
    # capital ratio
    kappa = params["bank"]["min_capital_ratio"]

    ### firm parameters ###
    # debt repayment 
    phi = 1 / (params["bank"]["loan_years"] * params["simulation"]["steps"])
    # depreciation rate 
    delta = params["firm"]["depreciation"]
    # capital to output ratio
    nu = params["cfirm"]["acceleration"]
    # capital firm inventory buffer
    xi = params["kfirm"]["excess_output"]

    ### household paremeters ###
    # marginal propensity to consume out of income
    cY = params["household"]["mpc_income"]
    # marginal propensity to consume out of deposits
    cM = params["household"]["mpc_deposits"]

    ### sector shares ###
    # number of cfirms 
    N_C = params["simulation"]["num_cfirms"]
    # number of kfirms 
    N_K = params["simulation"]["num_kfirms"]
    # consumption sector share
    s_C = N_C / (N_C + N_K)
    # capital sector share
    s_K = 1 - s_C

    ### debt behaviour parameters ###
    d0 = params["cfirm"]["d0"]
    d1 = params["cfirm"]["d1"]
    d2 = params["cfirm"]["d2"]

    ### balanced growth core (BGP) ###
    # households 
    omega = (s_C * (cM + gN - rM * (1 - cY))) / (cM + gN * cY - 2 * rM * cY * (1 - cY))
    m_H = (s_C * (1 - cY)) / (cM + gN * cY - 2 * rM * cY * (1 - cY))
    
    # kfirms 
    pi_K = (gN * (s_K * (1 - delta * xi) - omega * s_K)) / (gN - rM)
    
    # cfirms 
    pi_C = (gN * s_C * (s_C * (1 - omega - rM * nu) - (rL - rM) * (d0 + d1 * g))) / (s_C * (gN - rM) + gN * d2 * (rL - rM))
    d = d0 + d1 * g + d2 * (pi_C / s_C)
    m_C = d + pi_C / gN - nu * s_C

    ### derive variables ###
    # household equity 
    e_H = m_H

    # kfirm deposits 
    m_K = pi_K / gN 
    # kfirm equity 
    e_K = m_K 

    # cfirm equity 
    e_C = pi_C / gN 

    # total deposits 
    m = m_H + m_C + m_K

    # bank profits 
    pi_B = rL * d - rM * m
    # bank equity 
    e_B = max(pi_B / gN, kappa * d)
    # bank reserves 
    r = m + e_B - d

    ### total variables ###
    pi = pi_C + pi_K + pi_B
    e = e_C + e_C + e_B + e_H 

    ### print results ###
    print("Households:")
    print(f"- Wage share: omega = {omega:.4f}")
    print(f"- Household deposit ratio: m_H = {m_H:.4f}")

    print("\nK-firms")
    print(f"- K-firm profit share: pi_K = {pi_K:.4f}")
    print(f"- K-firm wage share: omega_K = {(s_K * omega):.4f}")
    print(f"- K-firm deposit ratio: m_K = {m_K:.4f}")
    print(f"- K-firm equity: e_K = {e_K:.4f}")

    print("\nC-firms:")
    print(f"- C-firm profit share: pi_C = {pi_C:.4f}")
    print(f"- C-firm wage share: omega_C = {(s_C * omega):.4f}")
    print(f"- C-firm debt ratio: d = {d:.4f}")
    print(f"- C-firm deposit ratio: m_C = {m_C:.4f}")
    print(f"- C-firm equity ratio: e_C = {e_C:.4f}")

    print("\nBanks:")
    print(f"- Bank profit share: pi_B = {pi_B:.4f}")
    print(f"- Bank equity ratio: e_C = {e_B:.4f}")
    print(f"- Bank reserve ratio: e_C = {r:.4f}")

    print("\nTotal variables")
    print(f"- Wage share: omega = {omega:.4f}")
    print(f"- Profit share: pi = {pi:.4f}")
    print(f"- Debt ratio: d = {d:.4f}")
    print(f"- Deposit ratio: m = {m:.4f}")
    print(f"- Equity ratio: e = {e:.4f}")
    print(f"- Reserve ratio: r = {r:.4f}")
    print(f"- Profit + Wage share = {pi + omega}")

if __name__ == "__main__":
    # loan parameters
    params = load_config("parameters.yaml")
    # run balanced growth path model 
    balanced_growth_solution(params)
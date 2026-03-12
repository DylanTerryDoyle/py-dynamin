import numpy as np
import pandas as pd
import powerlaw as pl
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pathlib import Path
from dynamin.utils import load_config, sql_engine
from analysis.utils import load_macro_data, plot_autocorrelation, plot_cross_correlation, normality_tests, plot_ccdf, annualise_macro_data, minksy_cycle_test

# is this analysis from the examples folder?
is_true = input("Run stylised_facts.py analysis for an example from the examples folder [y/n]: ").lower().startswith("y")

# if so, which example is this
if is_true:
    example_folder = input("What is the example folder called? ")

### Plot Parameters ###

# figure size
x_figsize = 10
y_figsize = x_figsize/2
# fontsize
fontsize = 25
# upper decile 
upper = 0.95
# lower decile
lower = 0.05
# update matplotlib pyplot settings
plt.rcParams.update({
    "font.family": "serif",
    "xtick.labelsize": fontsize,
    "ytick.labelsize": fontsize,
})

### Paths ###

# analysis directory
analysis_path = Path(__file__).parent
# analysis directory 
dynamin_path = Path(analysis_path).parent
# examples directory 
examples_path = dynamin_path / "examples"
# figure path
figure_path = analysis_path / "figures" / "stylised_facts"
# create figure path if it doesn"t exist
figure_path.mkdir(parents=True, exist_ok=True)

### parameters ###

# base parameters
if is_true:
    params = load_config(examples_path / example_folder / "config" / "parameters.yaml")
else:
    params = load_config("parameters.yaml")
# analysis parameters
steps = params["simulation"]["steps"]
num_years = params["simulation"]["years"]
start = params["simulation"]["start"] * steps
middle = int(start + (num_years * steps) / 2)
years = np.linspace(0, num_years, num_years * steps)
# significance level
significance = 0.01

# random simulation 
sim_index = 5
print("\nRandomly selected simulation index for plots: ", sim_index)

# database parameters
if is_true:
    db_params = load_config(examples_path / example_folder / "config" / "database.yaml")
else:
    db_params = load_config("database.yaml")

### SQL engine ###

engine = sql_engine(db_params)

### senarios data ###

scenarios = pd.read_sql_query(
    """
        SELECT * FROM scenarios
        ORDER BY scenario_id ASC
        ;
    """,
    engine
)

base_scenario = scenarios.iloc[0,:]

###-----------###
### load data ###
###-----------###

### empirical US data ###
# path
empirical_data_path = analysis_path/ "empirical_data"
# data
emp_real_gdp = pd.read_csv(empirical_data_path / "GDPC1.csv", index_col=0).to_numpy()
emp_consumption = pd.read_csv(empirical_data_path / "PCECC96.csv", index_col=0).to_numpy()
emp_investment = pd.read_csv(empirical_data_path / "GPDIC1.csv", index_col=0).to_numpy()
emp_employment = pd.read_csv(empirical_data_path / "CE16OV.csv", index_col=0).to_numpy()
emp_unemployment_rate = pd.read_csv(empirical_data_path / "UNRATE.csv", index_col=0).to_numpy()
emp_debt = pd.read_csv(empirical_data_path / "TBSDODNS.csv", index_col=0).to_numpy()
emp_productivity = emp_real_gdp / emp_employment

### model macro data ###

# total macro data
macro_data = load_macro_data(engine, int(base_scenario["scenario_id"]), params)
# randomly selected single simulation 
single_sim_data = macro_data.loc[macro_data["simulation_index"]==sim_index]
# simulated data 
sim_real_gdp = macro_data.pivot(index="simulation_index", columns="time", values="real_gdp").to_numpy()
sim_consumption = macro_data.pivot(index="simulation_index", columns="time", values="real_consumption").to_numpy()
sim_investment = macro_data.pivot(index="simulation_index", columns="time", values="real_investment").to_numpy()
sim_productivity = macro_data.pivot(index="simulation_index", columns="time", values="productivity").to_numpy()
sim_unemployment_rate = macro_data.pivot(index="simulation_index", columns="time", values="unemployment_rate").to_numpy()
sim_debt = macro_data.pivot(index="simulation_index", columns="time", values="debt").to_numpy()

###---------------------------###
### single simulation figures ###
###---------------------------###

print("\nCreating single simulation time-series plots...")

## plot GDP, investment, and consumption ### 

# real GDP and components
plt.figure(figsize=(x_figsize,y_figsize))
# real GDP
plt.plot(years, single_sim_data["real_gdp"], color="k", linewidth=1, label="GDP")
# consumption
plt.plot(years, single_sim_data["real_consumption"], color="k", linestyle="--", linewidth=1.2, label="Cons.")
# investment
plt.plot(years, single_sim_data["real_investment"], color="k", linestyle=":", linewidth=1.5, label="Inv.")
# log y axis
plt.yscale("log")
# recession shaded
for t in single_sim_data["time"].loc[single_sim_data["recession"]==1]:
    i = t - start
    if i > 0 and i < len(years):
        plt.axvspan(years[i-1], years[i], facecolor="0.2", alpha=0.25)
plt.yticks(fontsize=fontsize)
plt.xticks(fontsize=fontsize)
# legend
plt.legend(loc="upper left", fontsize=fontsize)
# save figure
plt.savefig(figure_path / "time_series_outputs", bbox_inches="tight")

### GDP shares ###

# output shares 
plt.figure(figsize=(x_figsize,y_figsize))
# plot debt ratio
plt.plot(years, single_sim_data["debt_ratio"], color="k", linewidth=1, label=r"D/Y")
# plot wage share 
plt.plot(years, single_sim_data["wage_share"], color="k", linestyle="--", linewidth=1, label=r"W/Y")
# plot profit share 
plt.plot(years, single_sim_data["profit_share"], color="k", linestyle=":", linewidth=1, label=r"$\Pi$/Y")
# plot labour productivity growth rate 
plt.axhline(0, color="k", linestyle="--", linewidth=1)
# recession shaded
for t in single_sim_data["time"].loc[single_sim_data["recession"]==1]:
    i = t - start
    if i > 0 and i < len(years):
        plt.axvspan(years[i-1], years[i], facecolor="0.2", alpha=0.25)
# ticks
plt.yticks(fontsize=fontsize)
plt.xticks(fontsize=fontsize)
# legend
plt.legend(loc="lower left", fontsize=fontsize)
# save figure
plt.savefig(figure_path / "time_series_gdp_shares", bbox_inches="tight")

### real GDP growth ###

# real gdp growth
plt.figure(figsize=(x_figsize,y_figsize))
# plot 
plt.plot(years, single_sim_data["rgdp_growth"], color="k", linewidth=1)
# 0 line
plt.axhline(0, color="k", linestyle="--", linewidth=1)
# recession shaded
for t in single_sim_data["time"].loc[single_sim_data["recession"]==1]:
    i = t - start
    if i > 0 and i < len(years):
        plt.axvspan(years[i-1], years[i], facecolor="0.2", alpha=0.25)
# ticks
plt.yticks(fontsize=fontsize)
plt.xticks(fontsize=fontsize)
# save figure
plt.savefig(figure_path / "time_series_rgdp_growth", bbox_inches="tight")

### inflation ###

# figure
plt.figure(figsize=(x_figsize,y_figsize))
# plot
plt.plot(years, single_sim_data["inflation"], color="k", linewidth=1)
plt.axhline(0, color="k", linestyle="--", linewidth=1)
# recession shaded
for t in single_sim_data["time"].loc[single_sim_data["recession"]==1]:
    i = t - start
    if i > 0 and i < len(years):
        plt.axvspan(years[i-1], years[i], facecolor="0.2", alpha=0.25)
# ticks
plt.yticks(fontsize=fontsize)
plt.xticks(fontsize=fontsize)
# save figure
plt.savefig(figure_path / "time_series_inflation", bbox_inches="tight")

### unemployment ###

# figure
plt.figure(figsize=(x_figsize,y_figsize))
# plot
plt.plot(years, single_sim_data["unemployment_rate"], color="k", linewidth=1)
plt.axhline(0, color="k", linestyle="--", linewidth=1)
# recession shaded
for t in single_sim_data["time"].loc[single_sim_data["recession"]==1]:
    i = t - start
    if i > 0 and i < len(years):
        plt.axvspan(years[i-1], years[i], facecolor="0.2", alpha=0.25)
# ticks
plt.yticks(fontsize=fontsize)
plt.xticks(fontsize=fontsize)
# save figure
plt.savefig(figure_path / "time_series_unemployment_rate", bbox_inches="tight")

### credit rate (total debt growth rate) ###

# figure
plt.figure(figsize=(x_figsize,y_figsize))
# plot 
plt.plot(years, single_sim_data["credit_gdp"], color="k", linewidth=1)
plt.axhline(0, color="k", linestyle="--", linewidth=1)
# recession shaded
for t in single_sim_data["time"].loc[single_sim_data["recession"]==1]:
    i = t - start
    if i > 0 and i < len(years):
        plt.axvspan(years[i-1], years[i], facecolor="0.2", alpha=0.25)
# ticks
plt.yticks(fontsize=fontsize)
plt.xticks(fontsize=fontsize)
# save figure
plt.savefig(figure_path / "time_series_credit_gdp", bbox_inches="tight")

###------------------------------###
### macroeconomic stylised facts ###
###------------------------------###

print("\n----------------------------")
print("MACROECONOMIC STYLISED FACTS")
print("----------------------------")

# ### 1. Miskyan Credit Cycles ###

# print("\nMacroeconomic Stylised Facts")

# print("\n1. Minskyan Cycles")

# # annual data 
# annual_data = annualise_macro_data(macro_data, steps=steps)

# # VAR model
# results, minsky_df = minksy_cycle_test(annual_data)

# # eigenvalue distribution plot

# eigenvalues = np.concatenate([results["ev1"].values,results["ev2"].values])

# ### plot of eigenvalue spectrum in unit circle ###

# print("- created eigenvalue spectrum plot")

# plt.figure(figsize=(x_figsize, x_figsize))
# # scatter eigenvalues
# plt.scatter(eigenvalues.real,eigenvalues.imag,color="skyblue",s=40,edgecolors="k")
# # axes lines
# plt.axhline(0, color="k", linewidth=0.5)
# plt.axvline(0, color="k", linewidth=0.5)
# # unit circle
# theta = np.linspace(0, 2*np.pi, 500)
# plt.plot(np.cos(theta), np.sin(theta), linestyle="--", color="black", linewidth=1)
# # force square geometry
# plt.gca().set_aspect("equal", adjustable="box")
# # ticks
# plt.xticks(fontsize=fontsize*0.8)
# plt.yticks(fontsize=fontsize*0.8)
# # save figure
# plt.savefig(figure_path / "eigenvalue_distribution", bbox_inches="tight")

# # share of simulations with debt cycles
# share_cycles = results["complex_pair"].mean()
# print(f"- share of simulations with debt cycles: {round(share_cycles * 100,4)}%")

# # share of simulations with debt cycles that are Minskyan
# share_cycles_minsky = results.loc[results["complex_ev"], "minsky_cycle"].mean()
# print(f"- share of simulations with debt cycles that are Minskyan: {round(share_cycles_minsky * 100,4)}%")

# # share of total simulation that have debt cycles and are Minskyan 
# # b_12 < 0 and b_21 > 0, and Im(eigenvalue) != 0
# share_minsky = results["minsky_cycle"].mean()
# print(f"- share of simulations with Minskyan debt cycles: {round(share_minsky * 100,4)}%")


### 2. real gdp growth shape ###

print("\n2. Ral GDP growth distribution")

# simulated real GDP distribution - generalised normal
num_bins = 100
res = np.histogram(macro_data["rgdp_growth"], bins=num_bins, density=True)
density = res[0]
bins = res[1]
x = np.linspace(-0.1, 0.15, 400)

# gennorm (exponential power/subbotin) fit 
params = stats.gennorm.fit(macro_data["rgdp_growth"])
pdf = stats.gennorm.pdf(x, params[0], params[1], params[2])

# laplace fit 
laplace_params = stats.laplace.fit(macro_data["rgdp_growth"])
laplace_pdf = stats.laplace.pdf(x, laplace_params[0], laplace_params[1])

# normal fit
norm_params = stats.norm.fit(macro_data["rgdp_growth"])
norm_pdf = stats.norm.pdf(x, norm_params[0], norm_params[1])

plt.figure(figsize=(x_figsize,x_figsize/1.3))
plt.scatter((bins[1:] + bins[:-1])/2, density, facecolors="none", edgecolors="k", marker="o", label="Simulated")
plt.plot(x, pdf, color="k", linewidth=1, label="Subbotin")
plt.plot(x, laplace_pdf, color="k", linewidth=1, label="Laplace", linestyle="--")
plt.plot(x, norm_pdf, color="k", linewidth=1, label="Normal", linestyle=":")
plt.ylim([0.008,50])
plt.yscale("log")
plt.yticks(fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.legend(fontsize=fontsize, loc="upper right")
plt.savefig(figure_path / "dist_rgdp_growth", bbox_inches="tight")

print("- created real GDP growth rate distribution plot")
print("- distribution fit parameters:")
print(f"  - Subbotin fit:\n    - beta = {params[0]}\n    - mu = {params[1]}\n    - alpha = {params[2]}")
print(f"  - Laplace fit:\n    - mu = {laplace_params[0]}\n    - alpha = {laplace_params[1]}")
print(f"  - Norm fit:\n    - mu = {norm_params[0]}\n    - alpha = {norm_params[1]}")
print(f"  - num obs = {len(macro_data)}")

# normality tests

print("- real GDP growth nomality hypothesis tests:")

normality_tests(macro_data["rgdp_growth"], significance=significance)

### 3. Recession duration ###
print("\n3. Recession duration")
# shift recession flag
macro_data["recession_shift"] = macro_data.groupby("simulation_id")["recession"].transform(lambda x: (x != x.shift()).cumsum())
# recession durations 
durations = macro_data[macro_data["recession"] == 1].groupby(["simulation_id", "recession_shift"]).size().values
# strictly positive (recesion defined as > 1 period)
durations = durations[durations > 1]
# skip if no recessions found
if len(durations) == 0:
    print("No recession durations found.")
else:
    # unique recession duration periods 
    unique_durations, counts = np.unique(durations, return_counts=True)
    # total number of recession periods
    total = counts.sum()
    # empirical probability density function
    pdf = counts / total

    # power-law fit (no xmin)
    log_d = np.log(unique_durations)
    log_pdf = np.log(pdf)
    slope_pl, intercept_pl, r_pl, _, _ = stats.linregress(log_d, log_pdf)
    alpha = -slope_pl
    pdf_powerlaw = np.exp(intercept_pl) * unique_durations ** (-alpha)
    rmse_pl = np.sqrt(np.mean((log_pdf - np.log(pdf_powerlaw)) ** 2))
    r2_pl = r_pl ** 2

    # exponential fit
    slope_exp, intercept_exp, r_exp, _, _ = stats.linregress(unique_durations, log_pdf)
    lambda_exp = -slope_exp
    pdf_exp = np.exp(intercept_exp) * np.exp(-lambda_exp * unique_durations)
    rmse_exp = np.sqrt(np.mean((log_pdf - np.log(pdf_exp)) ** 2))
    r2_exp = r_exp ** 2

    # update matplotlib pyplot settings


    # plot power-law fit 
    fig, ax = plt.subplots(figsize=(x_figsize,y_figsize * 1.5))
    ax.scatter(unique_durations, pdf, color="navy", edgecolors="k", s=40, label="Empirical PDF")
    ax.plot(unique_durations, pdf_powerlaw, color="limegreen", linewidth=3, label=rf"Power law ($\alpha$ = {alpha:.2f})")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.legend(fontsize=fontsize * 0.8)
    ax.tick_params(axis="both", labelsize=fontsize)
    plt.savefig(figure_path / "dist_recession_duration_power_law", bbox_inches="tight")

    # plot exponential fit 
    fig, ax = plt.subplots(figsize=(x_figsize,y_figsize * 1.5))
    ax.scatter(unique_durations, pdf, color="navy", edgecolors="k", s=40, label="Empirical PDF")
    ax.plot(unique_durations, pdf_exp, color="r", linestyle="--", linewidth=2, label=rf"Exponential ($\lambda$ = {lambda_exp:.2f})")
    ax.set_yscale("log")
    ax.legend(fontsize=fontsize * 0.8)
    ax.tick_params(axis="both", labelsize=fontsize)
    plt.savefig(figure_path / "dist_recession_duration_exponential", bbox_inches="tight")

    # print results
    print(f"- number of recession periods = {len(durations)}")
    print("- power-law fit (log-log regression):")
    print(f"  - alpha = {alpha:.4f}")
    print(f"  - R-squared = {r2_pl:.4f}")
    print(f"  - RMSE = {rmse_pl:.4f}")

    print("- exponential fit (log-linear regression):")
    print(f". - lambda = {lambda_exp:.4f}")
    print(f"  - R-squared = {r2_exp:.4f}")
    print(f"  - RMSE = {rmse_exp:.4f}")

    if r2_pl > r2_exp:
        print("- better fit (by R-squared): power-law")
    else:
        print("- better fit (by R-squared): exponential")

### 4. Autocorrelation of macro variables ###

print("\n4. Autocorrelation of macro variables")

print("- real GDP autocorrelation plot")

plot_autocorrelation(
    simulated=sim_real_gdp,
    empirical=emp_real_gdp,
    figsize=(x_figsize, y_figsize),
    fontsize=fontsize,
    savefig=figure_path / "acf_real_gdp"
)

print("- consumption autocorrelation plot")

plot_autocorrelation(
    simulated=sim_consumption,
    empirical=emp_consumption,
    figsize=(x_figsize, y_figsize),
    fontsize=fontsize,
    savefig=figure_path / "acf_consumption"
)

print("- investment autocorrelation plot")

plot_autocorrelation(
    simulated=sim_investment,
    empirical=emp_investment,
    figsize=(x_figsize, y_figsize),
    fontsize=fontsize,
    savefig=figure_path / "acf_investment"
)

print("- productivity autocorrelation plot")

plot_autocorrelation(
    simulated=sim_productivity,
    empirical=emp_productivity,
    figsize=(x_figsize, y_figsize),
    fontsize=fontsize,
    savefig=figure_path / "acf_productivity"
)

print("- unemployment rate autocorrelation plot")

plot_autocorrelation(
    simulated=sim_unemployment_rate,
    empirical=emp_unemployment_rate,
    figsize=(x_figsize, y_figsize),
    fontsize=fontsize,
    savefig=figure_path / "acf_unemployment_rate"
)

print("- corporate debt autocorrelation plot")

plot_autocorrelation(
    simulated=sim_debt,
    empirical=emp_debt,
    figsize=(x_figsize, y_figsize),
    fontsize=fontsize,
    savefig=figure_path / "acf_debt"
)

### 5. Cross-correlation between macro variables ###

print("\n5. Cross-correlation between macro variables")

print("- cross-correlation real GDP & real GDP")

plot_cross_correlation(
    xsimulated=sim_real_gdp,
    ysimulated=sim_real_gdp,
    xempirical=emp_real_gdp,
    yempirical=emp_real_gdp,
    figsize=(x_figsize,y_figsize),
    fontsize=fontsize,
    savefig=figure_path / "ccf_gdp_gdp"
)


print("- cross-correlation real GDP & consumption")

plot_cross_correlation(
    xsimulated=sim_real_gdp,
    ysimulated=sim_consumption,
    xempirical=emp_real_gdp,
    yempirical=emp_consumption,
    figsize=(x_figsize,y_figsize),
    fontsize=fontsize,
    savefig=figure_path / "ccf_gdp_consumption"
)

print("- cross-correlation real GDP & investment")

plot_cross_correlation(
    xsimulated=sim_real_gdp,
    ysimulated=sim_investment,
    xempirical=emp_real_gdp,
    yempirical=emp_investment,
    figsize=(x_figsize,y_figsize),
    fontsize=fontsize,
    savefig=figure_path / "ccf_gdp_investment"
)

print("- cross-correlation real GDP & corporate debt")

plot_cross_correlation(
    xsimulated=sim_real_gdp,
    ysimulated=sim_debt,
    xempirical=emp_real_gdp,
    yempirical=emp_debt,
    figsize=(x_figsize,y_figsize),
    fontsize=fontsize,
    savefig=figure_path / "ccf_gdp_debt"
)

print("- cross-correlation real GDP & unemployment rate")

plot_cross_correlation(
    xsimulated=sim_real_gdp,
    ysimulated=sim_unemployment_rate,
    xempirical=emp_real_gdp,
    yempirical=emp_unemployment_rate,
    figsize=(x_figsize,y_figsize),
    fontsize=fontsize,
    savefig=figure_path / "ccf_gdp_unemployment"
)

print("- cross-correlation corporate debt & consumption")

plot_cross_correlation(
    xsimulated=sim_debt,
    ysimulated=sim_unemployment_rate,
    xempirical=emp_debt,
    yempirical=emp_unemployment_rate,
    figsize=(x_figsize,y_figsize),
    fontsize=fontsize,
    savefig=figure_path / "ccf_debt_unemployment"
)

### 6. volatility hierarchy ###

print("\n6. Volatility hierarchy of GDP, consumption, and investment growth rates")

# real GDP 
gdp_std_by_sim = macro_data.groupby("simulation_id")["rgdp_growth"].std()

mean_gdp_std = gdp_std_by_sim.mean()
sd_gdp_std = gdp_std_by_sim.std()

emp_real_gdp_growth = np.log(emp_real_gdp[steps:-1]) - np.log(emp_real_gdp[1:-steps])

print("- real GDP growth volatility")
print(f"  - empirical standard deviation: {round(emp_real_gdp_growth.std() * 100,4)}%")
print("  - model standard deviation:")
print(f"    - average = {round(mean_gdp_std * 100,4)}%")
print(f"    - standard deviation = {round(sd_gdp_std * 100,4)}%")

# consumption 
cons_std_by_sim = macro_data.groupby("simulation_id")["consumption_growth"].std()

mean_cons_std = cons_std_by_sim.mean()
sd_cons_std = cons_std_by_sim.std()

emp_consumption_growth = np.log(emp_consumption[steps:-1]) - np.log(emp_consumption[1:-steps])

print("- consumption growth volatilty")
print(f"  - empirical standard deviation: {round(emp_consumption_growth.std() * 100,4)}%")
print("  - model standard deviation:")
print(f"  - average = {round(mean_cons_std * 100,4)}%")
print(f"  - standard deviation = {round(sd_cons_std * 100,4)}%")

# investment 
inv_std_by_sim = macro_data.groupby("simulation_id")["investment_growth"].std()

mean_inv_std = inv_std_by_sim.mean()
sd_inv_std = inv_std_by_sim.std()

emp_investment_growth = np.log(emp_investment[steps:-1]) - np.log(emp_investment[1:-steps])

print("- investment growth volatility")
print(f"  - empirical standard deviation: {round(emp_investment_growth.std() * 100,4)}%")
print("  - model standard deviation:")
print(f"    - average = {round(mean_inv_std * 100,4)}%")
print(f"    - standard deviation = {round(sd_inv_std * 100,4)}%")

### 7. pro-cyclicality of debt ###
# evidence provided in Minkyan analysis and cross-correlation
# of real GDP and corporate debt

print("7. Pro-cyclicality of corporate debt")
print("- see Minskyan evidence")
print("- see real GDP & debt cross-correlation")

### 8. relationship between credit and unemployment

print("\n8. Credit & unemployment relationship")

# model
credit_model = sm.OLS(single_sim_data["credit_gdp"], sm.add_constant(single_sim_data["unemployment_rate"]))
credit_results = credit_model.fit()
# plot results
plt.figure(figsize=(10,5))
plt.scatter(x=single_sim_data["unemployment_rate"], y=single_sim_data["credit_gdp"], color="skyblue", s=30, edgecolors="k")
plt.plot(single_sim_data["unemployment_rate"], credit_results.params["const"] + credit_results.params["unemployment_rate"]*single_sim_data["unemployment_rate"], color="r", linewidth=1)
# ticks
plt.yticks(fontsize=fontsize)
plt.xticks(fontsize=fontsize)
# save figure
plt.savefig(figure_path / "credit_curve", bbox_inches="tight")
# print results
print("- results:")
print(f"  - intercept = {credit_results.params["const"]:.4f}")
print(f"  - slope = {credit_results.params["unemployment_rate"]:.4f}")
print(f"  - R-squared = {credit_results.rsquared:.4f}")

### 9. Okun curve ###

print("\n9. Okun curve")

# model
okun_model = sm.OLS(single_sim_data["rgdp_growth"], sm.add_constant(single_sim_data["change_unemployment"]))
okun_results = okun_model.fit()
# plot results
plt.figure(figsize=(10,5))
plt.scatter(x=single_sim_data["change_unemployment"], y=single_sim_data["rgdp_growth"], color="skyblue", s=30, edgecolors="k")
plt.plot(single_sim_data["change_unemployment"], okun_results.params["const"] + okun_results.params["change_unemployment"]*single_sim_data["change_unemployment"], color="r", linewidth=1)
# ticks
plt.yticks(fontsize=fontsize)
plt.xticks(fontsize=fontsize)
# save figure
plt.savefig(figure_path / "okun_curve", bbox_inches="tight")
# print results
print("- results:")
print(f"  - intercept = {okun_results.params["const"]:.4f}")
print(f"  - slope = {okun_results.params["change_unemployment"]:.4f}")
print(f"  - R-squared = {okun_results.rsquared:.4f}")

### 10. price-Phillips curve ###

print("\n10. price-Phillips curve")

# model
ppc_model = sm.OLS(single_sim_data["inflation"], sm.add_constant(single_sim_data["unemployment_rate"]))
ppc_results = ppc_model.fit()
# plot results
plt.figure(figsize=(10,5))
plt.scatter(x=single_sim_data["unemployment_rate"], y=single_sim_data["inflation"], color="skyblue", s=30, edgecolors="k")
plt.plot(single_sim_data["unemployment_rate"], ppc_results.params["const"] + ppc_results.params["unemployment_rate"]*single_sim_data["unemployment_rate"], color="r", linewidth=1)
# ticks
plt.yticks(fontsize=fontsize)
plt.xticks(fontsize=fontsize)
# save figure
plt.savefig(figure_path / "phillips_curve_inflation", bbox_inches="tight")
# print results
print("- results:")
print(f"  - intercept = {ppc_results.params["const"]:.4f}")
print(f"  - slope = {ppc_results.params["unemployment_rate"]:.4f}")
print(f"  - R-squared = {ppc_results.rsquared:.4f}")

### 11. wage-Phillips curve ###

print("\n11. wage-phillips curve")

# Wage-Phillips Curve: relationship between wage inflation and unemployment rate
wpc_model = sm.OLS(single_sim_data["wage_inflation"], sm.add_constant(single_sim_data["unemployment_rate"]))
wpc_results = wpc_model.fit()
# plot results
plt.figure(figsize=(10,5))
plt.scatter(x=single_sim_data["unemployment_rate"], y=single_sim_data["wage_inflation"], color="skyblue", s=30, edgecolors="k")
plt.plot(single_sim_data["unemployment_rate"], wpc_results.params["const"] + wpc_results.params["unemployment_rate"]*single_sim_data["unemployment_rate"], color="r", linewidth=1)
# ticks
plt.yticks(fontsize=fontsize)
plt.xticks(fontsize=fontsize)
# save figure
plt.savefig(figure_path / "phillips_curve_wage", bbox_inches="tight")
# print results
print("- results:")
print(f"  - intercept = {wpc_results.params["const"]:.4f}")
print(f"  - slope = {wpc_results.params["unemployment_rate"]:.4f}")
print(f"  - R-squared = {wpc_results.rsquared:.4f}")



###------------------------------###
### microeconomic stylised facts ###
###------------------------------###

print("\n----------------------------")
print("MICROECONOMIC STYLISED FACTS")
print("----------------------------")

### micro data ###

# consumption firms 
cfirms = pd.read_sql_query(
    """
        SELECT 
            T3.scenario_id,
            T1.simulation_id,
            T2.simulation_index,
            T1.time,
            T1.id,
            T1.output,
            T1.output_growth,
            T1.firm_type,
            T1.labour,
            T1.debt,
            T1.equity,
            T1.market_share,
            T1.investment / T1.capital as investment_rate
        
        FROM firm_data AS T1

        -- join simulations and scenarios to get scenario id
        JOIN simulations AS T2 ON T2.simulation_id = T1.simulation_id
        JOIN scenarios AS T3 ON T3.scenario_id = T2.scenario_id
        
        WHERE T3.scenario_id = %(scenario_id)s 
            AND T1.time > %(start)s
            AND T1.firm_type = 'ConsumptionFirm'
        
        ORDER BY T1.simulation_id, T1.id, T1.time ASC
        ;
    """,
    engine,
    params = {
        "scenario_id": int(base_scenario["scenario_id"]),
        "start": start
    }
)

# capital firms 
kfirms = pd.read_sql_query(
    """
        SELECT 
            T3.scenario_id,
            T1.simulation_id,
            T2.simulation_index,
            T1.time,
            T1.id,
            T1.output,
            T1.output_growth,
            T1.labour,
            T1.debt,
            T1.equity,
            T1.market_share
        
        FROM firm_data AS T1

        -- join simulations and scenarios to get scenario id
        JOIN simulations AS T2 ON T2.simulation_id = T1.simulation_id
        JOIN scenarios AS T3 ON T3.scenario_id = T2.scenario_id
        
        WHERE T3.scenario_id = %(scenario_id)s 
            AND T1.time > %(start)s
            AND T1.firm_type = 'CapitalFirm'
        
        ORDER BY T1.simulation_id, T1.id, T1.time ASC
        ;
    """,
    engine,
    params = {
        "scenario_id": int(base_scenario["scenario_id"]),
        "start": start
    }
)

banks = pd.read_sql_query(
    """
        SELECT 
            T3.scenario_id,
            T1.simulation_id,
            T2.simulation_index,
            T1.id,
            T1.time,
            T1.loans,
            T1.market_share,
            T1.degree,
            T1.equity
        
        FROM bank_data AS T1

        -- join simulations and scenarios to get scenario id
        JOIN simulations AS T2 ON T2.simulation_id = T1.simulation_id
        JOIN scenarios AS T3 ON T3.scenario_id = T2.scenario_id
        
        WHERE T3.scenario_id = %(scenario_id)s 
            AND T1.time > %(start)s
        
        ORDER BY T1.simulation_id, T1.id, T1.time ASC
        ;
    """,
    engine,
    params = {
        "scenario_id": int(base_scenario["scenario_id"]),
        "start": start
    }
)

# midpoint snapshot 
snapshot_cfirms = cfirms[cfirms["time"] == middle]
snapshot_kfirms = kfirms[kfirms["time"] == middle]
snapshot_banks = banks[banks["time"] == middle]

# randomly selected cfirm id's
cfirms_sim = cfirms[cfirms["simulation_index"] == sim_index]
cfirm_ids = np.random.choice(cfirms_sim["id"].unique(), size=10, replace=False)

# randomly selected kfirm id's
kfirms_sim = kfirms[kfirms["simulation_index"] == sim_index]
kfirm_ids = np.random.choice(kfirms_sim["id"].unique(), size=10, replace=False)

# randomly selected bank id's
banks_sim = banks[banks["simulation_index"] == sim_index]
bank_ids = np.random.choice(banks_sim["id"].unique(), size=10, replace=False)

### 1. firms output growth distribution ###

print("\n1. Firms output growth distribution")

# number of bins
num_bins = 100

# cfirms output growth distribution
print("- consumption firms output growth dist")

snapshot_cfirms = cfirms[cfirms["time"] == middle]

# binned growth rates
cfirm_res = np.histogram(snapshot_cfirms["output_growth"], bins=num_bins, density=True)
cfirm_density = cfirm_res[0]
cfirm_bins = cfirm_res[1]
cfirm_x = np.linspace(-4, 4, 400)

# gennorm (exponential power/subbotin) fit 
cfirm_gennorm_params = stats.gennorm.fit(snapshot_cfirms["output_growth"])
cfirm_gennorm_pdf = stats.gennorm.pdf(cfirm_x, cfirm_gennorm_params[0], cfirm_gennorm_params[1], cfirm_gennorm_params[2])

# laplace fit 
cfirm_laplace_params = stats.laplace.fit(snapshot_cfirms["output_growth"])
cfirm_laplace_pdf = stats.laplace.pdf(cfirm_x, cfirm_laplace_params[0], cfirm_laplace_params[1])

# normal fit
cfirm_norm_params = stats.norm.fit(snapshot_cfirms["output_growth"])
cfirm_norm_pdf = stats.norm.pdf(cfirm_x, cfirm_norm_params[0], cfirm_norm_params[1])

plt.figure(figsize=(x_figsize,x_figsize/1.3))
plt.scatter((cfirm_bins[1:] + cfirm_bins[:-1])/2, cfirm_density, facecolors="none", edgecolors="k", marker="o", label="Simulated")
plt.plot(cfirm_x, cfirm_gennorm_pdf, color="k", linewidth=1, label="Subbotin")
plt.plot(cfirm_x, cfirm_laplace_pdf, color="k", linestyle="--", linewidth=1, label="Laplace")
plt.plot(cfirm_x, cfirm_norm_pdf, color="k", linestyle=":", linewidth=1, label="Normal")
plt.ylim([0.0002,15])
plt.xlim([-3.2,3.2])
plt.yscale("log")
plt.yticks(fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.legend(fontsize=fontsize * 0.8, loc="upper right")
plt.savefig(figure_path / "dist_cfirm_growth", bbox_inches="tight")
print(f"  - subbotin fit\n    - beta = {cfirm_gennorm_params[0]}\n    - mu = {cfirm_gennorm_params[1]}\n    - alpha = {cfirm_gennorm_params[2]}")
print(f"  - laplace fit\n    - mu = {cfirm_laplace_params[0]}\n    - alpha = {cfirm_laplace_params[1]}")
print(f"  - norm fit\n    - mu = {cfirm_norm_params[0]}\n    - alpha = {cfirm_norm_params[1]}")
print(f"  - num obs = {len(cfirms)}")

# large cfirms subbotin fit

large_cfirms = snapshot_cfirms[snapshot_cfirms["labour"] > 20]

# gennorm (exponential power/subbotin) fit 
large_cfirm_gennorm_params = stats.gennorm.fit(large_cfirms["output_growth"])
print(f"- large cfirms subbotin fit\n    - beta = {large_cfirm_gennorm_params[0]}\n    - mu = {large_cfirm_gennorm_params[1]}\n    - alpha = {large_cfirm_gennorm_params[2]}")

# consumption firm normality tests
print("- consumption firms output growth normality tests:")
normality_tests(cfirms["output_growth"], significance=significance)

# capital firm normality tests
print("- capital firms output growth normality tests:")
normality_tests(kfirms["output_growth"], significance=significance)

### 2. firms size distribution ###

print("\n2. Firms size distribution")

# cfirms output distribution 
print("- consumption firms size distribution")
# plot_ccdf(snapshot_cfirms["output"], figsize=(x_figsize,y_figsize), fontsize=fontsize, savefig=figure_path / "dist_cfirm_size")

# kfirms output distribution
print("- capital firms size distribution")
# plot_ccdf(snapshot_kfirms["output"], figsize=(x_figsize,y_figsize), fontsize=fontsize, savefig=figure_path / "dist_kfirm_size")

### 3. Lumpiness of investment rates ###

print("3. Lumpiness of investment rates")

print("- plot investment rates for selected consumption firms")

fig, ax = plt.subplots(figsize=(x_figsize,y_figsize))

for cfirm_id in cfirm_ids:
    cfirm = cfirms_sim[cfirms_sim["id"] == cfirm_id]
    ax.plot(cfirm["time"], cfirm["investment_rate"])

ax.tick_params(axis="both", labelsize=fontsize)
plt.savefig(figure_path / "investment_rate_lumpiness", bbox_inches="tight")

# skewness test

stat_skew, p_val2_skew = stats.skewtest(cfirms["investment_rate"])

# One-sided test for skew > 0
p_val1_skew = p_val2_skew / 2 if stat_skew > 0 else 1 - p_val2_skew / 2

print("- skewness test")
print(f"  - test stat = {stat_skew:.4f}")
print(f"One-sided p-value (skew > 0) = {p_val1_skew * 100:.4f}")

# Leptokurtic test

stat_kurtosis, p_val2_kurtosis = stats.kurtosistest(cfirms["investment_rate"])

# One-sided test for excess kurtosis > 0
p_val1_kurtosis = p_val2_kurtosis / 2 if stat_kurtosis > 0 else 1 - p_val2_kurtosis / 2

print("- kurtosis test")
print(f"  - test stat = {stat_kurtosis:.4f}")
print(f"  - p-value (excess kurtosis > 0) = {p_val1_kurtosis * 100:.4f}")

print("- investment rate normality tests:")
normality_tests(cfirms["investment_rate"], significance=significance)

### 4. Persistent productivity ###
print("\n4. Persistence of productivity")
print("- from productivity GBM process")

### 5. Persistent productivity differences ###
print("\n5. Persistent productivity differences")
print("- from productivity GBM process")

### 6. Persistency of market shares ###
print("\n6. Market share persistence")

print("- consumption firms")

print("- capital firms")

print("- banks")

### 7. bank degree distribution ###

print("7. Bank degree distribution")

bank_degrees = snapshot_banks["degree"][snapshot_banks["degree"] > 0]

plot_ccdf(bank_degrees, figsize=(x_figsize,y_figsize), fontsize=fontsize, savefig=figure_path / "dist_bank_degree")

print(f"\nFINISHED STYLISED FACTS ANALYSIS! Check stylised_facts figures folder\n=> {figure_path}\n")
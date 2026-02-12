import numpy as np
import pandas as pd
from pathlib import Path
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
from dynamin.utils import load_config, sql_engine
from analysis.utils import load_macro_data, plot_autocorrelation, plot_cross_correlation, normality_tests, plot_ccdf

# is this analysis from the examples folder?
is_true = input('Is this an analysis from the examples folder [y/n]: ').lower().startswith('y')

# if so, which example is this
if is_true:
    example_folder = input("What is the example folder called? ")

### Plot Parameters ###

# change matplotlib font to serif
plt.rcParams['font.family'] = ['serif']
# figure size
x_figsize = 10
y_figsize = x_figsize/2
# fontsize
fontsize = 25
# upper decile 
upper = 0.95
# lower decile
lower = 0.05

### Paths ###

# analysis directory
analysis_path = Path(__file__).parent
# analysis directory 
dynamin_path = Path(analysis_path).parent
# examples directory 
examples_path = dynamin_path / "examples"
# figure path
figure_path = analysis_path / "figures" / "stylised_facts"
# create figure path if it doesn't exist
figure_path.mkdir(parents=True, exist_ok=True)

### parameters ###

# base parameters file
if is_true:
    params = load_config(examples_path / example_folder / "config" / "parameters.yaml")
else:
    params = load_config("parameters.yaml")
# analysis parameters
steps = params['simulation']['steps']
num_years = params['simulation']['years']
start = params['simulation']['start'] * steps
middle = int(start + (num_years * steps) / 2)
years = np.linspace(0, num_years, num_years * steps)

# random simulation 
np.random.seed(params['simulation']['seed'])
sim_index = 0#np.random.randint(0, params['simulation']['num_sims'])
print("Randomly selected simulation index for plots: ", sim_index)

# database parameters
# base parameters file
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

### time-series plots ###

print("\nCreating time-series plots...")

### real output ###

print("- creating plot of real output")

# suffix for saving figures 
suffix = base_scenario["scenario_name"]

# get macro data 
macro_data = load_macro_data(engine, int(base_scenario["scenario_id"]), params)
single_sim_data = macro_data.loc[macro_data["simulation_index"]==sim_index]
# plot gdp growth 
# real GDP and components
plt.figure(figsize=(x_figsize,y_figsize))
# real GDP
plt.plot(years, single_sim_data['real_gdp'], color='k', linewidth=1, label='GDP')
# consumption
plt.plot(years, single_sim_data['real_consumption'], color='k', linestyle='--', linewidth=1.2, label='Cons.')
# investment
plt.plot(years, single_sim_data['real_investment'], color='k', linestyle=':', linewidth=1.5, label='Inv.')
# log y axis
plt.yscale('log')
# recession shaded
for t in single_sim_data["time"].loc[single_sim_data['recession']==1]:
    i = t - start
    if i > 0 and i < len(years):
        plt.axvspan(years[i-1], years[i], facecolor='0.2', alpha=0.25)
plt.yticks(fontsize=fontsize)
plt.xticks(fontsize=fontsize)
# legend
plt.legend(loc='upper left', fontsize=fontsize)
# save figure
plt.savefig(figure_path / "time_series_outputs", bbox_inches='tight')

# std over time within each simulation
cons_std_by_sim = (macro_data.groupby("simulation_id")["consumption_growth"].std())

mean_cons_std = cons_std_by_sim.mean()
sd_cons_std = cons_std_by_sim.std()

inv_std_by_sim = (macro_data.groupby("simulation_id")["investment_growth"].std())

mean_inv_std = inv_std_by_sim.mean()
sd_inv_std = inv_std_by_sim.std()

print('\nConsumption Growth Standard Deviation')
print(f"- consumption growth average std dev. = {mean_cons_std * 100}%")
print(f"- consumption growth std dev. std. dev. = {sd_cons_std * 100}%")
print('\nInvestment Growth Standard Deviation')
print(f"- investment growth average std dev. = {mean_inv_std * 100}%")
print(f"- investment growth std dev. std. dev. = {sd_inv_std * 100}%")

### GDP shares ###

print("\n- creating plot for GDP shares")

# output shares 
plt.figure(figsize=(x_figsize,y_figsize))
# plot debt ratio
plt.plot(years, single_sim_data['debt_ratio'], color='k', linewidth=1, label=r'D/Y')
# plot wage share 
plt.plot(years, single_sim_data['wage_share'], color='k', linestyle='--', linewidth=1, label=r'W/Y')
# plot profit share 
plt.plot(years, single_sim_data['profit_share'], color='k', linestyle=':', linewidth=1, label=r'$\Pi$/Y')
# plot labour productivity growth rate 
plt.axhline(0, color='k', linestyle='--', linewidth=1)
# recession shaded
for t in single_sim_data["time"].loc[single_sim_data['recession']==1]:
    i = t - start
    if i > 0 and i < len(years):
        plt.axvspan(years[i-1], years[i], facecolor='0.2', alpha=0.25)
# ticks
plt.yticks(fontsize=fontsize)
plt.xticks(fontsize=fontsize)
# legend
plt.legend(loc='lower left', fontsize=fontsize)
# save figure
plt.savefig(figure_path / "time_series_gdp_shares", bbox_inches="tight")

### real GDP growth ###

print("- creating plot for real GDP growth")

# real gdp growth
plt.figure(figsize=(x_figsize,y_figsize))
# plot 
plt.plot(years, single_sim_data['rgdp_growth'], color='k', linewidth=1)
# 0 line
plt.axhline(0, color='k', linestyle='--', linewidth=1)
# recession shaded
for t in single_sim_data["time"].loc[single_sim_data['recession']==1]:
    i = t - start
    if i > 0 and i < len(years):
        plt.axvspan(years[i-1], years[i], facecolor='0.2', alpha=0.25)
# ticks
plt.yticks(fontsize=fontsize)
plt.xticks(fontsize=fontsize)
# save figure
plt.savefig(figure_path / "time_series_rgdp_growth", bbox_inches='tight')

### inflation ###

print("- creating plot for inflation")

# figure
plt.figure(figsize=(x_figsize,y_figsize))
# plot
plt.plot(years, single_sim_data['inflation'], color='k', linewidth=1)
plt.axhline(0, color='k', linestyle='--', linewidth=1)
# recession shaded
for t in single_sim_data["time"].loc[single_sim_data['recession']==1]:
    i = t - start
    if i > 0 and i < len(years):
        plt.axvspan(years[i-1], years[i], facecolor='0.2', alpha=0.25)
# ticks
plt.yticks(fontsize=fontsize)
plt.xticks(fontsize=fontsize)
# save figure
plt.savefig(figure_path / 'time_series_inflation', bbox_inches='tight')

### inflation ###

print("- creating plot for unemployment rate")

# figure
plt.figure(figsize=(x_figsize,y_figsize))
# plot
plt.plot(years, single_sim_data['unemployment_rate'], color='k', linewidth=1)
plt.axhline(0, color='k', linestyle='--', linewidth=1)
# recession shaded
for t in single_sim_data["time"].loc[single_sim_data['recession']==1]:
    i = t - start
    if i > 0 and i < len(years):
        plt.axvspan(years[i-1], years[i], facecolor='0.2', alpha=0.25)
# ticks
plt.yticks(fontsize=fontsize)
plt.xticks(fontsize=fontsize)
# save figure
plt.savefig(figure_path / 'time_series_unemployment_rate', bbox_inches='tight')

### credit rate (total debt growth rate) ###

print("- creating plot for credit rate")

# figure
plt.figure(figsize=(x_figsize,y_figsize))
# plot 
plt.plot(years, single_sim_data['credit_gdp'], color='k', linewidth=1)
plt.axhline(0, color='k', linestyle='--', linewidth=1)
# recession shaded
for t in single_sim_data["time"].loc[single_sim_data['recession']==1]:
    i = t - start
    if i > 0 and i < len(years):
        plt.axvspan(years[i-1], years[i], facecolor='0.2', alpha=0.25)
# ticks
plt.yticks(fontsize=fontsize)
plt.xticks(fontsize=fontsize)
# save figure
plt.savefig(figure_path / 'time_series_credit_gdp', bbox_inches='tight')

print("\nCreating economic relationship plots...")

### inflation-Phillips curve ###

print("- creating inflation-phillips curve\n")
# Inflation-Phillips Curve: relationship between inflation and unemployment rate
pc_model = sm.OLS(single_sim_data['inflation'], sm.add_constant(single_sim_data['unemployment_rate']))
pc_results = pc_model.fit()
# plot results
plt.figure(figsize=(10,5))
plt.scatter(x=single_sim_data['unemployment_rate'], y=single_sim_data['inflation'], color='skyblue', s=30, edgecolors='k')
plt.plot(single_sim_data['unemployment_rate'], pc_results.params['const'] + pc_results.params['unemployment_rate']*single_sim_data['unemployment_rate'], color='r', linewidth=1)
# ticks
plt.yticks(fontsize=fontsize)
plt.xticks(fontsize=fontsize)
# save figure
plt.savefig(figure_path / 'phillips_curve_inflation', bbox_inches='tight')

print("- inflation-phillips model fit:\n")
print(pc_results.summary())

### wage-Phillips curve ###

print("- creating wage-phillips curve\n")
# Wage-Phillips Curve: relationship between wage inflation and unemployment rate
pc_model = sm.OLS(single_sim_data['wage_inflation'], sm.add_constant(single_sim_data['unemployment_rate']))
pc_results = pc_model.fit()
# plot results
plt.figure(figsize=(10,5))
plt.scatter(x=single_sim_data['unemployment_rate'], y=single_sim_data['wage_inflation'], color='skyblue', s=30, edgecolors='k')
plt.plot(single_sim_data['unemployment_rate'], pc_results.params['const'] + pc_results.params['unemployment_rate']*single_sim_data['unemployment_rate'], color='r', linewidth=1)
# ticks
plt.yticks(fontsize=fontsize)
plt.xticks(fontsize=fontsize)
# save figure
plt.savefig(figure_path / 'phillips_curve_wage', bbox_inches='tight')

print("- wage-phillips model fit:\n")
print(pc_results.summary())

### Keen (credit-unemployment) curve ###

print("- keen curve\n")
# Keen Curve: relationship between credit rate and unemployment rate
pc_model = sm.OLS(single_sim_data['credit_gdp'], sm.add_constant(single_sim_data['unemployment_rate']))
pc_results = pc_model.fit()
# plot results
plt.figure(figsize=(10,5))
plt.scatter(x=single_sim_data['unemployment_rate'], y=single_sim_data['credit_gdp'], color='skyblue', s=30, edgecolors='k')
plt.plot(single_sim_data['unemployment_rate'], pc_results.params['const'] + pc_results.params['unemployment_rate']*single_sim_data['unemployment_rate'], color='r', linewidth=1)
# ticks
plt.yticks(fontsize=fontsize)
plt.xticks(fontsize=fontsize)
# save figure
plt.savefig(figure_path / 'keen_curve', bbox_inches='tight')

print("- keen model fit:\n")
print(pc_results.summary())

### okun curve ###

print("- okun curve\n")
# okun Curve: relationship between real GDP growth and the change in the unemployment rate
pc_model = sm.OLS(single_sim_data['rgdp_growth'], sm.add_constant(single_sim_data['change_unemployment']))
pc_results = pc_model.fit()
# plot results
plt.figure(figsize=(10,5))
plt.scatter(x=single_sim_data['change_unemployment'], y=single_sim_data['rgdp_growth'], color='skyblue', s=30, edgecolors='k')
plt.plot(single_sim_data['change_unemployment'], pc_results.params['const'] + pc_results.params['change_unemployment']*single_sim_data['change_unemployment'], color='r', linewidth=1)
# ticks
plt.yticks(fontsize=fontsize)
plt.xticks(fontsize=fontsize)
# save figure
plt.savefig(figure_path / 'okun_curve', bbox_inches='tight')

print("- keen model fit:\n")
print(pc_results.summary())

### autocorrelation plots ###

print("\nCreating autocorrelation plots...")

# empirical data 

empirical_data_path = analysis_path/ "empirical_data"

# US data
emp_real_gdp = pd.read_csv(empirical_data_path / "GDPC1.csv", index_col=0).to_numpy()
emp_consumption = pd.read_csv(empirical_data_path / "PCECC96.csv", index_col=0).to_numpy()
emp_investment = pd.read_csv(empirical_data_path / "GPDIC1.csv", index_col=0).to_numpy()
emp_employment = pd.read_csv(empirical_data_path / "CE16OV.csv", index_col=0).to_numpy()
emp_unemployment_rate = pd.read_csv(empirical_data_path / "UNRATE.csv", index_col=0).to_numpy()
emp_debt = pd.read_csv(empirical_data_path / "TBSDODNS.csv", index_col=0).to_numpy()
# define productivity 
emp_productivity = emp_real_gdp / emp_employment

# simulated data 
sim_real_gdp = macro_data.pivot(index="simulation_index", columns="time", values="real_gdp").to_numpy()
sim_consumption = macro_data.pivot(index="simulation_index", columns="time", values="real_consumption").to_numpy()
sim_investment = macro_data.pivot(index="simulation_index", columns="time", values="real_investment").to_numpy()
sim_productivity = macro_data.pivot(index="simulation_index", columns="time", values="productivity").to_numpy()
sim_unemployment_rate = macro_data.pivot(index="simulation_index", columns="time", values="unemployment_rate").to_numpy()
sim_debt = macro_data.pivot(index="simulation_index", columns="time", values="debt").to_numpy()

# autocorrelation plots

plot_autocorrelation(
    simulated=sim_real_gdp,
    empirical=emp_real_gdp,
    figsize=(x_figsize, y_figsize),
    fontsize=fontsize,
    savefig=figure_path / "acf_real_gdp"
)

plot_autocorrelation(
    simulated=sim_consumption,
    empirical=emp_consumption,
    figsize=(x_figsize, y_figsize),
    fontsize=fontsize,
    savefig=figure_path / "acf_consumption"
)

plot_autocorrelation(
    simulated=sim_investment,
    empirical=emp_investment,
    figsize=(x_figsize, y_figsize),
    fontsize=fontsize,
    savefig=figure_path / "acf_investment"
)

plot_autocorrelation(
    simulated=sim_productivity,
    empirical=emp_productivity,
    figsize=(x_figsize, y_figsize),
    fontsize=fontsize,
    savefig=figure_path / "acf_productivity"
)

plot_autocorrelation(
    simulated=sim_unemployment_rate,
    empirical=emp_unemployment_rate,
    figsize=(x_figsize, y_figsize),
    fontsize=fontsize,
    savefig=figure_path / "acf_unemployment_rate"
)

plot_autocorrelation(
    simulated=sim_debt,
    empirical=emp_debt,
    figsize=(x_figsize, y_figsize),
    fontsize=fontsize,
    savefig=figure_path / "acf_debt"
)

### cross-correlation ###

print("\nCreating cross-correlation plots...")

plot_cross_correlation(
    xsimulated=sim_real_gdp,
    ysimulated=sim_real_gdp,
    xempirical=emp_real_gdp,
    yempirical=emp_real_gdp,
    figsize=(x_figsize,y_figsize),
    fontsize=fontsize,
    savefig=figure_path / "ccf_gdp_gdp"
)

plot_cross_correlation(
    xsimulated=sim_real_gdp,
    ysimulated=sim_consumption,
    xempirical=emp_real_gdp,
    yempirical=emp_consumption,
    figsize=(x_figsize,y_figsize),
    fontsize=fontsize,
    savefig=figure_path / "ccf_gdp_consumption"
)

plot_cross_correlation(
    xsimulated=sim_real_gdp,
    ysimulated=sim_investment,
    xempirical=emp_real_gdp,
    yempirical=emp_investment,
    figsize=(x_figsize,y_figsize),
    fontsize=fontsize,
    savefig=figure_path / "ccf_gdp_investment"
)

plot_cross_correlation(
    xsimulated=sim_real_gdp,
    ysimulated=sim_debt,
    xempirical=emp_real_gdp,
    yempirical=emp_debt,
    figsize=(x_figsize,y_figsize),
    fontsize=fontsize,
    savefig=figure_path / "ccf_gdp_debt"
)

plot_cross_correlation(
    xsimulated=sim_real_gdp,
    ysimulated=sim_unemployment_rate,
    xempirical=emp_real_gdp,
    yempirical=emp_unemployment_rate,
    figsize=(x_figsize,y_figsize),
    fontsize=fontsize,
    savefig=figure_path / "ccf_gdp_unemployment"
)

plot_cross_correlation(
    xsimulated=sim_debt,
    ysimulated=sim_unemployment_rate,
    xempirical=emp_debt,
    yempirical=emp_unemployment_rate,
    figsize=(x_figsize,y_figsize),
    fontsize=fontsize,
    savefig=figure_path / "ccf_debt_unemployment"
)

### real gdp growth shape ###

print("\nCreating real GDP growth distribution plot...")

# simulated real GDP distribution - generalised normal
num_bins = 100
res = np.histogram(macro_data['rgdp_growth'], bins=num_bins, density=True)
density = res[0]
bins = res[1]
x = np.linspace(-0.25, 0.3, 400)

# gennorm (exponential power/subbotin) fit 
params = stats.gennorm.fit(macro_data['rgdp_growth'])
pdf = stats.gennorm.pdf(x, params[0], params[1], params[2])

# laplace fit 
laplace_params = stats.laplace.fit(macro_data['rgdp_growth'])
laplace_pdf = stats.laplace.pdf(x, laplace_params[0], laplace_params[1])

# normal fit
norm_params = stats.norm.fit(macro_data['rgdp_growth'])
norm_pdf = stats.norm.pdf(x, norm_params[0], norm_params[1])

plt.figure(figsize=(x_figsize,x_figsize/1.3))
plt.scatter((bins[1:] + bins[:-1])/2, density, facecolors='none', edgecolors='k', marker='o', label='Simulated')
plt.plot(x, pdf, color='k', linewidth=1, label='Subbotin')
plt.plot(x, laplace_pdf, color='k', linewidth=1, label='Laplace', linestyle='--')
plt.plot(x, norm_pdf, color='k', linewidth=1, label='Normal', linestyle=':')
plt.ylim([0.003,50])
plt.yscale('log')
plt.yticks(fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.legend(fontsize=fontsize, loc='upper right')
plt.savefig(figure_path / 'dist_rgdp_growth', bbox_inches='tight')
print(f'- Subbotin fit:\n   - beta = {params[0]}\n   - mu = {params[1]}\n   - alpha = {params[2]}')
print(f'- Laplace fit:\n   - mu = {laplace_params[0]}\n   - alpha = {laplace_params[1]}')
print(f'- Norm fit:\n   - mu = {norm_params[0]}\n   - alpha = {norm_params[1]}')
print(f"num obs = {len(macro_data)}")

# normality tests
normality_tests(macro_data["rgdp_growth"], significance=0.01)

print("\nReal GDP growth nomality hypothesis test")

# significance level
significance = 0.01



### consumption firms ###

print("\nConsumption firm analysis...\n")

cfirms = pd.read_sql_query(
    """
        SELECT 
            T3.scenario_id,
            T1.id,
            T1.output,
            T1.output_growth,
            T1.firm_type
        
        FROM firm_data AS T1

        -- join simulations and scenarios to get scenario id
        JOIN simulations AS T2 ON T2.simulation_id = T1.simulation_id
        JOIN scenarios AS T3 ON T3.scenario_id = T2.scenario_id
        
        WHERE T3.scenario_id = %(scenario_id)s 
            AND T1.time = %(middle)s
            AND T1.firm_type = 'ConsumptionFirm'
        
        ORDER BY T1.simulation_id, T1.time ASC
        ;
    """,
    engine,
    params = {
        "scenario_id": int(base_scenario["scenario_id"]),
        "middle": middle
    }
)

print("- plot output growth rate & distribution fit")

# simulated real GDP distribution - generalised normal
num_bins = 100
res = np.histogram(cfirms['output_growth'], bins=num_bins, density=True)
density = res[0]
bins = res[1]

x = np.linspace(-2, 3, 400)

# gennorm (exponential power/subbotin) fit 
params = stats.gennorm.fit(cfirms['output_growth'])
pdf = stats.gennorm.pdf(x, params[0], params[1], params[2])

# laplace fit 
laplace_params = stats.laplace.fit(cfirms['output_growth'])
laplace_pdf = stats.laplace.pdf(x, laplace_params[0], laplace_params[1])

# normal fit
norm_params = stats.norm.fit(cfirms['output_growth'])
norm_pdf = stats.norm.pdf(x, norm_params[0], norm_params[1])

plt.figure(figsize=(x_figsize,x_figsize/1.3))
plt.scatter((bins[1:] + bins[:-1])/2, density, facecolors='none', edgecolors='k', marker='o', label='Simulated')
plt.plot(x, pdf, color='k', linewidth=1, label='Subbotin')
plt.plot(x, laplace_pdf, color='k', linestyle='--', linewidth=1, label='Laplace')
plt.plot(x, norm_pdf, color='k', linestyle=':', linewidth=1, label='Normal')
plt.ylim([0.0005,20])
plt.yscale('log')
plt.yticks(fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.legend(fontsize=fontsize, loc='upper right')
plt.savefig(figure_path / "dist_cfirm_growth", bbox_inches='tight')
print(f' - Subbotin fit\n   - beta = {params[0]}\n   - mu = {params[1]}\n   - alpha = {params[2]}')
print(f' - Laplace fit\n   - mu = {laplace_params[0]}\n   - alpha = {laplace_params[1]}')
print(f' - Norm fit\n   - mu = {norm_params[0]}\n   - alpha = {norm_params[1]}')
print(f'Num obs = {len(cfirms)}')

# normality tests
print("\n- C-firm Output Growth Normality Tests:")
normality_tests(cfirms["output_growth"], significance=0.01)

# CCDF distribution of cfirms output
plot_ccdf(cfirms['output'], figsize=(x_figsize,y_figsize), fontsize=fontsize, ylim=[0.00005,2], savefig=figure_path / "ccdf_cfirms_output.png")

print("\n- C-firm Size Normality Tests:")
normality_tests(cfirms["output"], significance=0.01)

### remove small cfirms ###

adjusted_cfirms = pd.read_sql_query(
    """
        SELECT 
            T3.scenario_id,
            T1.time,
            T1.output,
            T1.output_growth
        
        FROM firm_data AS T1

        -- join simulations and scenarios to get scenario id
        JOIN simulations AS T2 ON T2.simulation_id = T1.simulation_id
        JOIN scenarios AS T3 ON T3.scenario_id = T2.scenario_id
        
        WHERE T3.scenario_id = %(scenario_id)s 
            AND T1.time = %(middle)s
            AND T1.firm_type = 'ConsumptionFirm'
            AND T1.labour > 10
        
        ORDER BY T1.simulation_id, T1.time ASC
        ;
    """,
    engine,
    params = {
        "scenario_id": int(base_scenario["scenario_id"]),
        "middle": middle
    }
)

print("\n- Subbotin fit with no small firms:")

params = stats.gennorm.fit(adjusted_cfirms['output_growth'])
pdf = stats.gennorm.pdf(x, params[0], params[1], params[2])

print(f'   - beta = {params[0]}\n   - mu = {params[1]}\n   - alpha = {params[2]}')

kfirms = pd.read_sql_query(
    """
        SELECT 
            T3.scenario_id,
            T1.time,
            T1.output,
            T1.output_growth
        
        FROM firm_data AS T1

        -- join simulations and scenarios to get scenario id
        JOIN simulations AS T2 ON T2.simulation_id = T1.simulation_id
        JOIN scenarios AS T3 ON T3.scenario_id = T2.scenario_id
        
        WHERE T3.scenario_id = %(scenario_id)s 
            AND T1.time = %(middle)s
            AND T1.firm_type = 'CapitalFirm'
        
        ORDER BY T1.simulation_id, T1.time ASC
        ;
    """,
    engine,
    params = {
        "scenario_id": int(base_scenario["scenario_id"]),
        "middle": middle
    }
)

# normality tests
print("\n- K-firm Output Growth Normality Tests:")
normality_tests(kfirms["output_growth"], significance=0.01)

print("\n- K-firm Size Normality Tests:")
normality_tests(kfirms["output"], significance=0.01)

print(f"\nFINISHED STYLISED FACTS ANALYSIS! Check stylised_facts figures folder\n=> {figure_path}\n")
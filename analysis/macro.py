import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from dynamin.utils import load_config, sql_engine
from analysis.utils import load_macro_data, box_plot_scenarios, format_scenario_name

# is this analysis from the examples folder?
is_true = input("Run macro.py analysis for an example from the examples folder [y/n]: ").lower().startswith("y")

# if so, which example is this
if is_true:
    example_folder = input("What is the example folder called? ")

### Plot Parameters ###

# change matplotlib font to serif
plt.rcParams["font.family"] = ["serif"]
# figure size
x_figsize = 10
y_figsize = x_figsize/2
# fontsize
fontsize = 18
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
figure_path = analysis_path / "figures" / "macro"
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
years = np.linspace(0, num_years, num_years * steps)

# random simulation 
sim_index = 5
print("Randomly selected simulation index for plots: ", sim_index)

# database parameters 
if is_true:
    db_params = load_config(examples_path / example_folder / "config" / "database.yaml")
else:
    db_params = load_config("databse.yaml")

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

### Plot GDP Ratios ###

print(f"\nCreating GDP ratio plot")

fig = plt.figure(figsize=(x_figsize * 2, x_figsize * 2.2))

outer = gs.GridSpec(2, 2, figure=fig, wspace=0.25, hspace=0.15)

for i, scenario in scenarios.iterrows():

    row = i // 2
    col = i % 2

    inner = gs.GridSpecFromSubplotSpec(
        3, 1,
        subplot_spec=outer[row, col],
        hspace=0.05
    )

    macro_data = load_macro_data(engine, scenario["scenario_id"], params)

    macro_group = macro_data.groupby(by="time")
    macro_median = macro_group.quantile(0.5)
    macro_upper = macro_group.quantile(upper)
    macro_lower = macro_group.quantile(lower)

    ax0 = fig.add_subplot(inner[0])
    ax1 = fig.add_subplot(inner[1], sharex=ax0)
    ax2 = fig.add_subplot(inner[2], sharex=ax0)

    ### debt ratio ###
    ax0.plot(years, macro_median["debt_ratio"], color="k", linewidth=1)
    ax0.plot(years, macro_data.loc[macro_data["simulation_index"]==sim_index, "debt_ratio"], linestyle="--", color="k", linewidth=1)
    ax0.fill_between(years, macro_lower["debt_ratio"], macro_upper["debt_ratio"], alpha=0.2, linestyle="--", color="tab:red", linewidth=1)
    ax0.set_ylim((0.08, 0.72))
    ax0.set_yticks((0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7))
    ax0.set_ylabel("Debt Ratio", fontsize=fontsize + 2)

    ### wage share ###
    ax1.plot(years, macro_median["wage_share"], color="k", linewidth=1)
    ax1.plot(years, macro_data.loc[macro_data["simulation_index"]==sim_index, "wage_share"], linestyle="--", color="k", linewidth=1)
    ax1.fill_between(years, macro_lower["wage_share"], macro_upper["wage_share"], alpha=0.2, linestyle="--", color="tab:green", linewidth=1)
    ax1.set_ylim((0.73, 1.02))
    ax1.set_yticks((0.75, 0.8, 0.85, 0.9, 0.95, 1.0))
    ax1.set_ylabel("Wage Share", fontsize=fontsize + 2)

    ### profit share ###
    ax2.plot(years, macro_median["profit_share"], color="k", linewidth=1)
    ax2.plot(years, macro_data.loc[macro_data["simulation_index"]==sim_index, "profit_share"], linestyle="--", color="k", linewidth=1)
    ax2.fill_between(years, macro_lower["profit_share"], macro_upper["profit_share"], alpha=0.2, linestyle="--", color="tab:blue", linewidth=1)
    ax2.set_ylim((-0.01, 0.26))
    ax2.set_yticks((0.0, 0.05, 0.1, 0.15, 0.2, 0.25))
    ax2.set_ylabel("Profit Share", fontsize=fontsize + 2)
    ax2.set_xlabel("Years", fontsize=fontsize)

    # bottom title 
    # title = format_scenario_name(scenario["scenario_name"])
    full_title = f"({chr(97 + i)}) {format_scenario_name(scenario['scenario_name'])}"

    ax2.text(0.5, -0.25, full_title, transform=ax2.transAxes, ha="center", va="top", fontsize=fontsize + 5)

    # clean ticks
    for ax in [ax0, ax1, ax2]:
        ax.label_outer()
        ax.tick_params(labelsize=fontsize)

plt.savefig(figure_path / "gdp_shares.png", dpi=800, bbox_inches="tight")

### Plot Box Plots Scenarios ###

print("\nCreating box plots")

# scenario names
scenarios_short_names = ["G-S1", "G-S2", "ZG-S1", "ZG-S2"]

# macro data for all scenarios
macro_scenario_data = {
    short_name: load_macro_data(engine, scenario_id, params) for short_name, scenario_id in zip(scenarios_short_names, scenarios["scenario_id"])
}

# figure settings
xticks = [1, 2, 3, 4]
colours = ["tab:blue", "tab:blue", "tab:green", "tab:green"]

macro_vars = {
    "Real GDP Growth": "rgdp_growth", 
    "Productivity Growth": "productivity_growth", 
    "Inflation": "inflation", 
    "Wage Inflation": "wage_inflation", 
    "Interest Rate": "avg_loan_interest", 
    "Credit Rate": "credit_rate",
    "Unemployment Rate": "unemployment_rate", 
    "Gini Coefficient": "gini"
}

box_plot_scenarios(
    plot_data = macro_scenario_data,
    variables = macro_vars,
    scenarios_short_names = scenarios_short_names,
    ncols = 2,
    figsize = (x_figsize * 2, x_figsize * 2),
    fontsize = fontsize,
    colours = colours,
    figure_path = figure_path,
    figure_name = "box_plots_macro.png",
    whis = (lower*100, upper*100),
    sub_title_depth=0.15,
    dpi=800
)

### crisis probability ###
print("\nCrisis probability")
for short_name, macro_data in macro_scenario_data.items():
    # crisis flag per period
    macro_data["crises"] = (macro_data["rgdp_growth"] < -0.03).astype(int)
    # crisis occurrence per year per simulation
    yearly_crisis = (macro_data.groupby(["simulation_id", "year"])["crises"].max().reset_index(name="year_crisis"))
    # crisis probability per simulation
    crisis_prob_sim = (yearly_crisis.groupby("simulation_id")["year_crisis"].mean().reset_index(name="crisis_probability"))
    # print results
    print(f"- average across simulations: {crisis_prob_sim['crisis_probability'].mean():.2%}")
    print(f"- standard deviation across simulations: {crisis_prob_sim['crisis_probability'].std():.2%}")

### crisis severity (cumulative shortfall) ###
print("\nCrisis severity")
for short_name, macro_data in macro_scenario_data.items():
    # identify crisis spells
    macro_data["crisis_spell"] = (macro_data.groupby("simulation_id")["crises"].transform(lambda x: (x != x.shift()).cumsum()))
    # compute shortfall below -3%
    macro_data["growth_shortfall"] = np.where(macro_data["crises"] == 1, -0.03 - macro_data["rgdp_growth"],0)
    # sum shortfall within each crisis spell
    crisis_severity = (macro_data[macro_data["crises"] == 1].groupby(["simulation_id", "crisis_spell"])["growth_shortfall"].sum().reset_index(name="crisis_severity"))
    # average severity per simulation
    severity_sim = (crisis_severity.groupby("simulation_id")["crisis_severity"].mean().reset_index())
    # print results
    print(f"- average across simulations: {severity_sim['crisis_severity'].mean():.2%}")
    print(f"- standard deviation across simulations: {severity_sim['crisis_severity'].std():.2%}")

print("\nDiscussion Inflation Results")
for short_name, macro_data in macro_scenario_data.items():
    print(f"- {short_name} productivity growth:")
    avg_productivity_growth = macro_data["productivity_growth"].mean()
    print(f"  - average = {avg_productivity_growth:.4f}")
    print(f"- {short_name} wage inflation:")
    avg_wage_inflation = macro_data["wage_inflation"].mean()
    print(f"  - average = {avg_wage_inflation:.4f}")
    print(f"- {short_name} inflation:")
    avg_inflation = macro_data["inflation"].mean()
    print(f"  - average = {avg_inflation:.4f}")
    print(f"- {short_name} difference between wage inflation and productivity growth:")
    diff_inflation_productivity = avg_wage_inflation - avg_productivity_growth
    print(f"  - difference = {diff_inflation_productivity:.4f}\n")

print(f"\nFINISHED MACRO BATCH ANALYSIS! Check macro figures folder\n=> {figure_path}\n")
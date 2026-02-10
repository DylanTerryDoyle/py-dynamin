import numpy as np
import pandas as pd
from pathlib import Path
import scipy.stats as stats
import matplotlib.pyplot as plt
from dynamin.utils import load_config, sql_engine
from analysis.utils import load_macro_data, box_plot_scenarios

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

# current working directory path
analysis_path = Path(__file__).parent
# figure path
figure_path = analysis_path / "figures" / "stylised_facts"
# create figure path if it doesn't exist
figure_path.mkdir(parents=True, exist_ok=True)

### parameters ###

# parameters
params = load_config("parameters.yaml")
# analysis parameters
steps = params['simulation']['steps']
num_years = params['simulation']['years']
start = params['simulation']['start']*steps
years = np.linspace(0, num_years, num_years*steps)

# random simulation 
np.random.seed(params['simulation']['seed'])
sim_index = np.random.randint(0, params['simulation']['num_sims'])
print("Randomly selected simulation index for plots: ", sim_index)

# database parameters 
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

### start loop over databases ###

for i, scenario in scenarios.iterrows():
    # suffix for saving figures 
    suffix = scenario["scenario_name"]
    
    # get macro data 
    macro_data = load_macro_data(engine, scenario["scenario_id"], params)
        
    ### Plot GDP Ratios ###

    print(f"- creating GDP ratio figures")

    # average over simulations
    macro_group = macro_data.groupby(by="time")
    macro_median = macro_group.quantile(0.5)
    macro_upper = macro_group.quantile(upper)
    macro_lower = macro_group.quantile(lower)
    
    # create figure
    fig, axes = plt.subplots(3, 1, figsize=(x_figsize, x_figsize), sharex=True)
    
    # 1) debt ratio
    ax0 = axes[0]
    ax0.plot(years, macro_median["debt_ratio"], color='k', linewidth=1)
    ax0.plot(years, macro_data.loc[macro_data["simulation_index"]==sim_index, "debt_ratio"], linestyle="--", color="k", linewidth=1)
    ax0.fill_between(years, macro_lower["debt_ratio"], macro_upper["debt_ratio"], alpha=0.2, linestyle="--", color="tab:red", linewidth=1)
    ax0.set_ylim((0.45, 2.15))
    ax0.set_yticks((0.5, 1.0, 1.5, 2.0))
    # 2) wage share
    ax1 = axes[1]
    ax1.plot(years, macro_median["wage_share"], color='k', linewidth=1)
    ax1.plot(years, macro_data.loc[macro_data["simulation_index"]==sim_index, "wage_share"], linestyle="--", color="k", linewidth=1)
    ax1.fill_between(years, macro_lower["wage_share"], macro_upper["wage_share"], alpha=0.2, linestyle="--", color="tab:green", linewidth=1)
    ax1.set_ylim((0.65, 0.97))
    ax1.set_yticks((0.7, 0.8, 0.9))
    # 3) profit
    ax2 = axes[2]
    ax2.plot(years, macro_median["profit_share"], color='k', linewidth=1)
    ax2.plot(years, macro_data.loc[macro_data["simulation_index"]==sim_index, "profit_share"], linestyle="--", color="k", linewidth=1)
    ax2.fill_between(years, macro_lower["profit_share"], macro_upper["profit_share"], alpha=0.2, linestyle="--", color="tab:blue", linewidth=1)
    ax2.set_ylim((-0.05, 0.27))
    ax2.set_yticks((0.0, 0.1, 0.2))
    # tick size
    for ax in axes:
        ax.tick_params(labelsize=fontsize)
    # save fig 
    plt.tight_layout()
    plt.savefig(figure_path / f"gdp_shares_{suffix}", bbox_inches='tight')

### Plot Box Plots Scenarios ###

print("Creating box plots...")

# scenario names
scenarios_short_names = ["G1", "G2", "ZG1", "ZG2"]

# macro data for all scenarios
macro_scenario_data = {short_name: load_macro_data(engine, scenario_id, params) for short_name, scenario_id in zip(scenarios_short_names, scenarios["scenario_id"])}

# figure settings
xticks = [1, 2, 3, 4]
colours = ['tab:blue', 'tab:blue', 'tab:green', 'tab:green']

macro_vars = ["rgdp_growth", "productivity_growth", "inflation", "wage_inflation", "unemployment_rate", "gini", "credit_gdp", "yearly_crisis_prob"]

for var in macro_vars:
    box_plot_scenarios(
        macro_scenario_data,
        variable = var, 
        figsize = (x_figsize, y_figsize),
        fontsize = fontsize, 
        xlabels = scenarios_short_names,
        xticks = xticks,
        colours = colours,
        figure_path = figure_path
    )

print(f"\nFINISHED MACRO BATCH ANALYSIS! Check macro figures folder\n=> {figure_path}\n")
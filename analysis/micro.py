import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from dynamin.utils import load_config, sql_engine
from analysis.utils import load_macro_data, box_plot_scenarios

# is this analysis from the examples folder?
is_true = input('Run micro.py analysis for an example from the examples folder [y/n]: ').lower().startswith('y')

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
figure_path = analysis_path / "figures" / "micro"
# create figure path if it doesn't exist
figure_path.mkdir(parents=True, exist_ok=True)

### parameters ###

# parameters
if is_true:
    params = load_config(examples_path / example_folder / "config" / "parameters.yaml")
else:
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

### ESL/GDP plots ###

print(f"Creating ESL/GDP plots...")

for i, scenario in scenarios.iterrows():
    # suffix for saving figures 
    suffix = scenario["scenario_name"]
    
    # get macro data 
    macro_data = load_macro_data(engine, scenario["scenario_id"], params)
        
    ### Plot GDP Ratios ###

    # average over simulations
    macro_group = macro_data.groupby(by="time")
    macro_median = macro_group.quantile(0.5)
    macro_upper = macro_group.quantile(upper)
    macro_lower = macro_group.quantile(lower)
    
    # create figure
    plt.figure(figsize=(x_figsize, y_figsize))
    plt.plot(years, macro_median["esl_gdp"], color="k", linewidth=1)
    plt.plot(years, macro_data.loc[macro_data["simulation_index"]==sim_index, "esl_gdp"], linestyle="--", color="k", linewidth=1)
    plt.fill_between(years, macro_lower["esl_gdp"], macro_upper["esl_gdp"], color="grey", alpha=0.2, linewidth=1)
    plt.tight_layout()
    # plt.ylim((-0.05, 0.55))
    plt.yticks((0.0, 0.1, 0.2, 0.3, 0.4, 0.5),fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig(figure_path / f"esl_gdp_{suffix}", bbox_inches="tight")    
    

### Plot Box Plots Scenarios ###

print("Creating box plots...")

# scenario names
scenarios_short_names = ["G1", "G2", "ZG1", "ZG2"]

# macro data for all scenarios
macro_scenario_data = {short_name: load_macro_data(engine, scenario_id, params) for short_name, scenario_id in zip(scenarios_short_names, scenarios["scenario_id"])}

# figure settings
xticks = [1, 2, 3, 4]
colours = ['tab:blue', 'tab:blue', 'tab:green', 'tab:green']

micro_vars = ["cfirm_hpi", "kfirm_hpi", "bank_hpi", "cfirm_nhhi", "kfirm_nhhi", "bank_nhhi", "cfirm_probability_default_crisis0", "kfirm_probability_default_crisis0", "bank_probability_default_crisis0", "cfirm_probability_default_crisis1", "kfirm_probability_default_crisis1", "bank_probability_default_crisis1"]

for var in micro_vars:
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

print(f"FINISHED MICRO BATCH ANALYSIS! Check your micro figures folder\n=> {figure_path}\n")
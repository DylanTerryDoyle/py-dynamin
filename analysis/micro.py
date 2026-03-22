import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from dynamin.utils import load_config, sql_engine
from analysis.utils import load_micro_data, box_plot_scenarios

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
large_fontsize = 35
medium_fontsize = 25
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
start_years = params["simulation"]["start"]
start = start_years * steps
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

### Plot ESL/GDP Ratios ###

print(f"Creating ESL/GDP plots...")

# start loop over databases

for i, scenario in scenarios.iterrows():
    # suffix for saving figures 
    suffix = scenario["scenario_name"]
    
    # get macro data 
    micro_data = load_micro_data(engine, scenario["scenario_id"], params)

    # average over simulations
    micro_group = micro_data.groupby(by="time")
    micro_median = micro_group.quantile(0.5)
    micro_upper = micro_group.quantile(upper)
    micro_lower = micro_group.quantile(lower)

    # create figure
    plt.figure(figsize=(x_figsize, y_figsize))
    plt.plot(years, micro_median["esl_gdp"], color="k", linewidth=1)
    plt.plot(years, micro_data.loc[micro_data["simulation_index"]==sim_index, "esl_gdp"], linestyle="--", color="k", linewidth=1)
    plt.fill_between(years, micro_lower["esl_gdp"], micro_upper["esl_gdp"], color="grey", alpha=0.2, linewidth=1)
    plt.tight_layout()
    plt.ylim((-0.01, 0.11))
    plt.yticks((0.0, 0.02, 0.04, 0.06, 0.08, 0.1),fontsize=medium_fontsize)
    plt.xticks(fontsize=medium_fontsize)
    plt.savefig(figure_path / f"esl_gdp_{suffix}", bbox_inches="tight")


### Plot Box Plots Scenarios ###

print("Creating box plots...")

# scenario names
scenarios_short_names = ["G1", "G2", "ZG1", "ZG2"]

# macro data for all scenarios
micro_scenario_data = {short_name: load_micro_data(engine, scenario_id, params) for short_name, scenario_id in zip(scenarios_short_names, scenarios["scenario_id"])}

# figure settings
xticks = [1, 2, 3, 4]
colours = ["tab:blue", "tab:blue", "tab:green", "tab:green"]

micro_vars = [
    "cfirm_hpi", "kfirm_hpi", "bank_hpi", 
    "cfirm_nhhi", "kfirm_nhhi", "bank_nhhi", 
    "cfirm_probability_default_crises0", "kfirm_probability_default_crises0", "bank_probability_default_crises0", 
    "cfirm_probability_default_crises1", "kfirm_probability_default_crises1", "bank_probability_default_crises1"
]

for var in micro_vars:
    box_plot_scenarios(
        micro_scenario_data,
        variable = var, 
        figsize = (x_figsize, y_figsize),
        fontsize = large_fontsize, 
        xlabels = scenarios_short_names,
        xticks = xticks,
        colours = colours,
        figure_path = figure_path
    )

### Age Box Plots ###

# large firms market share quantile
q_large = 0.99
# small firms market share quantile
q_small = 0.50

print("Creating age plots...")

# Dictionaries for box plots
cfirm_large_ages = {}
cfirm_small_ages = {}
kfirm_large_ages = {}
kfirm_small_ages = {}
bank_large_ages = {}
bank_small_ages = {}

for short_name, scenario_id in zip(
    scenarios_short_names,
    scenarios["scenario_id"]
):

    # Load full cfirm dataset for this scenario
    cfirms = pd.read_sql_query(
        """
            SELECT 
                T3.scenario_id,
                T2.simulation_id,
                T2.simulation_index,
                T1.id,
                T1.age,
                T1.market_share,
                T1.time
            FROM firm_data AS T1
            JOIN simulations AS T2 
                ON T2.simulation_id = T1.simulation_id
            JOIN scenarios AS T3 
                ON T3.scenario_id = T2.scenario_id
            WHERE T3.scenario_id = %(scenario_id)s
                AND T1.firm_type = 'ConsumptionFirm'
                AND T1.time > %(start)s
            ORDER BY T2.simulation_index, T1.time ASC;
        """,
        engine,
        params={
            "scenario_id": scenario_id,
            "start": start
        }
    )

    # Load full kfirm dataset for this scenario
    kfirms = pd.read_sql_query(
        """
            SELECT 
                T3.scenario_id,
                T2.simulation_id,
                T2.simulation_index,
                T1.id,
                T1.age,
                T1.market_share,
                T1.time
            FROM firm_data AS T1
            JOIN simulations AS T2 
                ON T2.simulation_id = T1.simulation_id
            JOIN scenarios AS T3 
                ON T3.scenario_id = T2.scenario_id
            WHERE T3.scenario_id = %(scenario_id)s
                AND T1.firm_type = 'CapitalFirm'
                AND T1.time > %(start)s
            ORDER BY T2.simulation_index, T1.time ASC;
        """,
        engine,
        params={
            "scenario_id": scenario_id,
            "start": start
        }
    )

    # Load full bank dataset for this scenario
    banks = pd.read_sql_query(
        """
            SELECT 
                T3.scenario_id,
                T2.simulation_id,
                T2.simulation_index,
                T1.id,
                T1.age,
                T1.market_share,
                T1.time
            FROM firm_data AS T1
            JOIN simulations AS T2 
                ON T2.simulation_id = T1.simulation_id
            JOIN scenarios AS T3 
                ON T3.scenario_id = T2.scenario_id
            WHERE T3.scenario_id = %(scenario_id)s
                AND T1.time > %(start)s
            ORDER BY T2.simulation_index, T1.time ASC;
        """,
        engine,
        params={
            "scenario_id": scenario_id,
            "start": start
        }
    )

    # rescale cfirm ages in terms of start year
    cfirms["age"] = cfirms.groupby("simulation_id")["age"].transform(lambda x: num_years * ((x - x.min()) / (x.max() - x.min())))
    # rescale kfirm ages in terms of start year
    kfirms["age"] = kfirms.groupby("simulation_id")["age"].transform(lambda x: num_years * ((x - x.min()) / (x.max() - x.min())))
    # rescale bank ages in terms of start year
    banks["age"] = banks.groupby("simulation_id")["age"].transform(lambda x: num_years * ((x - x.min()) / (x.max() - x.min())))

    # Compute cfirm thresholds within scenario
    large_cfirm_threshold = cfirms["market_share"].quantile(q_large)
    small_cfirm_threshold = cfirms["market_share"].quantile(q_small)

    # Compute kfirm thresholds within scenario
    large_kfirm_threshold = kfirms["market_share"].quantile(q_large)
    small_kfirm_threshold = kfirms["market_share"].quantile(q_small)

    # Compute bank thresholds within scenario
    large_bank_threshold = banks["market_share"].quantile(q_large)
    small_bank_threshold = banks["market_share"].quantile(q_small)

    # print market share threshold value
    print(f"- scenario: {scenarios.loc[scenarios["scenario_id"] == scenario_id]["scenario_name"].iloc[0]}")
    print(f"  - large cfirms market share threshold: {large_cfirm_threshold}")
    print(f"  - small cfirms market share threshold: {small_cfirm_threshold}")
    print(f"  - large kfirms market share threshold: {large_kfirm_threshold}")
    print(f"  - small kfirms market share threshold: {small_kfirm_threshold}")
    print(f"  - large banks market share threshold: {large_bank_threshold}")
    print(f"  - small banks market share threshold: {small_bank_threshold}")

    # large ages
    cfirm_large_ages[short_name] = cfirms.loc[cfirms["market_share"] >= large_cfirm_threshold,["age"]]
    kfirm_large_ages[short_name] = kfirms.loc[kfirms["market_share"] >= large_kfirm_threshold,["age"]]
    bank_large_ages[short_name] = banks.loc[banks["market_share"] >= large_bank_threshold,["age"]]

    # rename large cfirm age variable
    for key in cfirm_large_ages:
        cfirm_large_ages[key] = cfirm_large_ages[key].rename(columns={"age": "cfirm_age_large"})
    # rename large kfirm age variable
    for key in kfirm_large_ages:
        kfirm_large_ages[key] = kfirm_large_ages[key].rename(columns={"age": "kfirm_age_large"})
    # rename large bank age variable
    for key in bank_large_ages:
        bank_large_ages[key] = bank_large_ages[key].rename(columns={"age": "bank_age_large"})

    # small ages
    cfirm_small_ages[short_name] = cfirms.loc[cfirms["market_share"] <= small_cfirm_threshold,["age"]]
    kfirm_small_ages[short_name] = kfirms.loc[kfirms["market_share"] <= small_kfirm_threshold,["age"]]
    bank_small_ages[short_name] = banks.loc[banks["market_share"] <= small_bank_threshold,["age"]]

    # rename small cfirm age variable
    for key in cfirm_small_ages:
        cfirm_small_ages[key] = cfirm_small_ages[key].rename(columns={"age": "cfirm_age_small"})
    # rename small kfirm age variable
    for key in kfirm_small_ages:
        kfirm_small_ages[key] = kfirm_small_ages[key].rename(columns={"age": "kfirm_age_small"})
    # rename small bank age variable
    for key in bank_small_ages:
        bank_small_ages[key] = bank_small_ages[key].rename(columns={"age": "bank_age_small"})

# Large cfirms
box_plot_scenarios(
    cfirm_large_ages,
    variable="cfirm_age_large",
    figsize=(x_figsize, y_figsize),
    fontsize=large_fontsize,
    xlabels=scenarios_short_names,
    xticks=xticks,
    colours=colours,
    figure_path=figure_path,
    ylim=(-5,105)
)

# Large kfirms
box_plot_scenarios(
    kfirm_large_ages,
    variable="kfirm_age_large",
    figsize=(x_figsize, y_figsize),
    fontsize=large_fontsize,
    xlabels=scenarios_short_names,
    xticks=xticks,
    colours=colours,
    figure_path=figure_path,
    ylim=(-5,105)
)

# Large banks
box_plot_scenarios(
    bank_large_ages,
    variable="bank_age_large",
    figsize=(x_figsize, y_figsize),
    fontsize=large_fontsize,
    xlabels=scenarios_short_names,
    xticks=xticks,
    colours=colours,
    figure_path=figure_path,
    ylim=(-5,105)
)

# Small cfirms
box_plot_scenarios(
    cfirm_small_ages,
    variable="cfirm_age_small",
    figsize=(x_figsize, y_figsize),
    fontsize=large_fontsize,
    xlabels=scenarios_short_names,
    xticks=xticks,
    colours=colours,
    figure_path=figure_path,
    ylim=(-5,105)
)

# Small kfirms
box_plot_scenarios(
    kfirm_small_ages,
    variable="kfirm_age_small",
    figsize=(x_figsize, y_figsize),
    fontsize=large_fontsize,
    xlabels=scenarios_short_names,
    xticks=xticks,
    colours=colours,
    figure_path=figure_path,
    ylim=(-5,105)
)

# Small banks
box_plot_scenarios(
    bank_small_ages,
    variable="bank_age_small",
    figsize=(x_figsize, y_figsize),
    fontsize=large_fontsize,
    xlabels=scenarios_short_names,
    xticks=xticks,
    colours=colours,
    figure_path=figure_path,
    ylim=(-5,105)
)

print(f"FINISHED MICRO BATCH ANALYSIS! Check your micro figures folder\n=> {figure_path}\n")
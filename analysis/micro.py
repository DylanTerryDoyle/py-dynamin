import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from dynamin.utils import load_config, sql_engine
from analysis.utils import load_micro_data, box_plot_scenarios, format_scenario_name, sub_boxplot

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

fig = plt.figure(figsize=(x_figsize * 2, x_figsize))

outer = gs.GridSpec(2, 2, figure=fig, wspace=0.2, hspace=0.4)

for i, scenario in scenarios.iterrows():

    row = i // 2
    col = i % 2

    inner = gs.GridSpecFromSubplotSpec(
        1, 1,
        subplot_spec=outer[row, col],
        hspace=0.05
    )

    micro_data = load_micro_data(engine, scenario["scenario_id"], params)

    # average over simulations
    micro_group = micro_data.groupby(by="time")
    micro_median = micro_group.quantile(0.5)
    micro_upper = micro_group.quantile(upper)
    micro_lower = micro_group.quantile(lower)

    ax = fig.add_subplot(inner[0])

    ### ESL/GDP ###
    ax.plot(years, micro_median["esl_gdp"], color="k", linewidth=1)
    ax.plot(years, micro_data.loc[micro_data["simulation_index"]==sim_index, "esl_gdp"], linestyle="--", color="k", linewidth=1)
    ax.fill_between(years, micro_lower["esl_gdp"], micro_upper["esl_gdp"], color="grey", alpha=0.2, linewidth=1)
    ax.set_ylim((-0.01, 0.11))
    ax.set_yticks((0.0, 0.02, 0.04, 0.06, 0.08, 0.1))
    ax.set_ylabel(r"ESL/GDP$_N$", fontsize=fontsize + 2)
    ax.set_xlabel("Years", fontsize=fontsize + 2)

    # bottom title 
    # title = format_scenario_name(scenario["scenario_name"])
    full_title = f"({chr(97 + i)}) {format_scenario_name(scenario['scenario_name'])}"

    ax.text(0.5, -0.25, full_title, transform=ax.transAxes, ha="center", va="top", fontsize=fontsize + 5)

    # clean ticks
    ax.label_outer()
    ax.tick_params(labelsize=fontsize)

plt.savefig(figure_path / "esl_gdp.png", dpi=800, bbox_inches="tight")


### Plot Box Plots Scenarios ###

print("Creating box plots...")

# scenario names
scenarios_short_names = ["G1", "G2", "ZG1", "ZG2"]

# macro data for all scenarios
micro_scenario_data = {short_name: load_micro_data(engine, scenario_id, params) for short_name, scenario_id in zip(scenarios_short_names, scenarios["scenario_id"])}

# figure settings
xticks = [1, 2, 3, 4]
colours = ["tab:blue", "tab:blue", "tab:green", "tab:green"]

micro_vars = {
    "C-Firm Instability": "cfirm_hpi",
    "K-Firm Instability": "kfirm_hpi",
    "Bank Instability": "bank_hpi",
    "C-Firm Concentration": "cfirm_nhhi",
    "K-Firm Concentration": "kfirm_nhhi",
    "Bank Concentration": "bank_nhhi",
    "C-Firm Defaults": "cfirm_probability_default_crises0",
    "K-Firm Defaults": "kfirm_probability_default_crises0",
    "Bank Defaults": "bank_probability_default_crises0",
    "C-Firm Crisis Defaults": "cfirm_probability_default_crises1",
    "K-Firm Crisis Defaults": "kfirm_probability_default_crises1",
    "Bank Crisis Defaults": "bank_probability_default_crises1"
}

ylabel_dict = {
    "cfirm_hpi": "HPI",
    "kfirm_hpi": "HPI",
    "bank_hpi": "HPI",
    "cfirm_nhhi": r"HHI$^*$",
    "kfirm_nhhi": r"HHI$^*$",
    "bank_nhhi": r"HHI$^*$",
    "cfirm_probability_default_crises0": "Pr(default|no crisis)",
    "kfirm_probability_default_crises0": "Pr(default|no crisis)",
    "bank_probability_default_crises0": "Pr(default|no crisis)",
    "cfirm_probability_default_crises1": "Pr(default|crisis)",
    "kfirm_probability_default_crises1": "Pr(default|crisis)",
    "bank_probability_default_crises1": "Pr(default|crisis)"
}

box_plot_scenarios(
    plot_data = micro_scenario_data,
    variables = micro_vars,
    scenarios_short_names = scenarios_short_names,
    ncols = 3,
    figsize = (x_figsize * 2, x_figsize * 1.75),
    fontsize = fontsize,
    colours = colours,
    ylabel_dict = ylabel_dict,
    whis = (lower*100, upper*100),
    wspace=0.3,
    hspace=0.35,
    sub_title_depth=0.15,
    figure_path = figure_path,
    figure_name = "box_plots_micro.png",
    dpi=800
)

### Age Box Plots ###

# large firms market share quantile
q_large = 0.99
# small firms market share quantile
q_small = 0.50

print("Creating age plots...")

# age box plots dictionary
age_data = {}

# dictionaries for large and small firms
cfirm_large_ages = {}
cfirm_small_ages = {}
kfirm_large_ages = {}
kfirm_small_ages = {}
bank_large_ages = {}
bank_small_ages = {}

for short_name, scenario_id in zip(scenarios_short_names, scenarios["scenario_id"]):

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

age_scenario_data = {}

# combine large and small ages into age_data dictionary for box plots
for scenario in scenarios_short_names:
    age_scenario_data[scenario] = pd.concat([
        cfirm_large_ages[scenario],
        kfirm_large_ages[scenario],
        bank_large_ages[scenario],
        cfirm_small_ages[scenario],
        kfirm_small_ages[scenario],
        bank_small_ages[scenario],
    ], axis=1)

age_vars = {
    "Large Consumption Firms": "cfirm_age_large",
    "Large Capital Firms": "kfirm_age_large",
    "Large Banks": "bank_age_large",
    "Small Consumption Firms": "cfirm_age_small",
    "Small Capital Firms": "kfirm_age_small",
    "Small Banks": "bank_age_small",
}

ylabel_dict = {
    "cfirm_age_large": "Age",
    "kfirm_age_large": "Age",
    "bank_age_large": "Age",
    "cfirm_age_small": "Age",
    "kfirm_age_small": "Age",
    "bank_age_small": "Age",
}

box_plot_scenarios(
    plot_data = age_scenario_data,
    variables = age_vars,
    scenarios_short_names = scenarios_short_names,
    ncols = 3,
    figsize = (x_figsize * 2, x_figsize),
    fontsize = fontsize,
    colours = colours,
    ylabel_dict = ylabel_dict,
    whis = (lower*100, upper*100),
    ylim_dict={value: (-5,105) for key, value in age_vars.items()},
    wspace=0.25,
    hspace=0.3,
    sub_title_depth=0.15,
    figure_path = figure_path,
    figure_name = "box_plots_age.png",
    dpi=800
)

print(f"FINISHED MICRO BATCH ANALYSIS! Check your micro figures folder\n=> {figure_path}\n")
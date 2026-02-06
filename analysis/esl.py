import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from numpy.typing import NDArray
# add parent directory to system path
from analysis.utils import load_yaml, sql_engine, bank_debtrank, expected_systemic_loss, load_macro_data

### Plot Parameters ###

# change matplotlib font to serif
plt.rcParams['font.family'] = ['serif']
# figure size
x_figsize = 10
y_figsize = x_figsize/2
# fontsize
fontsize = 25
# upper decile 
upper = 0.9
# lower decile
lower = 0.1

### Paths ###

# current working directory path
cwd_path = Path.cwd()

# analysis path
analysis_path = cwd_path / "analysis"

# figure path
figure_path = analysis_path / "figures" / "debt_rank"
# create figure path if it doesn't exist
figure_path.mkdir(parents=True, exist_ok=True)

# data path 
data_path = analysis_path / "data" / "debt_rank"
# create data path if it doesn't exist
data_path.mkdir(parents=True, exist_ok=True)

# parameters path 
params_path = cwd_path / "src" / "dynamin" / "config"

### parameters ###

# parameters
params = load_yaml(params_path /  "parameters.yaml")
# analysis parameters
steps = params['simulation']['steps']
num_years = params['simulation']['years']
start = params['simulation']['start']*steps
years = np.linspace(0, num_years, num_years*steps)

# database parameters 
db_params = load_yaml(params_path / "database.yaml")

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

def calculate_expected_systemic_loss(scenario_id: int, suffix: str, params: dict) -> NDArray:
    ### Data for DebtRank ###
    # get edges data
    print("- Downloading credit network edge data...")
    edges = pd.read_sql_query(
        """
        SELECT 
            T3.scenario_id, 
            T2.simulation_index,
            T1.*,
            T4.bank_defaults / %(num_banks)s AS bank_probability_default,
            T4.nominal_gdp
        FROM edge_data AS T1
        JOIN simulations AS T2 ON T2.simulation_id = T1.simulation_id
        JOIN scenarios AS T3 ON T3.scenario_id = T2.scenario_id
        LEFT JOIN macro_data AS T4 ON T4.simulation_id = T1.simulation_id AND T4.time = T1.time
        WHERE T3.scenario_id = %(scenario_id)s
        AND T1.time > %(start)s
        ORDER BY T3.scenario_id, T1.simulation_id, T1.time ASC
        """,
        engine,
        params={
            'scenario_id': scenario_id,
            'start': start,
            'num_banks': params["simulation"]["num_banks"]
        }
    )
    
    # ensure source and target are strings
    edges["source"] = edges["source"].astype(str)
    edges["target"] = edges["target"].astype(str)

    ### start debtrank calculation ###

    print(f"- Calculating DebtRank & ESL for scenario {suffix}...")
    # number of simulations 
    num_sims = len(edges['simulation_index'].unique())
    # number of periods per simulation
    num_periods = len(years)
    # array to hold expected systemic loss as a percentage of nominal GDP results
    esl_gdp = np.zeros(shape=(num_sims,num_periods))
    # simulation loop
    for s in tqdm(range(num_sims)):
        # time period loop for simulation s
        for t in range(num_periods):
            # edge data for current time period
            current_edges = edges.loc[(edges["simulation_index"] == s) & (edges["time"] == start + t+1)]
            # bank assets in period t
            bank_assets = current_edges.sort_values(by=["source"]).drop_duplicates(subset=["source"]).loc[:,["bank_assets"]].to_numpy().ravel()
            # firm assets in period t
            firm_assets = current_edges.sort_values(by=["target"]).drop_duplicates(subset=["target"]).loc[:,["firm_assets"]].to_numpy().ravel()
            # bank probability of default
            bank_probability_default = current_edges.sort_values(by=["target"]).drop_duplicates(subset=["source"]).loc[:,["bank_probability_default"]].to_numpy().ravel()
            print(bank_probability_default)
            # credit network loan matrix
            C = pd.pivot_table(current_edges, values=["loan"], index=["source"], columns=["target"]).fillna(0).to_numpy()
            # number of banks 
            num_banks, num_firms = C.shape
            # Bank loan vector: shape => (num_banks, 1)
            C_banks = C.sum(axis=1).reshape(-1,1)
            # Bank propagation matrix
            W_banks = np.divide(C, C_banks, out = np.zeros_like(C, dtype=np.float64), where = C_banks != 0)
            # Firm loan vector: shape => (num_firms, 1)
            C_firms = C.sum(axis=0).reshape(-1,1)
            # Firm propagation matrix
            W_firms = np.divide(C.T, C_firms, out=np.zeros_like(C.T, dtype=np.float64), where=C_firms != 0)
            # compute debtrank for both banks and firms if each bank fails 
            bank_dr, firm_dr = bank_debtrank(W_banks, W_firms, bank_assets, firm_assets, num_banks, num_firms)
            # expected systemic loss
            esl = expected_systemic_loss(bank_dr, firm_dr, bank_probability_default, bank_assets, firm_assets)
            # nominal GDP in period t
            nominal_gdp = edges.loc[(edges["simulation_index"] == s) & (edges["time"] == start + t+1)]["nominal_gdp"].iloc[0]
            print(nominal_gdp)
            # expected systemic loss as a percentage of nomnal GDP
            esl_gdp[s,t] = esl/nominal_gdp
        
    # return expected systemic loss
    return esl_gdp

for i, scenario in scenarios.iterrows():
    # suffix for saving figures 
    suffix = scenario["scenario_name"]
    
    # scenario id 
    scenario_id = scenario["scenario_id"]
    
    # calculate expected systemic loss (ESL) to GDP ratio
    esl_gdp = calculate_expected_systemic_loss(scenario_id, suffix, params)
    
    ### save results ###
    print("- Saving results to CSV file")
    df_esl_gdp = pd.DataFrame(esl_gdp).to_csv(data_path / f"esl_gdp_{suffix}.csv")
    
    ### plot results ###
    
    # median esl over simulations
    esl_median = np.quantile(esl_gdp, q=0.5, axis=0)
    # Top 9th decile
    esl_upper = np.quantile(esl_gdp, q=upper, axis=0)
    # Top 1st decile
    esl_lower = np.quantile(esl_gdp, q=lower, axis=0)
    
    plt.figure(figsize=(x_figsize,y_figsize))
    plt.plot(years, esl_median, color='k', linewidth=1, label='Mean')
    plt.fill_between(years, esl_median, esl_upper, color='grey', alpha=0.2, label='IDR')
    plt.fill_between(years, esl_median, esl_lower, color='grey', alpha=0.2)
    # legend
    plt.legend(loc='upper left', fontsize=fontsize)
    # ticks 
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    # y limit
    plt.ylim((-0.02, 0.27))
    # save figure
    plt.savefig(figure_path / f"esl_{suffix}.png", bbox_inches='tight')
    plt.close() 

    print(f"- Created plot of ESL/GDP for scenario {suffix}:")
    print(f"  => {figure_path}")

from pathlib import Path
from dynamin.simulator import run
from dynamin.utils import load_config

def run_pg_scenarios():
    # database parameters (dynamin config folder)
    db_params = load_config("database.yaml")
    
    # simulation parameters (dynamin config folder)
    params = load_config("parameters.yaml")

    # get file path to config folder
    config_path = Path(__file__).parent / "config"

    # scenario parameters (example config folder)
    pg_scenario_params = load_config(config_path / "pg_scenarios.yaml")
    
    # run model 
    run(
        params=params,
        db_params=db_params,
        scenario_params=pg_scenario_params,
        num_sims=params["simulation"]["num_sims"], 
        seed=params["simulation"]["seed"],
        parallel=True,
        n_workers=params["simulation"]["n_workers"],
        dynamic=False
    )

if __name__ == '__main__':
    run_pg_scenarios()
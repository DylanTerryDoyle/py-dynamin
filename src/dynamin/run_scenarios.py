from pathlib import Path
from dynamin.utils import load_yaml
from dynamin.simulator import run

def run_scenarios():
    # current working directory
    cwd_path = Path.cwd()

    ### Model Parameters ###
    params_path = cwd_path / "dynamin" / "config"

    # simulation parameters
    params = load_yaml(params_path / "parameters.yaml")

    # scenario parameters
    scenario_params = load_yaml(params_path / "scenarios.yaml")

    # database parameters
    db_params = load_yaml(params_path / "database.yaml")
    
    # run model 
    run(
        params=params,
        db_params=db_params,
        scenario_params=scenario_params,
        num_sims=params["simulation"]["num_sims"], 
        seed=params["simulation"]["seed"],
        parallel=True,
        n_workers=8,
        dynamic=False
    )

if __name__ == '__main__':
    run_scenarios()
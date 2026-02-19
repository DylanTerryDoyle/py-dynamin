from dynamin.simulator import run
from dynamin.utils import load_config

def run_baseline():
    # database parameters
    db_params = load_config("database.yaml")
    
    # simulation parameters
    params = load_config("parameters.yaml")
    
    # run model 
    run(
        params=params,
        db_params=db_params,
        num_sims=params["simulation"]["num_sims"], 
        seed=params["simulation"]["seed"],
        parallel=True,
        n_workers=params["simulation"]["n_workers"],
        dynamic=False
    )

if __name__ == '__main__':
    run_baseline()
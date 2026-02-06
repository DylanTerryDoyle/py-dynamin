### import libraries ###
import multiprocessing as mp
from tqdm import tqdm

# import model 
from dynamin.model import Model

# import data collector
from dynamin.data_collector import DataCollector

def _single_run(params: dict, db_params: dict, scenario_name: str | None=None, simulation_index: int=0, seed: int | None=None, dynamic: bool=False, progress_queue=None):
    """
    Helper to run one simulation in a separate process.
    """
    # instantiate data collector class
    dc = DataCollector(db_params)
    # connect to database
    dc.connect(db_params["database_name"])
    # instantiate model class
    model = Model(params)
    # update simulation data only
    dc.update_simulation_data(scenario_name, simulation_index, seed)
    # run model with progress callback
    if progress_queue is not None:
        model.run(seed, dc, progress_callback=lambda: progress_queue.put(1))
    else:
        model.run(seed, dc)
    # close database connection
    dc.close()
    
    return simulation_index

def _single_run_wrapper(args):
    """Wrapper to unpack arguments for parallel execution."""
    return _single_run(*args)

def _progress_monitor(progress_queue, total_steps, num_sims, scenario):
    """Monitor progress from all parallel simulations."""
    with tqdm(total=total_steps * num_sims, desc=f"Simulation Progress", position=0) as pbar:
        completed = 0
        while completed < total_steps * num_sims:
            try:
                msg = progress_queue.get(timeout=1)
                if msg == 'DONE':
                    break
                completed += 1
                pbar.update(1)
            except KeyboardInterrupt:
                # Allow clean exit on Ctrl+C
                break
            except:
                # Timeout or other non-critical errors - continue waiting
                continue

def _batch_run(
    params: dict, 
    db_params: dict, 
    scenario_name: str | None=None,
    num_sims: int=1,
    init_seed: int | None=None,
    parallel: bool=False, 
    n_workers: int | None=None,
    dynamic: bool=False
):
    """
    Run a batch of simulations.
    
    Parameters
    ----------
        params : dict
            dictionary of model parameters
        db_params : dict 
            dictionary of database parameters
        scenario : str | None
            scenario name
        num_sims : int
            number of simulations to run
        init_seed : int
            initial random seed value for reproducible results 
        parallel : bool
            flag for parallel computing 
        n_workers : int
            number of parallel workers, if left as None to uses all CPU cores or number of simulations
        dynamic : bool
            flag for dynamic data collection
    """    
    # random seeds
    seeds = [init_seed + i if init_seed is not None else None for i in range(num_sims)]

    # parallel run
    if parallel:
        n_workers = min(num_sims, n_workers or mp.cpu_count())
        print(f"Number of simulations: {num_sims} \nNumber of workers: {n_workers}\n")
        
        # Create progress queue and start monitor
        manager = mp.Manager()
        progress_queue = manager.Queue()
        
        # total number of steps per run
        total_steps = (params["simulation"]["start"] + params["simulation"]["years"]) * params["simulation"]["steps"]
                
        # Start progress monitor in separate process
        monitor = mp.Process(
            target=_progress_monitor, 
            args=(progress_queue, total_steps, num_sims, scenario_name),
            daemon=True  # Daemon process will terminate when main process exits
        )
        monitor.start()
        
        # Prepare arguments for each simulation
        args_list = [
            (params, db_params, scenario_name, i, seeds[i], dynamic, progress_queue)
            for i in range(num_sims)
        ]
        
        # Run simulations in parallel
        with mp.Pool(n_workers) as pool:
            pool.map(_single_run_wrapper, args_list)
        
        # Signal monitor to finish
        progress_queue.put('DONE')
        
        # Wait for monitor to finish (with timeout)
        monitor.join(timeout=5)
        if monitor.is_alive():
            monitor.terminate()
        
    # sequential run
    else:
        for i in tqdm(range(num_sims), desc=f"Simulation Progress", leave=True):
            _single_run(
                params=params,
                db_params=db_params,
                simulation_index=i,
                seed=seeds[i],
                dynamic=dynamic,
                progress_queue=None
            )

def _scenario_run(
    data_collector: DataCollector,
    params: dict,
    db_params: dict,
    scenario_params: dict | None,
    num_sims: int=1,
    init_seed: int | None=None,
    parallel: bool=False,
    n_workers: int | None=None,
    dynamic: bool=False
):
    """
    Run a set of scenarios, each with a batch of simulations.
    
    Parameters
    ----------
        params : dict
            dictionary of model parameters
        db_params : dict 
            dictionary of database parameters
        scenario_params : dict
            dictionary of scenario parameters
        num_sims : int
            number of simulations for batch
        init_seed : int
            initial random seed value for reproducible results 
        parallel : bool
            flag for parallel computing 
        n_workers : int
            number of parallel workers, if left as None to uses all CPU cores or number of simulations
        dynamic : bool
            flag for dynamic data collection
    """
    # scenario names 
    if scenario_params is not None:
        scenario_names = scenario_params["scenario_names"]
    else:
        scenario_names = [None]
    
    # run scenarios
    for i, scenario_name in enumerate(scenario_names):
        # update scenario database
        data_collector.connect(db_params["database_name"])
        data_collector.update_scenario_data(scenario_name)
        data_collector.close()
        # update parameters for this scenario
        if scenario_params is not None and scenario_name is not None:
            for section in scenario_params:
                if section not in params or not isinstance(scenario_params[section], dict):
                    continue
                common_keys = params[section].keys() & scenario_params[section].keys()
                for key in common_keys:
                    params[section][key] = scenario_params[section][key][i]
        
        print(f"\nRunning scenario: {scenario_name}\n")
        # run the batch
        _batch_run(
            params=params,
            db_params=db_params,
            scenario_name=scenario_name,
            num_sims=num_sims,
            init_seed=init_seed,
            parallel=parallel,
            n_workers=n_workers,
            dynamic=dynamic
        )

def run(
    params: dict,
    db_params: dict,
    scenario_params: dict | None=None,
    num_sims: int=1,
    seed: int | None=0,
    parallel: bool=False,
    n_workers: int | None=None,
    dynamic: bool=False
):
    """
    Run simulations of the model.
    
    Parameters
    ----------
        params : dict
            dictionary of model parameters
        db_params : dict 
            dictionary of database parameters
        scenario_params : dict | None
            dictionary of scenario parameters
        num_sims : int
            number of simulations for batch
        seed : int | None
            initial random seed value for reproducible results 
        parallel : bool
            flag for parallel computing 
        n_workers : int | None
            number of parallel workers, if left as None to uses all CPU cores or number of simulations
        dynamic : bool
            flag for dynamic data collection
    """
    # ensure database exists 
    temp_model = Model(params)
    dc = DataCollector(db_params)
    dc.create_database(db_params["database_name"])
    dc.initialise_database(temp_model, dynamic)
    dc.close()
    del temp_model
    
    _scenario_run(
        data_collector=dc,
        params=params,
        db_params=db_params,
        scenario_params=scenario_params,
        num_sims=num_sims,
        init_seed=seed,
        parallel=parallel,
        n_workers=n_workers,
        dynamic=dynamic
    )
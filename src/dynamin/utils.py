import yaml
from pathlib import Path
from sqlalchemy import create_engine

def load_yaml(file: Path | str) -> dict:
    """
    Load YAML file as dictionary.
    
    Parameters
    ----------
        filename : str
            name of YAML file to load
    
    Returns
    -------
        file_dict : dict
            YAML file loaded as dictionary 
    """
    with open(file, 'r') as f:
        file_dict = yaml.safe_load(f)
    return file_dict

def sql_engine(db_params: dict):
    # get database connection parameters
    user = db_params["user"]
    password = db_params["password"]
    host = db_params["host"]
    port = db_params["port"]
    database = db_params["database_name"]

    # database connection URL
    url = (
        f"postgresql+psycopg2://{user}:{password}"
        f"@{host}:{port}/{database}"
    )
    # return created database engine
    return create_engine(url)
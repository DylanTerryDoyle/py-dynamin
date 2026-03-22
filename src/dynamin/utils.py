import yaml
from pathlib import Path
from sqlalchemy import create_engine
from importlib.resources import files

def load_config(file: str | Path) -> dict:
    """
    Load a YAML config file as a dictionary.

    Behavior:
        - If a string is passed, treat it as a filename in the installed package `dynamin/config/`.
        - If a Path is passed, load the YAML from that path.

    Parameters
    ----------
    file : str or Path
        Name of the YAML file (loaded from installed package) or full Path.

    Returns
    -------
    config_dict : dict
        Parsed YAML contents.

    Raises
    ------
    FileNotFoundError
        If the file is not found.
    """
    if isinstance(file, Path):
        # explicit path provided
        yaml_path = file
    else:
        # treat as package-installed config
        try:
            yaml_path = files("dynamin").joinpath("config") / file
        except ModuleNotFoundError:
            raise FileNotFoundError(f"Package 'dynamin' not installed, cannot find {file}")

    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with yaml_path.open("r") as f:
        return yaml.safe_load(f)

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
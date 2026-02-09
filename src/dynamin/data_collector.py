### import libraries ###
import psycopg2
import numpy as np
from psycopg2.extras import execute_batch

# import model for typing
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from dynamin.model import Model

### DataCollector class ###
class DataCollector:
    """
    DataCollector
    =============
    
    Attributes
    ----------
        connection : connection 
            connection to prostgres database
            
        cursor : cursor 
            cursor to prostgres database
            
    Methods
    -------
        __init__(self, db_params: dict, scenario_name: str | None = None, sim_index: int = 0, seed: int | None = None) -> None
        
        initialise_database(self, database_name: str, model: Model) -> None
        
        update_database(self, model: Model, time: int) -> None
        
        close_database(self) -> None
    """
    
    def __init__(self, db_params: dict):
        self.db_params          = db_params
        self.connection         = None
        self.cursor             = None
        self.scenario_id        = None
        self.simulation_id      = None
        self.macro_attrs        = db_params["macro"]
        self.firm_attrs         = db_params["firms"]
        self.bank_attrs         = db_params["banks"]
        self.household_attrs    = db_params["households"]
    
    def create_database(self, database_name: str):
        """
        Check if database exists, create it if it doesn't.
        
        Parameters
        ----------
            database_name : str
                name of the database to check/create
        """
        try:
            # Connect to the default 'postgres' database
            temp_connection = psycopg2.connect(
                user="postgres",
                password=self.db_params["password"],
                host=self.db_params["host"],
                port=self.db_params["port"]
            )
            temp_connection.autocommit = True

            with temp_connection.cursor() as temp_cursor:
                temp_cursor.execute(
                    "SELECT 1 FROM pg_database WHERE datname = %s", (database_name,)
                )
                if temp_cursor.fetchone():
                    print(f"Database '{database_name}' already exists.")
                else:
                    temp_cursor.execute(f'CREATE DATABASE "{database_name}";')
                    print(f"Database {database_name} created successfully.")

        except Exception as e:
            print(f"Error creating database '{database_name}': {e}")

        finally:
            if 'temp_connection' in locals() and temp_connection:
                temp_connection.close()

    def drop_database(self, database_name: str):
        """
        Drop a PostgreSQL database if it exists.

        Parameters
        ----------
        db_params : dict
            Dictionary with keys: user, password, host, port
        database_name : str
            Name of the database to drop
        """
        # Connect to a "safe" database (usually postgres)
        conn = psycopg2.connect(
            user="postgres",
            host=self.db_params["host"],
            port=self.db_params["port"]
        )
        conn.autocommit = True  # MUST enable autocommit to DROP a database

        try:
            with conn.cursor() as cur:
                # Check if database exists
                cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (database_name,))
                if cur.fetchone():
                    # Terminate all connections to the database first
                    cur.execute(f"""
                        SELECT pg_terminate_backend(pid)
                        FROM pg_stat_activity
                        WHERE datname = %s
                        AND pid <> pg_backend_pid();
                    """, (database_name,))
                    
                    # Drop the database
                    cur.execute(f'DROP DATABASE "{database_name}";')
                    print(f"Database {database_name} dropped successfully.")
                else:
                    print(f"Database {database_name} does not exist.")
        finally:
            conn.close()
    
    def connect(self, database_name: str):
        """
        Connect to the specified database.
        
        Parameters
        ----------
            database_name : str
                name of the database to connect to
        """
        self.connection = psycopg2.connect(
            database=database_name,
            user=self.db_params["user"],
            password=self.db_params["password"],
            host=self.db_params["host"],
            port=self.db_params["port"],
        )
        self.connection.autocommit = False
        self.cursor = self.connection.cursor()
    
    def _initialise_scenarios_table(self):
        """
        Create scenarios table if it doesn't exist.
        """
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS scenarios (
                scenario_id SERIAL PRIMARY KEY,
                scenario_index INT UNIQUE NOT NULL,
                scenario_name TEXT UNIQUE NOT NULL
            );
        """)
        
    def _initialise_simulations_table(self):
        """
        Create simulations table if it doesn't exist.
        """
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS simulations (
                simulation_id SERIAL PRIMARY KEY,
                scenario_id INTEGER NOT NULL REFERENCES scenarios(scenario_id),
                simulation_index INTEGER NOT NULL,
                seed INTEGER,
                UNIQUE (scenario_id, simulation_index)
            );
        """)
        
    def _initialise_macro_table(self, model: 'Model', dynamic: bool=False):
        """
        Create macro data table with dynamic columns.
        
        Parameters
        ----------
            model : Model
                the model instance to collect data from
            dynamic : bool
                dynamically initialise table columns to all variables for each class
        """
        if dynamic:
            self.macro_attrs = [name for name, value in model.__dict__.items() if isinstance(value, np.ndarray) and not name.startswith("_")]
        
        columns = ",\n    ".join(f"{name} DOUBLE PRECISION NOT NULL" for name in self.macro_attrs)
        self.cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS macro_data (
                simulation_id INTEGER REFERENCES simulations(simulation_id),
                time INTEGER,
                {columns},
                PRIMARY KEY (simulation_id, time)
            );
        """)
    
    def _initialise_firm_table(self, model: 'Model', dynamic: bool=False):
        """
        Create firm_data table with dynamic columns.
        
        Parameters
        ----------
            model : Model
                the model instance to collect data from
            dynamic : bool
                dynamically initialise table columns to all variables for each class
        """
        if dynamic:
            temp_cfirm = model.cfirms[0]
            temp_kfirm = model.kfirms[0]
            cfirm_attrs = {name for name, value in temp_cfirm.__dict__.items() if isinstance(value, np.ndarray)}
            kfirm_attrs = {name for name, value in temp_kfirm.__dict__.items() if isinstance(value, np.ndarray)}
            self.firm_attrs = sorted(cfirm_attrs.union(kfirm_attrs))
        
        columns = ",\n    ".join(f"{name} DOUBLE PRECISION" for name in self.firm_attrs)
        self.cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS firm_data (
                simulation_id INTEGER REFERENCES simulations(simulation_id),
                time INTEGER,
                id INTEGER,
                firm_type TEXT,
                {columns},
                PRIMARY KEY (simulation_id, time, id)
            );
        """)
    
    def _initialise_household_table(self, model: 'Model', dynamic: bool=False):
        """
        Create household_data table with dynamic columns.
        
        Parameters
        ----------
            model : Model
                the model instance to collect data from
            dynamic : bool
                dynamically initialise table columns to all variables for each class
        """
        if dynamic:
            temp_household = model.households[0]
            self.household_attrs = [name for name, value in temp_household.__dict__.items() if isinstance(value, np.ndarray)]
        
        columns = ",\n    ".join(f"{name} DOUBLE PRECISION NOT NULL" for name in self.household_attrs)
        self.cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS household_data (
                simulation_id INTEGER REFERENCES simulations(simulation_id),
                time INTEGER,
                id INTEGER,
                employed BOOLEAN,
                {columns},
                PRIMARY KEY (simulation_id, time, id)
            );
        """)
    
    def _initialise_bank_table(self, model: 'Model', dynamic: bool=False):
        """
        Create bank_data table with dynamic columns.
        
        Parameters
        ----------
            model : Model
                the model instance to collect data from
            dynamic : bool
                dynamically initialise table columns to all variables for each class
        """
        if dynamic:
            temp_bank = model.banks[0]
            self.bank_attrs = [name for name, value in temp_bank.__dict__.items() if isinstance(value, np.ndarray)]
        columns = ",\n    ".join(f"{name} DOUBLE PRECISION NOT NULL" for name in self.bank_attrs)
        self.cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS bank_data (
                simulation_id INTEGER REFERENCES simulations(simulation_id),
                time INTEGER,
                id INTEGER,
                {columns},
                PRIMARY KEY (simulation_id, time, id)
            );
        """)
    
    def initialise_database(
        self, 
        model: 'Model',
        dynamic: bool=False
    ) -> None:
        """
        Initialises the database by creating tables if they don't exist
        and ensures scenario + simulation rows exist.
        
        Parameters
        ----------
            database_name : str
                name of the database to connect to
            model : Model
                the model instance to collect data from
            dynamic : bool
                dynamically initialise table columns to all variables for each class
        """
        # 1. Connect to the database
        self.connect(self.db_params["database_name"])

        # 2. initialise tables
        self._initialise_scenarios_table()
        self._initialise_simulations_table()
        self._initialise_macro_table(model, dynamic)
        self._initialise_firm_table(model, dynamic)
        self._initialise_household_table(model, dynamic)
        self._initialise_bank_table(model, dynamic)
        
        # 3. Commit the changes
        self.connection.commit()
        
    
    def _upsert(self, table: str, columns: list[str], rows: list[list], conflict_cols: list[str]):
        """
        Perform an upsert (insert or update) operation on the specified table.
        
        Parameters
        ----------
            table : str
                name of the table
            columns : list[str]
                list of column names
            rows : list[list]
                list of rows to insert/update
            conflict_cols : list[str]
                list of columns to check for conflicts
        """
        # 1. check for empty rows
        if not rows:
            return

        # 2. construct SQL query
        placeholders = ", ".join(["%s"] * len(columns))
        col_list = ", ".join(columns)
        conflict = ", ".join(conflict_cols)

        # 3. determine columns to update on conflict
        update_cols = [c for c in columns if c not in conflict_cols]
        assignments = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols)

        # 4. construct final SQL query
        sql = f"""
            INSERT INTO {table} ({col_list})
            VALUES ({placeholders})
            ON CONFLICT ({conflict})
            DO UPDATE SET {assignments};
        """

        # 5. execute batch upsert
        execute_batch(self.cursor, sql, rows)
    
    def _upsert_macro_data(self, model: 'Model', time: int, attrs: list[str] | None=None):
        """
        Upsert macro_data table.
        
        Parameters
        ----------
            model : Model
                the model instance to collect data from.
            time : int
                time period
        """
        if attrs is None:
            attrs = self.macro_attrs
        
        columns = ["simulation_id", "time"] + attrs
        rows = [[
            self.simulation_id,
            time,
            *[float(getattr(model, a)[time]) for a in attrs],
        ]]

        self._upsert(
            table="macro_data",
            columns=columns,
            rows=rows,
            conflict_cols=["simulation_id", "time"],
        )

    def _upsert_firm_data(self, model: 'Model', time: int, attrs: list[str] | None=None):
        """
        Upsert firm_data table.
        
        Parameters
        ----------
            model : Model
                the model instance to collect data from.
            time : int
                time period
        """
        if attrs is None:
            attrs = self.firm_attrs
        
        columns = ["simulation_id", "time", "id", "firm_type"] + attrs
        rows = []

        for firm in model.firms:
            rows.append([
                self.simulation_id,
                time,
                firm.id,
                firm.__class__.__name__,
                *[
                    float(getattr(firm, a)[time]) if hasattr(firm, a) else 0.0
                    for a in attrs
                ],
            ])

        self._upsert(
            table="firm_data",
            columns=columns,
            rows=rows,
            conflict_cols=["simulation_id", "time", "id"],
        )

    def _upsert_household_data(self, model: 'Model', time: int, attrs: list[str] | None=None):
        """
        Upsert household_data table.
        
        Parameters
        ----------
            model : Model
                the model instance to collect data from.
            time : int
                time period
        """
        if attrs is None:
            attrs = self.household_attrs
        
        columns = ["simulation_id", "time", "id", "employed"] + attrs
        rows = [
            [
                self.simulation_id,
                time,
                h.id,
                h.employed,
                *[float(getattr(h, a)[time]) for a in attrs],
            ]
            for h in model.households
        ]

        self._upsert(
            table="household_data",
            columns=columns,
            rows=rows,
            conflict_cols=["simulation_id", "time", "id"],
        )

    def _upsert_bank_data(self, model: 'Model', time: int, attrs: list[str] | None=None):
        """
        Upsert bank_data table. 
        
        Parameters
        ----------
            model : Model
                the model instance to collect data from.
            time : int
                time period
                
        """
        if attrs is None:
            attrs = self.bank_attrs
        
        columns = ["simulation_id", "time", "id"] + attrs
        rows = [
            [
                self.simulation_id,
                time,
                b.id,
                *[float(getattr(b, a)[time]) for a in attrs],
            ]
            for b in model.banks
        ]

        self._upsert(
            table="bank_data",
            columns=columns,
            rows=rows,
            conflict_cols=["simulation_id", "time", "id"],
        )
        
    def _upsert_scenarios(self, name: str | None=None):
        """
        Upsert (insert or update) scenarios table.
        
        Parameters
        ----------
            name : str
                name of scenario, defaults to "baseline" if None
        """
        if name is None:
            name = "baseline"
        
        # 1. Does this scenario already exist?
        self.cursor.execute(
            "SELECT scenario_id, scenario_index FROM scenarios WHERE scenario_name = %s",
            (name,)
        )
        row = self.cursor.fetchone()
        if row:
            self.scenario_id, scenario_index = row
            return

        # 2. New scenario: compute next dense scenario_index
        self.cursor.execute("SELECT COALESCE(MAX(scenario_index), 0) + 1 FROM scenarios;")
        next_index = self.cursor.fetchone()[0]

        # 3. Insert with explicit scenario_index, let SERIAL generate scenario_id
        self.cursor.execute(
            """
            INSERT INTO scenarios (scenario_index, scenario_name)
            VALUES (%s, %s)
            RETURNING scenario_id;
            """,
            (next_index, name)
        )
        self.scenario_id = self.cursor.fetchone()[0]

    def _get_scenario_id(self, name: str | None=None) -> None:
        """
        Find and return scenario id.

        Parameters
        ----------
        name : str | None
            scenario name, defaults to "baseline" if None
        """
        if name is None:
            name = "baseline"

        self.cursor.execute(
            """
            SELECT scenario_id, scenario_index, scenario_name
            FROM scenarios
            WHERE scenario_name = %s
            """,
            (name,)
        )
        row = self.cursor.fetchone()
        if row is None:
            raise RuntimeError(
                f"Scenario '{name}' not found in scenarios table. "
                "Scenarios must be created in the main process via update_scenario_data()."
            )
        # return scenario id (index 0 of row)
        return row[0]


    def _upsert_simulations(self, scenario_name: str | None=None, index: int=0):
        """
        Upsert (insert or update) simulations table.
        
        Parameters
        ----------
            scenario_name : str | None
                scenario name, defaults to "baseline" if None
            index : int | None
                simulation index, default is 0
        """
        if scenario_name is None:
            scenario_name = "baseline"
        # get scenario id 
        self.scenario_id = self._get_scenario_id(scenario_name)
        
        # Insert or do nothing if exists
        self.cursor.execute(
            """
            INSERT INTO simulations (scenario_id, simulation_index)
            VALUES (%s, %s)
            ON CONFLICT (scenario_id, simulation_index) DO NOTHING
            RETURNING simulation_id;
            """,
            (self.scenario_id, index)
        )
        row = self.cursor.fetchone()
        if row:
            self.simulation_id = row[0]
        else:
            # simulation already exists, get its ID
            self.cursor.execute(
                "SELECT simulation_id FROM simulations WHERE scenario_id=%s AND simulation_index=%s",
                (self.scenario_id, index)
            )
            self.simulation_id = self.cursor.fetchone()[0]

    def _upsert_seed(self, seed: int | None = None):
        """
        Upsert (insert or update) random seed to simulations table.
        
        Parameters
        ----------
            seed : int | None
                seed value 
        """
        # Update the seed
        self.cursor.execute(
            """
            UPDATE simulations
            SET seed = %s
            WHERE simulation_id = %s
            """,
            (seed, self.simulation_id)
        )
        
    def update_scenario_data(self, name: str | None) -> None:
        """
        Update scenarios data.
        
        Parameters
        ----------
            name : str | None
                name of scenario, defaults to "baseline" if None
        """
        self._upsert_scenarios(name)

    def update_simulation_data(self, scenario_name: str | None=None, simulation_index: int=0, seed: int | None = None):
        """
        Update simulation data such as seed in the database.
        
        Parameters
        ----------
            simulation_index : int | None
                simulation index; defaults to 0 if None
            seed : int | None
                seed value to update
        """
        self._upsert_simulations(scenario_name, simulation_index)
        self._upsert_seed(seed)
        self.connection.commit()    

    def update_agent_data(self, model: 'Model', time: int) -> None:
        """
        Saves data in PostgreSQL database.
        
        Parameters
        ----------
            model : Model
                The model instance to collect data from.
            time : int
                time period
        """
        self._upsert_macro_data(model, time)
        self._upsert_firm_data(model, time)
        self._upsert_household_data(model, time)
        self._upsert_bank_data(model, time)
        # commit changes
        self.connection.commit()
    
    def commit(self):
        """
        Commits the changes to the database.
        """
        self.connection.commit()

    def close(self):
        """
        Closes the database connection.
        """
        if self.connection:
            self.connection.commit()
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
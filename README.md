# Py-DYNAMIN

Python DYNamic Agent-based MINskyan (Py-DYNAMIN) model.

The Py-DYNAMIN model is a reusable agent-based macroeconomic modelling framework implemented in Python. It allows you to run baseline economic scenarios and post-growth scenario analysis for research and reproducibility purposes, the PG-DYNAMIN model. The model requires PostgreSQL for database operations and standard Python scientific libraries for analysis.

## Features

- Agent-based macroeconomic modeling
- Baseline scenario and post-growth scenarios
- PostgreSQL integration for data storage
- Analysis tools for macro, micro, and stylised facts
- Example scripts for academic reproducibility (PG-DYNAMIN scenario analysis)

## Installation

You can install the baseline model Py-DYNAMIN via pip from PyPI using:

```bash
pip install py-dynamin
```

Or, clone the repository for full access to the code, examples (PG-DYNAMIN) and analysis scripts:

'''bash 
git clone https://github.com/DylanTerryDoyle/py-dynamin.git
```

## Dependencies:

- Python >= 3.10

- Core: numpy, scikit-learn, psycopg2-binary, pyyaml, tqdm

- Optional for analysis: pandas, matplotlib, scipy, powerlaw, statsmodels

- Database: PostgreSQL must be installed and running, see: https://www.postgresql.org/ for installation details.

## Project Structure

```
py-dynamin/
├── LICENSE
├── README.md
├── pyproject.toml
├── src
│   └── dynamin
│        ├── __init__.py
│        ├── agents
│        │   ├── abtract_firm.py
│        │   ├── banks.py
│        │   ├── capital_firm.py
│        │   ├── consumption_firm.py
│        │   └── household.py
│        ├── config
│        │   ├── database.yaml
│        │   └── parameters.yaml
│        ├── baseline.py
│        ├── data_collector.py
│        ├── model.py
│        ├── simulator.py
│        └── utils.py
├── examples
│   └── post_growth_scenarios
│       ├── config
│       │   ├── database.yaml
│       │   ├── parameters.yaml
│       │   └── pg_scenarios.yaml
│       └── pg_scenarios.py
├── analysis
│   ├── __init__.py
│   ├── empirical_data
│   │   ├── CE16OV.csv
│   │   ├── GDPC1.csv
│   │   ├── GPDIC1.csv
│   │   ├── PCECC96.csv
│   │   ├── TBSDODNS.csv
│   │   └── UNRATE.csv
│   ├── esl.py
│   ├── macro.py
│   ├── micro.py
│   ├── stylised_facts.py
│   └── utils.py
└── docs
    └── DyNAMIN Model Description.pdf
```

- `src/dynamin`: core library for the baseline model.
- `src/dynamin/config`: baseline parameters and database configuration (database.yaml), update user/password to match your PostgreSQL setup.
- `examples/post_growth_scenarios`: parameters and scripts for running post-growth scenario analysis.
- `analysis`: scripts for running batch analyses of model results.

## Quick Start

1. Ensure PostgreSQL is installed and running. Update the `database.yaml` user/password fields to match your database credentials.
2. Install Py-DYNAMIN via pip:

```bash
pip install py-dynamin
```

3. Run baseline scenario (installed in package):

```bash
python -m dynamin.baseline
```

4. Run post-growth scenarios (from repository)

To run the post-growth scenario examples, clone the repository to access the `examples` folder:
```bash
git clone https://github.com/DylanTerryDoyle/py-dynamin.git
cd py-dynamin
python examples/post_growth_scenarios/pg_scenarios.py
```

5. Run analysis (from repository)

After running the simulations, you can analyze results using the analysis scripts:
```bash
git clone https://github.com/DylanTerryDoyle/py-dynamin.git
cd py-dynamin
python analysis/stylised_facts.py
python analysis/macro.py
python analysis/micro.py
```
Results will be printed to the terminal and figures can be found in the `analysis figures` folders

## Contributing

Contributions are welcome. Please fork the repository, make changes, and submit a pull request. Ensure that all new code includes tests where applicable.

## License

This project is licensed under the GNU Affero General Public License v3.0. See [LICENSE](LICENSE) for details.

## Contact

Dylan Terry-Doyle  
Email: d.c.terry-doyle@sussex.ac.uk
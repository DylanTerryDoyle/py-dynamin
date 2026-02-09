### Import Libraries ###
import copy
import numpy as np
from tqdm import tqdm
from numpy.typing import NDArray
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

### Import Agents ###
from dynamin.agents.bank import Bank
from dynamin.agents.household import Household
from dynamin.agents.capital_firm import CapitalFirm
from dynamin.agents.consumption_firm import ConsumptionFirm

# import data collector
from dynamin.data_collector import DataCollector

### Model Class ###
class Model:
    
    """
    Market class
    ============
    Runs the Dynamic Agent-based Model of the Macroeconomy (DAMM). 
    
    Attributes
    ----------
        params : dict
            model parameters
        
        simulations : int
            number of simulation
        
        steps : int
            number of time steps per year
        
        time : int
            total number of time steps 
        
        start : int
            simulation start cut-off
        
        dt : float
            time delta, inverse steps
        
        growth : float
            average growth rate
        
        num_households : int
            number of households in simulation
        
        num_banks : int
            number of banks in simulation
        
        num_cfirms : int
            number of consumption firms in simulation
        
        num_kfirms : int
            number of capital firm in simulation
        
        num_firms : int
            total number of firms in simulation
        
        cfirm_id : int
            current max cfirm id 
        
        kfirm_id : int
            current max kfirm id
        
        length : int
            length of banks dataset
        
        initial_output : float
            initial real gdp
    
    Data
    ----
        cfirm_data : DataFrame
            C-firm default data
        
        kfirm_data : DataFrame
            K-firm default data
        
        real_consumption : NDArray
            real consumption 
        
        nominal_consumption : NDArray
            nominal consumption
        
        real_investment : NDArray
            real investment
        
        nominal_investment : NDArray
            nominal investment
        
        real_gdp : NDArray
            real gross domestic product (GDP)
        
        nominal_gdp : NDArray
            nominal gross domestic product (GDP)
            
        esl : NDArray
            expected systemic loss (ESL)
        
        debt : NDArray
            total nominal debt
        
        profits : NDArray
            total nominal profits
        
        avg_cfirm_price : NDArray
            consumption good Paasche price index
        
        avg_kfirm_price : NDArray
            capital good Paasche price index
        
        cfirm_nhhi : NDArray
            C-firm normalised Herfindahl-Hirschman Index
        
        kfirm_nhhi : NDArray
            K-firm normalised Herfindahl-Hirschman Index
        
        cfirm_hpi : NDArray
            C-firm Hymer-Pashigian Instability Index
        
        kfirm_hpi : NDArray
            K-firm Hymer-Pashigian Instability Index
        
        cfirm_defaults : NDArray
            total C-firm bankruptcies per period
        
        kfirm_defaults : NDArray
            total K-firm bankruptcies per period
        
        wages : NDArray
            total household wages
        
        avg_wage : NDArray
            average household wages
        
        employment : NDArray
            total number of employed households
        
        unemployment_rate : NDArray
            unemployment rate
        
        gini : NDArray
            Gini Coefficient
        
        bank_nhhi : NDArray
            bank normalised Herfindahl-Hirschman Index
        
        bank_hpi : NDArray
            bank Hymer-Pashigian Instability Index
        
        avg_loan_interest : NDArray
            average bank loan interest rate
        
        bank_defaults : NDArray
            number of bank bankruptcies per period
    
    Methods
    -------
        __init__(self, params: dict) -> None
                
        new_entrants(self, t: int) -> None
        
        labour_market(self, t: int) -> None
        
        production(self, t: int) -> None
        
        consumption_market(self, t: int) -> None
        
        capital_market(self, t: int) -> None
        
        probability_default(self, t: int) -> None
        
        credit_market(self, t: int) -> None
        
        default_data(self, t: int) -> None
        
        bankruptcies(self, t: int) -> None
        
        market_shares(self, t: int) -> None
                
        copy_cfirm(self, cfirm: ConsumptionFirm) -> ConsumptionFirm
        
        copy_kfirm(self, kfirm: CapitalFirm) -> CapitalFirm
        
        compute_cfirm_probabilities(self, t: int) -> list[float]
        
        compute_kfirm_probabilities(self, t: int) -> list[float]
        
        compute_labour_probabilities(self, t: int) -> list[float]
        
        compute_loan_probabilities(self, t: int) -> list[float]
        
        compute_gini(self, t: int) -> float
        
        @staticmethod    
        bank_debtrank(W_banks: NDArray[np.float64], W_firms: NDArray[np.float64], bank_assets: NDArray[np.float64], firm_assets: NDArray[np.float64], num_banks: int, num_firms: int, max_iterations: int = 100, epsilon: float = 1e-8) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        
        compute_esl(self, t: int) -> float
        
        indicators(self, t: int) -> None
                
        step(self, t: int) -> None
        
        run(self, params: dict, init_seed: int | None = None) -> None
    """

    def __init__(self, params: dict) -> None:
        """
        Market class initialisation.
        
        Parameters
        ----------
            params : dict
                model parameters
        """
        ### Parameters ###
        self.params:                dict  = params
        self.simulations:           int   = params['simulation']['num_sims']
        self.steps:                 int   = params['simulation']['steps']
        self.time:                  int   = (params['simulation']['years'] + params['simulation']['start'])*self.steps + 1
        self.start:                 int   = params['simulation']['start']*self.steps
        self.dt:                    float = 1/self.steps
        self.num_households:        int   = params['simulation']['num_households']
        self.num_banks:             int   = params['simulation']['num_banks']
        self.num_cfirms:            int   = params['simulation']['num_cfirms']
        self.num_kfirms:            int   = params['simulation']['num_kfirms']
        self.num_firms:             int   = self.num_cfirms + self.num_kfirms
        self.cfirm_id:              int   = params['simulation']['num_kfirms']
        self.kfirm_id:              int   = params['simulation']['num_cfirms']
        self.length:                int   = params['bank']['length']
        self.initial_output:        float = np.floor(self.num_households/self.num_firms)
        
        ### probability of default data ### 
        # logistic regression data
        self._cfirm_X = np.zeros((self.length, 1))
        self._cfirm_y = np.zeros(self.length, dtype=np.int8)
        self._kfirm_X = np.zeros((self.length, 1))
        self._kfirm_y = np.zeros(self.length, dtype=np.int8)
        # circular buffer pointers
        self.cfirm_ptr = 0
        self.kfirm_ptr = 0
        self.cfirm_count = 0
        self.kfirm_count = 0
        
        ### Market Data ##
        self.real_consumption:      NDArray = np.zeros(shape=self.time)
        self.nominal_consumption:   NDArray = np.zeros(shape=self.time)
        self.real_investment:       NDArray = np.zeros(shape=self.time)
        self.nominal_investment:    NDArray = np.zeros(shape=self.time)
        self.real_gdp:              NDArray = np.zeros(shape=self.time)
        self.nominal_gdp:           NDArray = np.zeros(shape=self.time)
        self.esl:                   NDArray = np.zeros(shape=self.time)
        self.debt:                  NDArray = np.zeros(shape=self.time)
        self.profits:               NDArray = np.zeros(shape=self.time)
        self.avg_cfirm_price:       NDArray = np.zeros(shape=self.time)
        self.avg_kfirm_price:       NDArray = np.zeros(shape=self.time)
        self.cfirm_nhhi:            NDArray = np.zeros(shape=self.time)
        self.kfirm_nhhi:            NDArray = np.zeros(shape=self.time)
        self.cfirm_hpi:             NDArray = np.zeros(shape=self.time)
        self.kfirm_hpi:             NDArray = np.zeros(shape=self.time)
        self.cfirm_defaults:        NDArray = np.zeros(shape=self.time)
        self.kfirm_defaults:        NDArray = np.zeros(shape=self.time)
        self.wages:                 NDArray = np.zeros(shape=self.time)
        self.avg_wage:              NDArray = np.zeros(shape=self.time)
        self.employment:            NDArray = np.zeros(shape=self.time)
        self.unemployment_rate:     NDArray = np.zeros(shape=self.time)
        self.gini:                  NDArray = np.zeros(shape=self.time)
        self.bank_nhhi:             NDArray = np.zeros(shape=self.time)
        self.bank_hpi:              NDArray = np.zeros(shape=self.time)
        self.avg_loan_interest:     NDArray = np.zeros(shape=self.time)
        self.bank_defaults:         NDArray = np.zeros(shape=self.time)
        
        ### Initial values ### 
        self.real_gdp[0]            = self.initial_output*self.num_firms
        self.nominal_gdp[0]         = self.real_gdp[0]
        self.real_consumption[0]    = self.initial_output*self.num_cfirms
        self.real_investment[0]     = self.initial_output*self.num_kfirms
        self.avg_cfirm_price[0]     = params['firm']['price']
        self.avg_kfirm_price[0]     = params['firm']['price']
        
        ### initialise agents ###
        # initial values 
        debt_ratio = (self.params['cfirm']['d0'] + self.params['cfirm']['d1'] * self.params['firm']['growth'] + self.params['cfirm']['d2'] * self.params['cfirm']['acceleration'] * (self.params['firm']['growth'] + self.params['firm']['depreciation'])) / (1 + self.params['cfirm']['d2'] * self.params['firm']['growth'])
        profit_share = self.params['cfirm']['acceleration'] * (self.params['firm']['depreciation'] + self.params['firm']['growth']) - self.params['firm']['growth'] * debt_ratio
        initial_wage = 1 - profit_share - self.params['bank']['loan_interest'] * debt_ratio
        # initialse agent objects into lists 
        self.cfirms:       list[ConsumptionFirm] = [ConsumptionFirm(x, self.initial_output, initial_wage, self.params) for x in range(self.num_cfirms)]
        self.kfirms:       list[CapitalFirm] = [CapitalFirm(x, self.initial_output, initial_wage, self.params) for x in range(self.num_kfirms)]
        self.banks:        list[Bank] = [Bank(x, self.params) for x in range(self.num_banks)]
        self.households:   list[Household] = [Household(x, initial_wage, self.params) for x in range(self.num_households)] 
        # total firms list
        self.firms:        list[ConsumptionFirm | CapitalFirm] = self.cfirms + self.kfirms
        # initialse firm banks and employees
        for firm in self.firms:
            firm_bank: Bank = self.banks[np.random.choice(np.arange(self.num_banks))]
            firm_bank.add_deposit_firm(firm)
            firm_bank.add_loan_firm(firm)
            firm.compute_new_loan(firm.loans[0], firm_bank.loan_interest[0], firm_bank.id, 0)
            initial_employees = self.initial_output
            for household in self.households:
                if not household.employed:
                    firm.hire(household)
                    initial_employees -= 1
                    if initial_employees == 0:
                        break
        # initialise household banks
        for household in self.households:
            household_bank: Bank = self.banks[np.random.choice(np.arange(self.num_banks))]
            household_bank.add_household(household)
        # initialise banks customer data
        for bank in self.banks:
            bank.initialise_accounts()
        # intialise average makret wage and total debt
        self.avg_wage[0] = self.cfirms[0].wage[0]
        self.debt[0] = self.cfirms[0].debt[0]*self.num_cfirms
        
        ### initialise cfirm probability of default model ###
        # cfirm probability model 
        self.cfirm_pd_model = SGDClassifier(
            loss="log_loss",
            max_iter=10,
            warm_start=True
        )
        # kfirm probability model 
        self.kfirm_pd_model = SGDClassifier(
            loss="log_loss",
            max_iter=10,
            warm_start=True
        )
        # cfirm data 
        self.cfirm_scaler = StandardScaler()
        self.cfirm_scaler_initialised = False
        self.cfirm_model_initialised = False
        # kfirm data 
        self.kfirm_scaler = StandardScaler()
        self.kfirm_scaler_initialised = False
        self.kfirm_model_initialised = False
    
    def new_entrants(self, t: int) -> None:
        """
        New Consumption Firms and Capital Firms enter the market at a 1:1 ratio of the previous period bankrupt firms.
        New entrants are a random copy of encombent firms with prices and wages equal to their market averages and no debt.
        
        Parameters
        ----------
            t : int
                time period
        """
        # new cfirms 
        if self.cfirm_defaults[t-1] > 0:
            # randomly choose firms for new entrants to copy
            centrant_indices: list[np.int32] = list(np.random.choice(np.arange(len(self.cfirms)), size=int(self.cfirm_defaults[t-1]))) 
            for i in centrant_indices:
                # cfirm at index i
                cfirm = self.cfirms[i]
                # copy randomly chosen firm
                new_cfirm = self.copy_cfirm(cfirm)
                # create new cfirm id
                new_cfirm.id = self.cfirm_id
                # increase cfirm id
                self.cfirm_id += 1
                # initialise a single employee
                new_cfirm.employees = []
                for household in self.households:
                    if not household.employed:
                        new_cfirm.hire(household)
                        break
                # randomly choose a new bank
                cfirm_bank: Bank = self.banks[np.random.choice(np.arange(self.num_banks))]
                cfirm_bank.add_deposit_firm(new_cfirm)
                cfirm_bank.add_loan_firm(new_cfirm)
                # update entrant data
                new_cfirm.price[t-1] = self.avg_cfirm_price[t-1]
                new_cfirm.wage[t-1] = self.avg_wage[t-1]
                new_cfirm.loans[:] = 0
                new_cfirm.repayment[:] = 0
                new_cfirm.interest[:] = 0 
                new_cfirm.bank_ids[:] = np.nan
                new_cfirm.debt[:] = 0
                new_cfirm.age[:] = 0
                self.cfirms.append(new_cfirm) 
        # new kfirms
        if self.kfirm_defaults[t-1] > 0:
            # randomly choose firms for new entrants to copy
            kentrant_indices: list[np.int32] = list(np.random.choice(np.arange(len(self.kfirms)), size = int(self.kfirm_defaults[t-1]))) 
            for i in kentrant_indices:
                # cfirm at index i
                kfirm = self.kfirms[i]
                # copy randomly chosen firm
                new_kfirm = self.copy_kfirm(kfirm)
                # create new kfirm id
                new_kfirm.id = self.kfirm_id
                # increase kfirm id
                self.kfirm_id += 1
                # initialise a single employee
                new_kfirm.employees = []
                for household in self.households:
                    if not household.employed:
                        new_kfirm.hire(household)
                        break
                # randomly choose a new bank
                kfirm_bank: Bank = self.banks[np.random.choice(np.arange(self.num_banks))]
                kfirm_bank.add_deposit_firm(new_kfirm)
                kfirm_bank.add_loan_firm(new_kfirm)
                # update entrant data
                new_kfirm.price[t-1] = self.avg_kfirm_price[t-1]
                new_kfirm.wage[t-1] = self.avg_wage[t-1]
                new_kfirm.loans[:] = 0
                new_kfirm.repayment[:] = 0
                new_kfirm.interest[:] = 0 
                new_kfirm.bank_ids[:] = np.nan
                new_kfirm.debt[:] = 0
                new_kfirm.age[:] = 0
                self.kfirms.append(new_kfirm) 
        # update new total firms list
        self.firms = self.cfirms + self.kfirms 
    
    def labour_market(self, t: int) -> None:
        """
        A decentralised labour market opens. Firms update their vacancies and wages, households send job applications 
        to firms with open vacancies, then firms hire households if they require more labour.
        
        Parameters
        ----------
            t : int
                time period
        """
        # firm labour demand & wages
        for firm in self.firms:
            firm.determine_vacancies(t)
            firm.determine_wages(self.avg_wage[t-1], t)
        # probabilities for households to select firms to send job applications to 
        probabilities = self.compute_labour_probabilities(t)
        # households send job applications
        for household in self.households:
            household.determine_applications(self.firms, probabilities, t)
        # firms determine new labour for production
        for firm in self.firms:
            firm.determine_labour(t)

    def production(self, t: int) -> None:
        """
        Firms engage in production, they first update their productivity then produce goods on either the consumption or capital
        market. They then update their inventories, output growth, cost of production, and prices.
        
        Parameters
        ----------
            t : int
                time period
        """
        # Consumption firm output, productivity, costs & prices
        for cfirm in self.cfirms:
            cfirm.determine_productivity(t)
            cfirm.determine_output(t)
            cfirm.determine_inventories(t)
            cfirm.determine_output_growth(t)
            cfirm.determine_costs(t)
            cfirm.determine_prices(self.avg_cfirm_price[t-1], t)
        # Capital firm output, productivity, costs & prices
        for kfirm in self.kfirms:
            kfirm.determine_productivity(t)
            kfirm.determine_output(t)
            kfirm.determine_inventories(t)
            kfirm.determine_output_growth(t)
            kfirm.determine_costs(t)
            kfirm.determine_prices(self.avg_kfirm_price[t-1], t)

    def consumption_market(self, t: int) -> None:
        """
        A decentralised consumption good market opens. Households visit each C-firm and consumes consumption goods, 
        they then update their deposit with any remaining income.
        
        Parameters
        ----------
            t : int
                time period
        """
        # probabilities for households to select cfirms to consume cgoods from 
        probabilities = self.compute_cfirm_probabilities(t)
        # household consumption and deposits
        for household in self.households:
            household.determine_income(t)
            household.determine_desired_consumption(t)
            household.determine_consumption(self.cfirms, probabilities, t)
            household.determine_deposits(t)
    
    def capital_market(self, t: int) -> None:
        """
        A decentralised capital market opens. C-firms place orders of investment goods from K-firms.
        
        Parameters
        ----------
            t : int
                time period
        """
        # probabilities for cfirms to select kfirms to order kgoods from
        probabilities = self.compute_kfirm_probabilities(t) 
        # cfirm profits, investment, desired labour and desired loan
        for cfirm in self.cfirms:
            cfirm.determine_expected_demand(t)
            cfirm.determine_desired_output(t)
            cfirm.determine_profits(t)
            cfirm.determine_equity(t)
            cfirm.determine_desired_investment(t)
            cfirm.determine_investment(self.kfirms, probabilities, t)
            cfirm.determine_capital(t)
            cfirm.determine_desired_labour(t)
            cfirm.determine_desired_loan(t)
            cfirm.determine_leverage(t)
        # kfirm profits, desired labour and desired loan
        for kfirm in self.kfirms:
            kfirm.determine_expected_demand(t)
            kfirm.determine_desired_output(t)
            kfirm.determine_profits(t)
            kfirm.determine_equity(t)
            kfirm.determine_desired_labour(t)
            kfirm.determine_desired_loan(t)
            kfirm.determine_leverage(t)
            
    def probability_default(self, t: int) -> None:
        """
        Calculates the probability of firm defaults using a logistic regression model
        with standardised features.
        """

        ### cfirm probability of default estimation ###
        if self.cfirm_count >= 10 and np.unique(self._cfirm_y[:self.cfirm_count]).size > 1:

            # raw training data
            X_raw = self._cfirm_X[:self.cfirm_count, [0]]  # Only leverage (first column)
            y = self._cfirm_y[:self.cfirm_count]

            # update scaler
            if not self.cfirm_scaler_initialised:
                self.cfirm_scaler.fit(X_raw)
                self.cfirm_scaler_initialised = True
            else:
                self.cfirm_scaler.partial_fit(X_raw)

            # standardise training data
            X = self.cfirm_scaler.transform(X_raw)

            # fit classifier
            if not self.cfirm_model_initialised:
                self.cfirm_pd_model.partial_fit(X, y, classes=np.array([0, 1]))
                self.cfirm_model_initialised = True
            else:
                self.cfirm_pd_model.partial_fit(X, y)

            # prediction data (raw) - only leverage
            X_pred_raw = np.array([[cfirm.leverage[t]] for cfirm in self.cfirms])

            # standardise prediction data
            X_pred = self.cfirm_scaler.transform(X_pred_raw)

            # predict probabilities
            probs = self.cfirm_pd_model.predict_proba(X_pred)[:, 1]

            # assign probabilities
            for cfirm, p in zip(self.cfirms, probs):
                cfirm.probability_default[t] = float(p)

        ### kfirm probability of default estimation ###
        if self.kfirm_count >= 10 and np.unique(self._kfirm_y[:self.kfirm_count]).size > 1:

            # raw training data
            X_raw = self._kfirm_X[:self.kfirm_count, [0]]  # Only leverage (first column)
            y = self._kfirm_y[:self.kfirm_count]

            # update scaler
            if not self.kfirm_scaler_initialised:
                self.kfirm_scaler.fit(X_raw)
                self.kfirm_scaler_initialised = True
            else:
                self.kfirm_scaler.partial_fit(X_raw)

            # standardise training data
            X = self.kfirm_scaler.transform(X_raw)

            # fit classifier
            if not self.kfirm_model_initialised:
                self.kfirm_pd_model.partial_fit(X, y, classes=np.array([0, 1]))
                self.kfirm_model_initialised = True
            else:
                self.kfirm_pd_model.partial_fit(X, y)

            # prediction data (raw) - only leverage
            X_pred_raw = np.array([[kfirm.leverage[t]] for kfirm in self.kfirms])

            # standardise prediction data
            X_pred = self.kfirm_scaler.transform(X_pred_raw)

            # predict probabilities
            probs = self.kfirm_pd_model.predict_proba(X_pred)[:, 1]

            # assign probabilities
            for kfirm, p in zip(self.kfirms, probs):
                kfirm.probability_default[t] = float(p)
                
    def default_data(self, t: int) -> None:
        """
        Updates firm bankruptcy data to be used in the estimation of the probability of firm defaults.
        
        Parameters
        ----------
            t : int
                time period
        """
        # C-firms
        for cfirm in self.cfirms:
            # where to write next data point
            i = self.cfirm_ptr  
            self._cfirm_X[i, 0] = cfirm.leverage[t]
            # No longer storing amortisation_coverage in column 1
            self._cfirm_y[i] = int(cfirm.bankrupt)
            # update index pointer
            self.cfirm_ptr = (self.cfirm_ptr + 1) % self.length
            self.cfirm_count = min(self.cfirm_count + 1, self.length)

        # K-firms
        for kfirm in self.kfirms:
            # where to write next data point
            i = self.kfirm_ptr
            self._kfirm_X[i, 0] = kfirm.leverage[t]
            # No longer storing amortisation_coverage in column 1
            self._kfirm_y[i] = int(kfirm.bankrupt)
            # update index pointer
            self.kfirm_ptr = (self.kfirm_ptr + 1) % self.length
            self.kfirm_count = min(self.kfirm_count + 1, self.length)
        
    def credit_market(self, t: int) -> None:
        """
        A decentralised credit market opens. Firms visit banks and demand their desired loan, banks extend loans based on their 
        risk tolerence proxied by their capital ratio.
        
        Parameters
        ----------
            t : int
                time period
        """
        # banks update all data
        for bank in self.banks:
            # bank.determine_probability_default(t)
            bank.determine_loans(self.firms, t)
            bank.determine_deposits(t)
            bank.determine_profits(t)
            bank.determine_equity(t)
            bank.determine_reserves(t)
            bank.determine_capital(t)
            bank.determine_loan_interest(t)
        # probabilities for firms to select a bank to request a loan from
        probabilities = self.compute_loan_probabilities(t)
        # cfirm loan request and deposits
        for cfirm in self.cfirms:
            cfirm.determine_loan(self.banks, probabilities, t)
            cfirm.determine_deposits(t)
            cfirm.determine_default(t)
        # kfirm loan request and deposits
        for kfirm in self.kfirms:
            kfirm.determine_loan(self.banks, probabilities, t)
            kfirm.determine_deposits(t)
            kfirm.determine_default(t)

    def bankruptcies(self, t: int) -> None:
        """
        Firms go bankrupt if they have zero or negative deposits and exit their markets. 
        Banks also go bankrupt if they have zero or negative equity and are bailed out by their depositors.
        
        Parameters
        ----------
            t : int
                time period
        """
        # cfirm bankruptcies
        for cfirm in self.cfirms.copy(): 
            if cfirm.bankrupt:
                self.cfirm_defaults[t] += 1
                # banks update bad loans
                for bank in self.banks:
                    bank.determine_bad_loans(cfirm, t)
                # fire employees
                for household in cfirm.employees.copy():
                    cfirm.fire(household)
                # remove cfirm from simulation
                self.cfirms.remove(cfirm)
        # kfirm bankruptcies
        for kfirm in self.kfirms.copy():
            if kfirm.bankrupt:
                self.kfirm_defaults[t] += 1
                # banks update bad loans
                for bank in self.banks:
                    bank.determine_bad_loans(kfirm, t)
                # fire employees
                for household in kfirm.employees.copy():
                    kfirm.fire(household)
                # remove kfirm from simulation
                self.kfirms.remove(kfirm)
        # update firms list
        self.firms = self.cfirms + self.kfirms 
        # bank bankruptcies
        for bank in self.banks:
            bank.determine_default(self.avg_loan_interest[t-1], t)
            if bank.bankrupt:
                self.bank_defaults[t] += 1

    def market_shares(self, t: int) -> None:
        """
        Calculates C-firms, K-firms and banks market shares.
        
        Parameters
        ----------
            t : int
                time period
        """
        # final market shares for firms and banks
        coutput = sum(cfirm.output[t] for cfirm in self.cfirms)
        koutput = sum(kfirm.output[t] for kfirm in self.kfirms)
        for cfirm in self.cfirms:
            cfirm.determine_balance_sheet(t)
            cfirm.determine_market_share(coutput, t)
        for kfirm in self.kfirms:
            kfirm.determine_balance_sheet(t)
            kfirm.determine_market_share(koutput, t)
        loans = sum(bank.loans[t] for bank in self.banks)
        for bank in self.banks:
            bank.determine_balance_sheet(t)
            bank.determine_market_share(loans, t)

    def copy_cfirm(self, cfirm: ConsumptionFirm) -> ConsumptionFirm:
        """
        Copies a consumption firm object.
        
        Parameters
        ----------
            cfirm : ConsumptionFirm
                consumption firm object
        
        Returns
        -------
            copy of cfirm object : ConsumptionFirm
        """
        # Shallow copy of large data file
        memo = {id(cfirm.employees): cfirm.employees}
        # Deep copy of everything else
        new_cfirm = copy.deepcopy(cfirm, memo)  
        return new_cfirm
    
    def copy_kfirm(self, kfirm: CapitalFirm) -> CapitalFirm:
        """
        Copies a capital firm object.
        
        Parameters
        ----------
            kfirm : CapitalFirm
                capital firm object
        
        Returns
        -------
            copy of kfirm object : CapitalFirm
        """
        # Shallow copy of large data file
        memo = {id(kfirm.employees): kfirm.employees}
         # Deep copy of everything else
        new_kfirm = copy.deepcopy(kfirm, memo)
        return new_kfirm
    
    def compute_cfirm_probabilities(self, t: int) -> list[float]:
        """
        Calculates probability of households to visit cfirms.
        
        Parameters
        ----------
            t : int
                time period
        
        Returns
        -------
            probabilities : list[float]
        """
        total = sum(cfirm.output[t] for cfirm in self.cfirms)
        probabilities = [cfirm.output[t] / total for cfirm in self.cfirms]
        return probabilities
    
    def compute_kfirm_probabilities(self, t: int) -> list[float]:
        """
        Calculates probability of C-firms to visit K-firms.
        
        Parameters
        ----------
            t : int
                time period
        
        Returns
        -------
            probabilities : list[float]
        """
        total = sum(kfirm.output[t] for kfirm in self.kfirms)
        probabilities = [kfirm.output[t] / total for kfirm in self.kfirms]
        return probabilities
    
    def compute_labour_probabilities(self, t: int) -> list[float]:
        """
        Calculates probability of households to visit firms.
        
        Parameters
        ----------
            t : int
                time period
        
        Returns
        -------
            probabilities : list[float]
        """
        total = sum(firm.labour[t] for firm in self.firms)
        probabilities = [firm.labour[t] / total for firm in self.firms]
        return probabilities
    
    def compute_loan_probabilities(self, t: int) -> list[float]:
        """
        Calculates probability of firms to visit banks.
        
        Parameters
        ----------
            t : int
                time period
        
        Returns
        -------
            probabilities : list[float]
        """
        avg_loan = sum(bank.loans[t] for bank in self.banks) / self.num_banks
        total = sum(bank.loans[t] if bank.loans[t] > 1e-6 else avg_loan for bank in self.banks)
        probabilities = [bank.loans[t] / total if bank.loans[t] > 1e-6 else avg_loan / total for bank in self.banks]
        return probabilities
    
    def compute_gini(self, t: int) -> float:
        """
        Calculates the Gini Coefficient between household, measures the amount of inequality.
        
        Parameters
        ----------
            t : int
                time period
        
        Returns
        -------
            Gini Coefficient : float
        """
        # sorted wealth vector 
        wealth = np.sort([household.deposits[t] for household in self.households])
        # number of households
        n = wealth.shape[0]
        # index vector
        index = np.arange(1, n+1)
        # gini coefficient
        gini = ((np.sum((2 * index - n - 1) * wealth)) / (n * np.sum(wealth)))
        return gini
    
    @staticmethod    
    def bank_debtrank(
        W_banks: NDArray[np.float64],
        W_firms: NDArray[np.float64],
        bank_assets: NDArray[np.float64],
        firm_assets: NDArray[np.float64],
        num_banks: int,
        num_firms: int,
        max_iterations: int = 100,
        epsilon: float = 1e-6
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Description
        -----------
        Function to calculate DebtRank if each bank when bankrupt for a bank-firm bipartitie credit network (Battiston et al, 2012; Aoyama et al, 2013).
        
        References
        ----------
        
        Battiston, S., Puliga, M., Kaushik, R., Tasca, P., & Caldarelli, G. (2012). Debtrank: Too central to fail? 
        financial networks, the fed and systemic risk. Scientific reports, 2(1), 541.
            
        Aoyama, H., Battiston, S., & Fujiwara, Y. (2013). DebtRank analysis of the Japanese credit network. 
        Discussion papers, Research Institute of Economy, Trade and Industry (RIETI).
        
        Parameters
        ----------
            W_banks : NDArray[float64]
                bank propagation matrix

            W_firms: NDArray[float64]
                firm propagation matrix
            
            bank_assets: NDArray[float64]
                bank asset values
            
            firm_assets: NDArray[float64]
                firm asset values
            
            max_iterations: int
                maximum number of iterations per loop
            
            epsilon: float
                convergence tolerance
                
        Returns
        -------
            (debtrank_bank, debtrank_firm) : tuple(NDArray[Float64], NDArray[float64])
                a tuple of bank DebtRanks and firm DebtRanks when each bank becomes bankrupt
        """
        debtrank_bank = np.zeros(num_banks, dtype=np.float64)
        debtrank_firm = np.zeros(num_banks, dtype=np.float64)

        for i in range(num_banks):
            # initialise distress and state variables
            bank_distress = np.zeros(num_banks, dtype=np.float64)
            firm_distress = np.zeros(num_firms, dtype=np.float64)
            bank_state = np.zeros(num_banks, dtype=np.int8)
            firm_state = np.zeros(num_firms, dtype=np.int8)

            # initial shock: bank i distressed (bankrupt)
            bank_distress[i] = 1.0
            bank_state[i] = 1

            for _ in range(max_iterations):
                # record previous distress for convergence check
                prev_total = np.sum(bank_distress) + np.sum(firm_distress)

                ### update distress ###
                # calculate new firm distress from currently distressed banks
                distressed_banks = (bank_state == 1).astype(np.float64)
                firm_distress_new = np.minimum(1.0, firm_distress + W_firms @ (distressed_banks * bank_distress))

                # calculate new bank distress from currently distressed firms
                distressed_firms = (firm_state == 1).astype(np.float64)
                bank_distress_new = np.minimum(1.0, bank_distress + W_banks @ (distressed_firms * firm_distress))

                ### update states ###
                # undistressed firms to distressed (U -> D)
                firm_state[(firm_distress_new > 0) & (firm_state == 0)] = 1
                # distressed firms to inactive (D -> I)
                firm_state[firm_state == 1] = 2
                # update firm distress
                firm_distress = firm_distress_new

                # undistressed banks to distressed (U -> D)
                bank_state[(bank_distress_new > 0) & (bank_state == 0)] = 1
                # distressed banks to inactive (D -> I)
                bank_state[bank_state == 1] = 2
                # update bank distress
                bank_distress = bank_distress_new

                ### check convergence ###
                total = np.sum(bank_distress) + np.sum(firm_distress)
                if abs(total - prev_total) < epsilon:
                    break
            
            ### calculate DebtRank ###
            # bank-layer debtrank for bank i: exclude initially distressed bank i
            bank_mask = np.arange(num_banks) != i
            debtrank_bank[i] = np.sum(bank_distress[bank_mask] * bank_assets[bank_mask]) / np.sum(bank_assets[bank_mask])
            # firm-layer debtrank for bank i: include all firms
            debtrank_firm[i] = np.sum(firm_distress * firm_assets) / np.sum(firm_assets)

        return debtrank_bank, debtrank_firm
    
    def compute_esl(self, t: int) -> float:
        """
        Calculate expected systemic loss (ESL) using DebtRank for the current period.
        
        Parameters
        ----------
            t : int
                time period
        
        Returns
        -------
            esl : float
                Expected systemic loss (absolute value)
        """
        # Get current period data
        num_banks = self.num_banks
        num_firms = len(self.cfirms) + len(self.kfirms)
        
        # Bank assets: equity + reserves (total assets)
        bank_assets = np.array([bank.assets[t] for bank in self.banks])
        
        # Firm assets: equity (operating firms only)
        firm_assets = np.array([firm.assets[t] for firm in self.firms])
        
        # Bank probability of default (PD): use individual bank PD if available, else aggregate
        bank_pd = np.array([bank.defaults / t for bank in self.banks])
        
        # Build loan exposure matrix C (banks x firms)
        # Initialize sparse matrix (most banks don't lend to all firms)
        C = np.zeros((num_banks, num_firms))
        
        firm_idx = 0
        for firm in self.firms:
            for bank in self.banks:
                loan_amount = firm.compute_bank_loans(bank.id)
                if loan_amount > 0:
                    # Find bank index
                    bank_idx = bank.id
                    C[bank_idx, firm_idx] = loan_amount
            firm_idx += 1
        
        # Skip if no meaningful network
        if np.sum(C) == 0 or np.sum(bank_assets) == 0 or np.sum(firm_assets) == 0:
            return 0.0
        
        # Compute propagation matrices (handle zero divisions safely)
        # Bank total loans
        C_banks = C.sum(axis=1, keepdims=True)
        # Firm total loans
        C_firms = C.sum(axis=0, keepdims=True).T
        
        # Bank-to-firm propagation matrix W_banks
        W_banks = np.divide(C, C_banks, out=np.zeros_like(C), where=C_banks!=0)
        
        # Firm-to-bank propagation matrix W_firms (transpose)
        W_firms = np.divide(C.T, C_firms, out=np.zeros_like(C.T), where=C_firms!=0)
        
        # Compute DebtRank for each bank failure
        bank_dr, firm_dr = self.bank_debtrank(
            W_banks, 
            W_firms, 
            bank_assets, 
            firm_assets, 
            num_banks, 
            num_firms
        )
        
        # Total assets 
        total_bank_assets = np.sum(bank_assets)
        total_firm_assets = np.sum(firm_assets)
        
        # Expected Systemic Loss (Poledna et al., 2015) 
        esl = np.sum(bank_pd * (bank_dr * total_bank_assets + firm_dr * total_firm_assets))
         
        # return esl => ensure non-negative
        return float(max(0.0, esl)) 

    def indicators(self, t: int) -> None:
        """
        Calculates all macro indicators for the model.
        
        Parameters
        ----------
            t : int
                time period
        """
        # number of firms
        num_cfirms = len(self.cfirms)
        num_kfirms = len(self.kfirms)
        # Market Indicators
        self.real_consumption[t]        = sum(cfirm.output[t] for cfirm in self.cfirms)
        self.nominal_consumption[t]     = sum(cfirm.output[t] * cfirm.price[t] for cfirm in self.cfirms)
        self.real_investment[t]         = sum(kfirm.output[t] for kfirm in self.kfirms)
        self.nominal_investment[t]      = sum(kfirm.output[t] * kfirm.price[t] for kfirm in self.kfirms)
        self.real_gdp[t]                = self.real_consumption[t] + self.real_investment[t]
        self.nominal_gdp[t]             = self.nominal_consumption[t] + self.nominal_investment[t]
        self.debt[t]                    = sum(firm.debt[t] for firm in self.firms)
        self.profits[t]                 = sum(firm.profits[t] for firm in self.firms) + sum(bank.profits[t] for bank in self.banks)
        self.avg_cfirm_price[t]         = self.nominal_consumption[t] / self.real_consumption[t]
        self.avg_kfirm_price[t]         = self.nominal_investment[t] / self.real_investment[t]
        self.cfirm_nhhi[t]              = (sum(cfirm.market_share[t] ** 2 for cfirm in self.cfirms) - 1 / num_cfirms) / (1 - 1 / num_cfirms)
        self.kfirm_nhhi[t]              = (sum(kfirm.market_share[t] ** 2 for kfirm in self.kfirms) - 1 / num_kfirms) / (1 - 1 / num_kfirms)
        self.employment[t]              = sum(len(firm.employees) for firm in self.firms)
        self.unemployment_rate[t]       = (self.num_households - self.employment[t]) / self.num_households
        self.wages[t]                   = sum(firm.wage_bill[t] for firm in self.firms)
        self.avg_wage[t]                = self.wages[t] / self.employment[t]
        self.gini[t]                    = self.compute_gini(t)
        self.bank_nhhi[t]               = (sum(bank.market_share[t] ** 2 for bank in self.banks) - 1 / self.num_banks) / (1 - 1 / self.num_banks)
        self.avg_loan_interest[t]       = sum(bank.loan_interest[t] for bank in self.banks) / self.num_banks
        # expected systemic loss
        self.esl[t]                     = self.compute_esl(t)
        # lagged variables
        if t > self.steps:
            self.cfirm_hpi[t]               = sum(abs(cfirm.market_share[t] - cfirm.market_share[t-self.steps]) for cfirm in self.cfirms)
            self.kfirm_hpi[t]               = sum(abs(kfirm.market_share[t] - kfirm.market_share[t-self.steps]) for kfirm in self.kfirms)
            self.bank_hpi[t]                = sum(abs(bank.market_share[t] - bank.market_share[t-self.steps]) for bank in self.banks)
    
    def step(self, t: int) -> None:
        """
        Update model time step.
        
        Parameters
        ----------
            t : int 
                time period index
        """
        self.new_entrants(t)
        self.labour_market(t)
        self.production(t)
        self.consumption_market(t)
        self.capital_market(t)
        self.probability_default(t)
        self.credit_market(t)
        self.default_data(t)
        self.bankruptcies(t)
        self.market_shares(t)
        self.indicators(t)

    def run(self, seed: int | None=None, data_collector: DataCollector=None, progress_callback=None) -> None:
        """
        Runs a single simulation of the model.
        
        Parameters
        ----------
            seed : int | None
                random seed for the simulation
            data_collector : DataCollector | None
                data collector instance for saving results
            progress_callback : callable | None
                callback function to report progress (called after each step)
        """
        # random seed 
        if seed is not None:
            np.random.seed(seed)
        
        # start simulation with or without progress bar
        if progress_callback is not None:
            # Parallel mode - no individual progress bar, just callback
            for t in range(1, self.time):
                self.step(t)
                # update database 
                if data_collector is not None:
                    data_collector.update_agent_data(self, t)
                # report progress
                progress_callback()
        else:
            # Sequential mode - show progress bar
            for t in tqdm(range(1, self.time), desc="Steps", leave=False):
                self.step(t)
                # update database 
                if data_collector is not None:
                    data_collector.update_agent_data(self, t)
        
        # commit to database
        if data_collector is not None: 
            data_collector.commit()
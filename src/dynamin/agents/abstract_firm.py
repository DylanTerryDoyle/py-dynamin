### Import Libraries ###
import numpy as np
from numpy.typing import NDArray
from abc import ABC, abstractmethod

### Import Agents for Typing ###
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from dynamin.agents.household import Household
    from dynamin.agents.bank import Bank

### Abstract Firm Class ###
class AbstractFirm(ABC):
    
    """
    Firm Class
    ==========
    Base class of firm agent used to simulate firms in the model.
    
    Attributes
    ----------
        id : int
            bank object id
    
        steps : int
            number of time steps per year
        
        time : int
            total number of time steps
    
        dt : float  
            time delta, inverse steps
        
        adapt : float
            adaptive update speed
        
        adjust : float
            speed of adjustment 
        
        growth : float
            average firm productivity growth
        
        sigma : float
            firm productivity standard deviation
        
        sigma_p : float
            firm price standard deviation
        
        sigma_w : float
            firm wage standard deviation
        
        depreciation : float
            capital good depreciation rate

        num_banks : int
            number of banks visited
    
        deposit_interest : float
            bank deposit interest rate
        
        loan_periods : int
            number of repayment periods for a loan 
        
        repayment_rate : float
            loan repayment rate, inverse loan periods
        
        bankrupt : bool
            bankruptcy condition
    
    Data
    ----
        
        employees : list[Household]
            household employees
        
        applications : list[Household]
            applications from households
        
        productivity : NDArray
            labour productivity
        
        productivity_growth : NDArray
            labour productivity growth rate
        
        expected_productivity : NDArray
            expected labour productivity 
        
        output : NDArray
            production output 
        
        output_growth : NDArray
            output growth rate
        
        desired_output : NDArray
            desired output 
        
        demand : NDArray
            demand for goods
        
        expected_demand : NDArray
            expected demand for goods
        
        quantity : NDArray
            quantity of goods sold
        
        inventories : NDArray
            inventories of goods
        
        desired_inventories : NDArray
            desired inventories 
        
        labour : NDArray
            number of employees
        
        desired_labour : NDArray
            desired number of employees
        
        labour_demand : NDArray
            demand for new employees
        
        vacancies : NDArray
            number of vacant positions 
        
        wage : NDArray
            wage rate 
        
        wage_bill : NDArray
            wage bill (wage x labour)
        
        price : NDArray
            price of goods
        
        profits : NDArray
            profits 
        
        profit_share : NDArray
            profit share of revenue 
        
        equity : NDArray
            equity 
        
        deposits : NDArray
            deposits at bank
        
        desired_loan : NDArray
            desired loan from banks
        
        debt : NDArray
            debt
        
        total_repayment : NDArray
            total cost of loan repayment 
        
        total_interest : NDArray
            total cost of interest payments 
        
        leverage : NDArray
            leverage ratio
        
        probability_default : NDArray
            probability of defaulting 
        
        age : NDArray
            age in years
        
        market_share : NDArray
            market share 
        
        assets : NDArray
            value of assets
        
        liabilities : NDArray
            value of liabilities
        
        loans : NDArray
            current outstanding loans
        
        repayment : NDArray
            current repayment amount for each loan
        
        interest : NDArray
            current interest cost on each loan 
        
        bank_ids : NDArray
            bank id for each loan
    
    Methods 
    -------
        determine_vacancies(self, t: int) -> None
        
        determine_wages(self, avg_wage: float, t: int) -> None
        
        determine_labour(self, t: int) -> None
        
        determine_productivity(self, t: int) -> None
        
        determine_output(self, t: int) -> None
        
        determine_inventories(self, t: int) -> None
        
        determine_market_share(self, market_output: float, t: int) -> None
        
        determine_output_growth(self, t: int) -> None
        
        determine_costs(self, t: int) -> None
        
        determine_prices(self, avg_price: float, t: int) -> None
        
        determine_profits(self, t: int) -> None
        
        determine_equity(self, t: int) -> None
        
        determine_expected_demand(self, t: int) -> None
        
        determine_desired_output(self, t: int) -> None
        
        determine_desired_labour(self, t: int) -> None
        
        determine_desired_loan(self, t: int) -> None
        
        determine_leverage(self, t) -> None
        
        determine_loan(self, banks: list[Bank], probabilities: list[float], t: int) -> None
        
        determine_deposits(self, t: int) -> None
        
        determine_default(self, t: int) -> None
        
        determine_balance_sheet(self, t: int) -> None
        
        compute_total_debt(self, t: int) -> float
        
        compute_amortisation(self, loan: float, interest: float) -> float
        
        compute_new_loan(self, loan: float, interest: float, bank_id: int, t: int) -> None
        
        compute_total_repayment(self) -> float
        
        compute_total_interest(self) -> float
        
        compute_bank_loans(self, bank_id: int) -> float
        
        compute_bank_interest(self, bank_id: int) -> float
        
        hire(self, household: Household) -> None
        
        fire(self, household: Household) -> None
    """
    
    def __init__(self, id: int, initial_output: float, initial_wage: float, params: dict) -> None:
        """
        Firm class initialisation.
        
        Parameters
        ----------
            id : int
                unique id
        
            init_output : float
                initial output
        
            init_wage : float
                initial wage rate
        
            params : dict
                model parameters
        """
        # Parameters
        self.id:                        int   = id
        self.steps:                     int   = params['simulation']['steps']
        self.time:                      int   = (params['simulation']['years'] + params['simulation']['start']) * self.steps + 1
        self.dt:                        float = 1 / self.steps
        self.adapt:                     float = params['firm']['adapt'] * self.dt
        self.adjust:                    float = params['firm']['adjust'] * self.dt
        self.growth:                    float = params['firm']['growth'] * self.dt
        self.sigma:                     float = params['firm']['sigma'] * np.sqrt(self.dt)
        self.sigma_p:                   float = params['firm']['sigma_p'] * np.sqrt(self.dt)
        self.sigma_w:                   float = params['firm']['sigma_w'] * np.sqrt(self.dt)
        self.depreciation:              float = params['firm']['depreciation'] * self.dt
        self.wage_buffer:               float = params['firm']['wage_buffer']
        self.num_banks:                 int   = params['firm']['num_banks']
        self.deposit_interest:          float = params['bank']['deposit_interest'] * self.dt
        self.loan_periods:              int   = params['bank']['loan_years'] * self.steps
        self.repayment_rate:            float = 1 / self.loan_periods
        self.bankrupt:                  bool  = False
        # Mutable Data      
        self.employees:                 list['Household'] = []
        self.applications:              list['Household'] = []
        # Timeseries Data   
        self.productivity:              NDArray = np.zeros(shape=self.time)
        self.productivity_growth:       NDArray = np.zeros(shape=self.time)
        self.expected_productivity:     NDArray = np.zeros(shape=self.time)
        self.output:                    NDArray = np.zeros(shape=self.time)
        self.output_growth:             NDArray = np.zeros(shape=self.time)
        self.desired_output:            NDArray = np.zeros(shape=self.time)
        self.demand:                    NDArray = np.zeros(shape=self.time)
        self.expected_demand:           NDArray = np.zeros(shape=self.time)
        self.quantity:                  NDArray = np.zeros(shape=self.time)
        self.inventories:               NDArray = np.zeros(shape=self.time)
        self.desired_inventories:       NDArray = np.zeros(shape=self.time)
        self.labour:                    NDArray = np.zeros(shape=self.time)
        self.desired_labour:            NDArray = np.zeros(shape=self.time)
        self.labour_demand:             NDArray = np.zeros(shape=self.time)
        self.vacancies:                 NDArray = np.zeros(shape=self.time)
        self.wage:                      NDArray = np.zeros(shape=self.time)
        self.wage_bill:                 NDArray = np.zeros(shape=self.time)
        self.price:                     NDArray = np.zeros(shape=self.time)
        self.profits:                   NDArray = np.zeros(shape=self.time)
        self.profit_share:              NDArray = np.zeros(shape=self.time)
        self.equity:                    NDArray = np.zeros(shape=self.time)
        self.deposits:                  NDArray = np.zeros(shape=self.time)
        self.desired_loan:              NDArray = np.zeros(shape=self.time)
        self.debt:                      NDArray = np.zeros(shape=self.time)
        self.total_repayment:           NDArray = np.zeros(shape=self.time)
        self.total_interest:            NDArray = np.zeros(shape=self.time)
        self.leverage:                  NDArray = np.zeros(shape=self.time)
        self.probability_default:       NDArray = np.zeros(shape=self.time)
        self.age:                       NDArray = np.zeros(shape=self.time)
        self.market_share:              NDArray = np.zeros(shape=self.time)
        self.assets:                    NDArray = np.zeros(shape=self.time)
        self.liabilities:               NDArray = np.zeros(shape=self.time)
        # accounts      
        self.loans:                     NDArray = np.zeros(shape=self.time)
        self.repayment:                 NDArray = np.zeros(shape=self.time)
        self.interest:                  NDArray = np.zeros(shape=self.time)
        self.bank_ids:                  NDArray = np.zeros(shape=self.time)
        # Initial values        
        self.output[:]                  = initial_output
        self.desired_output[0]          = initial_output
        self.demand[0]                  = initial_output
        self.expected_demand[0]         = initial_output
        self.productivity[:]            = params['firm']['productivity']
        self.expected_productivity[0]   = params['firm']['productivity']
        self.desired_labour[0]          = initial_output / self.productivity[0]
        self.price[0]                   = params['firm']['price']
        self.wage[0]                    = initial_wage
        self.bank_ids[:]                = np.nan

    def determine_vacancies(self, t: int) -> None:
        """
        Calculate vacancies.
        
        Parameters
        ----------
            t : int
                time period
        """
        # reset applications list
        self.applications = []
        # calculate total employees/labour
        self.labour[t] = len(self.employees)
        # new labour demand
        self.labour_demand[t] = self.desired_labour[t-1] - self.labour[t]
        # number of vacancies
        self.vacancies[t] = max(self.labour_demand[t], 0)
        
    def determine_wages(self, avg_wage: float, t: int) -> None:
        """
        Calculate wage rate. 
        
        Parameters
        ----------
            avg_wage : float
                average market wage rate
            
            t : int
                time period
        """
        # update wage rate
        if self.labour_demand[t] >= 0:
            self.wage[t] = self.wage[t-1] * (1 + self.sigma_w * abs(np.random.randn())) + self.adjust * (avg_wage - self.wage[t-1])
        else:
            self.wage[t] = self.wage[t-1] * (1 - self.sigma_w * abs(np.random.randn())) + self.adjust * (avg_wage - self.wage[t-1])

    def determine_labour(self, t: int) -> None:
        """
        Hire or fire employees.
        
        Parameters
        ----------
            t : int
                time period
        """
        # firm hires new employees
        if self.labour_demand[t] > 0 and len(self.applications) > 0:
            # number of employees to hire
            num_households_hired = min(int(abs(self.labour_demand[t])), len(self.applications))
            # randomly choose household to hire from applications list
            hire_household_indices: list[np.int32] = list(np.random.choice(np.arange(len(self.applications)), size=num_households_hired, replace=False)) 
            # hire households from applications 
            for i in hire_household_indices:
                # household at index i
                household = self.applications[i]
                # if housheold is unemployed 
                if not household.employed:
                    # hire household
                    self.hire(household)
        # firm fires current employees 
        elif self.labour_demand[t] < 0:
            # number of employees to fire
            num_households_fired = min(int(abs(self.labour_demand[t])), len(self.employees)-1)
            # fire households from employees
            for _ in range(num_households_fired):
                # random household index
                index: np.int32 = np.random.choice(np.arange(len(self.employees)))
                # household at index i
                household = self.employees[index]
                # fire household
                self.fire(household)

    def determine_productivity(self, t: int) -> None:
        """
        Calculate productivity.
        
        Parameters
        ----------
            t : int
                time period
        """
        # update productivity
        self.productivity[t] = self.productivity[t-1] * np.exp(self.growth - 0.5 * (self.sigma ** 2) + self.sigma * np.random.randn())
        # annual productivity growth as the difference in the natural log
        self.productivity_growth[t] = np.log(self.productivity[t]) - np.log(self.productivity[t-self.steps])
        # expected productivity in the next period
        self.expected_productivity[t] = self.productivity[t] * np.exp(self.growth)
    
    @abstractmethod
    def determine_output(self, t: int) -> None:
        """
        Calculate output.
        
        Parameters
        ----------
            t : int
                time period
        """
        # To be overwritten
        pass

    @abstractmethod
    def determine_inventories(self, t: int) -> None:
        """
        Calculate inventories.
        
        Parameters
        ----------
            t : int
                time period
        """
        # To be overwritten
        pass 
    
    def determine_market_share(self, market_output: float, t: int) -> None:
        """
        Calculate market share.
        
        Parameters
        ----------
            market_output : float
                total market output (consumption or investment)
        
            t : int
                time period
        """
        if market_output != 0:
            self.market_share[t] = self.output[t] / market_output

    def determine_output_growth(self, t: int) -> None:
        """
        Calculate output growth rate.
        
        Parameters
        ----------
            t : int
                time period
        """
        # annual output growth as the difference in the natural log
        if self.output[t-self.steps] > 0 and t >= self.steps:
            self.output_growth[t] = np.log(self.output[t]) - np.log(self.output[t-self.steps])

    def determine_costs(self, t: int) -> None:
        """
        Calculate firm debt, total repayment, total interest, wage bill and pay employees.
        
        Parameters
        ----------
            t : int
                time period 
        """
        # compute total debt owed to all banks
        self.debt[t] = self.compute_total_debt(t)
        # compute total repayment costs
        self.total_repayment[t] = self.compute_total_repayment()
        # compute total interest costs
        self.total_interest[t] = self.compute_total_interest()
        # compte total wage bill
        self.wage_bill[t] = self.wage[t] * self.labour[t]
        # pay employees
        for household in self.employees:
            household.wage[t] = self.wage[t]

    def determine_prices(self, avg_price: float, t: int) -> None:
        """
        Calculate prices.
        
        Parameters
        ----------
            t : int
                time period
        """
        # update prices
        if self.desired_inventories[t-1] >= self.inventories[t-1]:
            self.price[t] = self.price[t-1] * (1 + self.sigma_p * abs(np.random.randn())) + self.adjust * (avg_price - self.price[t-1])
        else:
            self.price[t] = self.price[t-1] * (1 - self.sigma_p * abs(np.random.randn())) + self.adjust * (avg_price - self.price[t-1])

    def determine_profits(self, t: int) -> None:
        """
        Calculate profits and profit share.
        
        Parameters
        ----------
            t : int
                time period
        """
        # to be overwritten
        pass

    def determine_equity(self, t: int) -> None:
        """
        Calculate equity.
        
        Parameters
        ----------
            t : int 
                time period
        """
        # update equity with profits
        self.equity[t] = self.equity[t-1] + self.profits[t]

    def determine_expected_demand(self, t: int) -> None:
        """
        Calculate expected demand.
        
        Parameters
        ----------
            t : int 
                time period
        """
        # update expected demand in next period
        self.expected_demand[t] = self.expected_demand[t-1] + self.adapt * (self.demand[t] - self.expected_demand[t-1])

    @abstractmethod
    def determine_desired_output(self, t: int) -> None:
        """
        Calculate desired output.
        
        Parameters
        ----------
            t : int
                time period
        """
        # To be overwritten
        pass

    @abstractmethod
    def determine_desired_labour(self, t: int) -> None:
        """
        Calculate desired labour.
        
        Parameters
        ----------
            t : int 
                time period
        """
        # To be overwritten
        pass

    @abstractmethod
    def determine_desired_loan(self, t: int) -> None:
        """
        Calculate desired loan.
        
        Parameters
        ----------
            t : int
                time period
        """
        # To be overwritten
        pass

    def determine_leverage(self, t) -> None:
        """
        Calculate leverage ratio.
        
        Parameters
        ----------
            t : int
                time period
        """
        # expected debt in next period
        expected_debt = self.debt[t] * (1 - self.repayment_rate) + self.desired_loan[t]
        if self.deposits[t-1] + self.profits[t] > 0:
            # leverage ratio between (0,1)
            self.leverage[t] = expected_debt/(self.deposits[t-1] + self.profits[t] + expected_debt)
        else:
            self.leverage[t] = 1
    
    def determine_loan(self, banks: list['Bank'], probabilities: list[float], t: int) -> None:
        """
        Visit banks and demand loans, banks supply loans based on their credit risk.
        
        Parameters
        ----------
            banks : list[Bank]
                list of all banks
            
            probabilities : list[float]
                probability of visiting a given banks 
            
            t : int
                time period
        """
        # visit banks for a new loan if desired loan is positive
        if self.desired_loan[t] > 0:
            # randomly visit banks
            visited_bank_indices: list[np.int32] = list(np.random.choice(np.arange(len(banks)), size=self.num_banks, p=probabilities)) 
            # bank with lowest interest rate index
            i = sorted(visited_bank_indices, key=lambda i: banks[i].loan_interest[t], reverse=False)[0]
            # bank at index i
            bank = banks[i]
            # bank computes loan supplied to firm
            loan_supply = bank.compute_loan_supply(self, t)
            # if the banks choses to supply the loan  
            if loan_supply > 0:
                # bank adds firm to loan accounts
                bank.add_loan_firm(self)
                # firm computes new loan costs
                self.compute_new_loan(loan_supply, bank.loan_interest[t], bank.id, t)

    @abstractmethod
    def determine_deposits(self, t: int) -> None:
        """
        Calculate deposits.
                
        Parameters
        ----------
            t : int
                time period
        """
        # To be overwritten
        pass

    def determine_default(self, t: int) -> None:
        """
        Calculate age and bankruptcy condition.
        
        Parameters
        ----------
            t : int
                time period
        """
        # increase age by simulation time
        self.age[t] = self.age[t-1] + self.dt
        # bankruptcy condition when deposits are zero or negative
        if self.deposits[t] <= 1e-6:
            self.bankrupt = True
    
    @abstractmethod
    def determine_balance_sheet(self, t: int) -> None:
        """
        Calculate firm balance sheet, assets and liabilities.
        
        Parameters
        ----------
            t : int
                time period
        """
        # To be overwritten
        pass

    def compute_total_debt(self, t: int) -> float:
        """
        Calculate total debt.
        
        Parameters
        ----------
            t : int
                time period
        
        Returns
        -------
            total debt : float
        """
        # updates loans and compute total debt
        for tau in range(t):
            # update loan by repayment amount 
            self.loans[tau] = max(self.loans[tau] - self.repayment[tau], 0)
            # if the loan is repaid remove from accounts
            if self.loans[tau] <= 1e-6:
                self.repayment[tau] = 0
                self.interest[tau] = 0
                self.bank_ids[tau] = np.nan
        # total debt as sum of all outstanding loans
        return self.loans.sum()
    
    def compute_amortisation(self, loan: float, interest: float) -> float:
        """
        Calculate amortisation cost of a loan
        
        Parameters
        ----------
            loan : float
                loan amount
            
            interest : float
                interest rate on the loan
        
        Returns
        -------
            amortisation cost : float
        """
        # period interest rate 
        period_interest = interest * self.dt
        # amortisation cost each period
        return loan*((period_interest * (1 + period_interest) ** self.loan_periods) / ((1 + period_interest) ** self.loan_periods - 1))

    def compute_new_loan(self, loan: float, interest: float, bank_id: int, t: int) -> None:
        """
        Calculate new loan, repayment cost, interest cost, and bank id.
        
        Parameters
        ---------- 
            loan : float 
                loan amount
            
            interest : float 
                interest rate on the loan
            
            bank_id : int 
                id of the lender bank
            
            t : int 
                time period
        """
        # compute amortisation cost of the new loan
        amortisation = self.compute_amortisation(loan, interest)
        # compute repayment cost of the new loan
        repayment_amount = self.repayment_rate * loan
        # compute interest cost of the new loan
        interest_amount = amortisation - repayment_amount
        # add loan to accounts
        self.loans[t] = loan
        self.repayment[t] = repayment_amount
        self.interest[t] = interest_amount
        self.bank_ids[t] = bank_id
    
    def compute_total_repayment(self) -> float:
        """
        Calculate total repayment cost. 
        
        Returns
        -------
            total repayment cost : float
        """
        return self.repayment.sum()

    def compute_total_interest(self) -> float:
        """
        Calculate total interest cost.
        
        Returns
        -------
            total interest cost : float
        """
        return self.interest.sum()

    def compute_bank_loans(self, bank_id: int) -> float:
        """
        Calculate loan to a specific bank with id bank_id.
        
        Parameters
        ----------
            bank_id : int
                id of lender bank
        
        Returns
        -------
            loans to bank : float
        """
        return np.where(self.bank_ids == bank_id, self.loans, 0).sum() 
        
    def compute_bank_interest(self, bank_id: int) -> float:
        """
        Calculate interest payments to a specific bank with id bank_id.
        
        Parameters
        ----------
            bank_id : int
                id of lender bank
        
        Returns
        -------
            interest to bank : float
        """
        return np.where(self.bank_ids == bank_id, self.interest, 0).sum() 
    
    def hire(self, household: 'Household') -> None:
        """
        Hire household.
        
        Parameters
        ----------
            housheold : Household
                household to hire
        """
        household.employed = True
        self.employees.append(household)

    def fire(self, household: 'Household') -> None:
        """
        Fire household
        
        Parameters
        ----------
            household : Household
                household to fire
        """
        household.employed = False
        self.employees.remove(household)
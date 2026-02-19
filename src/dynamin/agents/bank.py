### Import Libraries
import numpy as np
from numpy.typing import NDArray

### Import Agents for Typing ###
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from dynamin.agents.household import Household
    from dynamin.agents.consumption_firm import ConsumptionFirm
    from dynamin.agents.capital_firm import CapitalFirm

### Bank Class ### 
class Bank:
    
    """
    Bank class
    ==========
    Bank agent class used to simulate banks in the model.
    
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
    
        adjust : float
            speed of adjustment 
        
        deposit_interest : float
            deposit interest rate
        
        natural_interest : float
            initial loan ineterst rate 
        
        sigma : float
            loan interest rate standard deviation
        
        loan_periods : int
            number of loan repayment periods 
    
        min_capital_ratio : float
            risk free capital ratio
        
        defaults : int
            total number of defaults
    
        bankrupt : bool
            bankruptcy condition
    
    Data
    ----
        households : list[Household]
            households with deposits at the bank
    
        loan_firms : list[ConsumptionFirm | CapitalFirm]
            firms with loans at the bank
    
        deposit_firms : list[ConsumptionFirm | CapitalFirm]
            firms with deposits at the bank
        
        profits : NDArray
            nominal profits
    
        loan_interest : NDArray
            loan interest rate
        
        equity : NDArray
            nominal equity 
        
        deposits : NDArray
            total deposits
        
        loans : NDArray
            total loans
    
        bad_loans : NDArray
            total bad loans per period
        
        expected_loss : NDArray
            expected bad loans 
        
        expected_loss_ratio : NDArray
            expected bad loans ratio to total loans
        
        reserves : NDArray
            nominal reserves
    
        reserve_ratio : NDArray
            reserve to deposit ratio
        
        capital_ratio : NDArray
            capital ratio (equity to loans ratio)
             
        desired_capital_ratio : NDArray
            minimum desired capital ratio
        
        market_share : NDArray
            market share of loans
        
        assets : NDArray
            value of assets
    
        liabilities : NDArray
            value of liabilities
        
        age : NDArray
            age of bank in years
    
    Methods
    -------
        initialise_accounts(self) -> None
        
        determine_loans(self, firms: list['ConsumptionFirm | CapitalFirm'], t: int) -> None
        
        determine_deposits(self, t: int) -> None
        
        determine_profits(self, t: int) -> None
        
        determine_equity(self, t: int) -> None
        
        determine_reserves(self, t: int) -> None
        
        determine_capital(self, t: int) -> None
        
        determine_loan_interest(self, t: int) -> None
        
        determine_bad_loans(self, bankrupt_firm: 'ConsumptionFirm | CapitalFirm', t: int) -> None
        
        determine_default(self, avg_loan_interest: float, t: int) -> None
        
        determine_market_share(self, total_loans: float, t: int) -> None
        
        compute_loan_supply(self, firm: 'Firm', t: int) -> float
        
        compute_total_loan_interest(self) -> float
        
        compute_total_loans(self) -> float
        
        compute_expected_loss(self, t: int) -> float
        
        compute_total_deposits(self, t: int) -> float
        
        add_household(self, household: 'Household') -> None
        
        remove_household(self, household: 'Household') -> None
        
        add_deposit_firm(self, firm: 'Firm | ConsumptionFirm | CapitalFirm') -> None
        
        remove_deposit_firm(self, firm: 'Firm | ConsumptionFirm | CapitalFirm') -> None
        
        add_loan_firm(self, firm: 'Firm | ConsumptionFirm | CapitalFirm') -> None
        
        remove_loan_firm(self, firm: 'Firm | ConsumptionFirm | CapitalFirm') -> None
        
        update_loan_firms(self, firms: list['ConsumptionFirm | CapitalFirm']) -> None
    """
    
    def __init__ (self, id: int, params: dict) -> None:
        """
        Bank class initialisation.
        
        Parameters
        ----------
            id : int
                unique id
            params : dict
                model parameters
        """
        # Parameters
        self.id:                        int   = id
        self.steps:                     int   = params['simulation']['steps']
        self.time:                      int   = (params['simulation']['years'] + params['simulation']['start']) * self.steps + 1
        self.dt:                        float = 1 / self.steps
        self.adjust:                    float = params['bank']['adjust'] * self.dt
        self.deposit_interest:          float = params['bank']['deposit_interest'] * self.dt
        self.natural_interest:          float = params['bank']['loan_interest']
        self.sigma:                     float = params['bank']['sigma'] * np.sqrt(self.dt)
        self.loan_periods:              int   = params['bank']['loan_years'] * self.steps
        self.min_capital_ratio:         float = params['bank']['min_capital_ratio']
        self.defaults:                  int   = 0
        self.bankrupt:                  bool  = False
        # Mutable data
        self.households:                list['Household'] = []
        self.loan_firms:                list['ConsumptionFirm | CapitalFirm'] = []
        self.deposit_firms:             list['ConsumptionFirm | CapitalFirm'] = []
        # Data 
        self.profits:                   NDArray = np.zeros(shape=self.time)
        self.loan_interest:             NDArray = np.zeros(shape=self.time)
        self.equity:                    NDArray = np.zeros(shape=self.time)
        self.deposits:                  NDArray = np.zeros(shape=self.time)
        self.loans:                     NDArray = np.zeros(shape=self.time)
        self.bad_loans:                 NDArray = np.zeros(shape=self.time)
        self.expected_loss:             NDArray = np.zeros(shape=self.time)
        self.expected_loss_ratio:       NDArray = np.zeros(shape=self.time)
        self.reserves:                  NDArray = np.zeros(shape=self.time)
        self.reserve_ratio:             NDArray = np.zeros(shape=self.time)
        self.capital_ratio:             NDArray = np.zeros(shape=self.time)
        self.desired_capital_ratio:     NDArray = np.zeros(shape=self.time)
        self.market_share:              NDArray = np.zeros(shape=self.time)
        self.assets:                    NDArray = np.zeros(shape=self.time)
        self.liabilities:               NDArray = np.zeros(shape=self.time)
        self.age:                       NDArray = np.zeros(shape=self.time)
        # Initial values
        self.loan_interest[0]           = params['bank']['loan_interest'] + params['firm']['inflation']

    def __repr__(self) -> str:
        """
        Returns printable unique bank agent representaton.
        
        Returns
        -------
            representation : str
        """
        return f'Bank: {self.id}'
    
    def __str__(self) -> str:
        """ 
        Returns printable agent type.
        
        Returns
        -------
            agent type : str
        """
        return 'Bank'

    def initialise_accounts(self) -> None:
        """
        Calculate inital values for time series.
        """
        self.loans[0] = self.compute_total_loans()
        self.deposits[0] = self.compute_total_deposits(0)
        self.desired_capital_ratio[0] = self.min_capital_ratio
        self.equity[0] = self.desired_capital_ratio[0] * self.loans[0]
        self.reserves[0] = self.deposits[0] + self.equity[0] - self.loans[0]
        self.reserve_ratio[0] = self.reserves[0] / self.deposits[0]
        self.profits[0] = self.compute_total_loan_interest() - self.deposit_interest * self.deposits[0]
    
    def determine_loans(self, firms: list['ConsumptionFirm | CapitalFirm'], t: int) -> None:
        """
        Update firms that still have outstanding loans, calculate total loans, expected bad loans, 
        and the expected bad loans ratio.
        
        Parameters
        ----------
            firms : list[ConsumptionFirm | CapitalFirm]
                list of all firms in the market
            
            t : int
                time period
        """
        # update loan firms, remove all firms that have repaid loans
        self.update_loan_firms(firms)
        # compute total loans extended to firms
        self.loans[t] = self.compute_total_loans()
        # compute expected bad loans, firm loans times probability of default
        self.expected_loss[t] = self.compute_expected_loss(t)
        if self.loans[t] != 0:
            # calculate expected bad loans as a ratio of total loans
            self.expected_loss_ratio[t] = self.expected_loss[t] / self.loans[t]

    def determine_deposits(self, t: int) -> None:
        """
        Calculate total deposits of firms and households.
        
        Paremeters
        ----------
            t : int
                time period 
        """
        # compute firm and household all deposits
        self.deposits[t] = self.compute_total_deposits(t-1)

    def determine_profits(self, t: int) -> None:
        """
        Calculate total bank profits.

        Parameters
        ----------
            t : int
                time period
        """
        # profits as the difference between loan interest and deposit interest
        self.profits[t] = self.compute_total_loan_interest() - self.deposit_interest * self.deposits[t]
        
    def determine_equity(self, t: int) -> None:
        """
        Calculate bank equity.
        
        Parameters
        ----------
            t : int
                time period
        """
        # update equity with profits
        self.equity[t] = self.equity[t-1] + self.profits[t] 
        
    def determine_reserves(self, t: int) -> None:
        """
        Calculate bank reserves and reserve to deposit ratio.
        
        Parameters
        ----------
            t : int
                time period
        """
        # calculate reserves from accounting identity
        self.reserves[t] = self.deposits[t] + self.equity[t] - self.loans[t]
        if self.deposits[t] != 0:
            # reserve as a ratio of total deposits
            self.reserve_ratio[t] = self.reserves[t] / self.deposits[t]

    def determine_capital(self, t: int) -> None:
        """
        Calculate bank capital ratio and minimum desired capital ratio.
        
        Parameters
        ----------
            t : int
                time period
        """
        # capital adequacy ratio, capital (equity) as a ratio of loans (assets) 
        if self.loans[t] != 0:
            self.capital_ratio[t] = self.equity[t] / self.loans[t]
        # minimum capital adequacy ratio, function of risk (expected bad loans ratio)
        self.desired_capital_ratio[t] = max(self.min_capital_ratio, self.expected_loss_ratio[t])

    def determine_loan_interest(self, inflation: float, t: int) -> None:
        """
        Calculate bank loan interest rate.
        
        Parameters
        ----------
            t : int 
                time period
        """
        loan_interest_target = max(inflation + self.natural_interest, self.natural_interest)
        # update loan interest rate
        if self.desired_capital_ratio[t] >= self.capital_ratio[t] and self.loans[t] > 1e-6:
            self.loan_interest[t] = self.loan_interest[t-1] * (1 + self.sigma * abs(np.random.randn())) + self.adjust * (loan_interest_target - self.loan_interest[t-1])
        else:
            self.loan_interest[t] = self.loan_interest[t-1] * (1 - self.sigma * abs(np.random.randn())) + self.adjust * (loan_interest_target - self.loan_interest[t-1])

    def determine_bad_loans(self, bankrupt_firm: 'ConsumptionFirm | CapitalFirm', t: int) -> None:
        """
        Calculate bank bad loans from bankrupt_firm.
        
        Parameters
        ----------
            bankrupt_firm : ConsumptionFirm | CapitalFirm
                bankrupt firm object
            
            t : int
                time period
        """
        # remove bankrupt firm from deposit accounts
        if bankrupt_firm in self.deposit_firms:
            self.remove_deposit_firm(bankrupt_firm)
        # remove bankrupt firm from loan accounts
        if bankrupt_firm in self.loan_firms:
            self.remove_loan_firm(bankrupt_firm)
            # all loans extended to bankrupt firm
            firm_loans = bankrupt_firm.compute_bank_loans(self.id)
            # increase bad loans 
            self.bad_loans[t] += firm_loans
            # reduce equity as banks absorb all risk
            self.equity[t] -= firm_loans

    def determine_default(self, avg_loan_interest: float, t: int) -> None:
        """
        Determine if bank is bankrupt, bail out if the bank is bankrupt.
        
        Parameters
        ----------
            avg_loan_interest : float
                average market loan interest rate
            
            t : int
                time period
        """
        # bankrupcty condition when equity is zero or negative
        if self.equity[t] <= 1e-6:
            self.bankrupt = True
            self.defaults += 1
            self.age[t] = 0
            # bail out amount
            bail_out = self.desired_capital_ratio[t] * (self.loans[t] + self.reserves[t])
            # increase equity by bail out amoun
            self.equity[t] = bail_out
            # depositors pay for bail out
            depositors = self.deposit_firms + self.households
            for depositor in depositors:
                depositor_cost = bail_out * (depositor.deposits[t] / self.deposits[t])
                depositor.deposits[t] -= depositor_cost
            # reset loan interest in industry averae
            self.loan_interest[t] = avg_loan_interest
        # no bankruptcy
        else:
            self.age[t] = self.age[t-1] + self.dt
            self.bankrupt = False
            
    def determine_balance_sheet(self, t: int) -> None:
        # calculate assets
        self.assets[t] = self.loans[t] + self.reserves[t]
        # calculate liabilities
        self.liabilities[t] = self.deposits[t]
    
    def determine_market_share(self, total_loans: float, t: int) -> None:
        """
        Calculate bank loan market share.
        
        Parameters
        ----------
            total_loans : float
                total market loans
        
            t : int
                time period
        """
        self.market_share[t] = self.loans[t] / total_loans

    def compute_loan_supply(self, firm: 'ConsumptionFirm | CapitalFirm', t: int) -> float:
        """
        Calculates and returns loan supply to the input firm demand a loan.
        
        Parameters
        ----------
            firm : Firm
                firm object
            
            t : int
                time period
        
        Returns
        -------
            loan_supply : float
                loan extended to firm object
        """
        # loan supply condition
        if self.desired_capital_ratio[t] < self.capital_ratio[t] or self.loans[t] <= 1e-6:
            loan_supply = firm.desired_loan[t]
        else:
            loan_supply = 0
        return loan_supply

    def compute_total_loan_interest(self) -> float:
        """
        Calculates and returns total loan interest collected from all firms.
        
        Returns
        -------
            loan_interest : float
                total loan interest paid by firms
        """
        loan_interest = sum(firm.compute_bank_interest(self.id) for firm in self.loan_firms)
        return loan_interest
    
    def compute_total_loans(self) -> float:
        """
        Calculates and returns total outstanding loans.
        
        Returns
        -------
            loans : float
                total outstanding loans to firms 
        """
        loans = sum(firm.compute_bank_loans(self.id) for firm in self.loan_firms)
        return loans
    
    def compute_expected_loss(self, t: int) -> float:
        """
        Calculates and returns expected loss.
        
        Parameters
        ----------
            t : int
                time period
        
        Returns
        -------
            exp_loss : float
                banks expected loss
        """
        exp_loss = sum(firm.compute_bank_loans(self.id) * firm.probability_default[t] for firm in self.loan_firms)
        return exp_loss
    
    def compute_total_deposits(self, t: int) -> float:
        """
        Calculates and returns total deposits held by firms and households.
        
        Parameters
        ----------
            t : int
                time period
        
        Returns
        -------
            deposits : float
                total deposits of firms and households
        """
        firm_deposits = sum(firm.deposits[t] for firm in self.deposit_firms)
        household_deposits = sum(household.deposits[t] for household in self.households)
        deposits = firm_deposits + household_deposits
        return deposits
    
    def add_household(self, household: 'Household') -> None:
        """
        Add household to list of deposit households.
        
        Parameters
        ----------
            household : Household
                household object 
        """
        if household not in self.households:
            self.households.append(household)

    def remove_household(self, household: 'Household') -> None:
        """
        Remove household from list of deposit households.
        
        Parameters
        ----------
            household : Household
                household object 
        """
        if household in self.households:
            self.households.remove(household)

    def add_deposit_firm(self, firm: 'ConsumptionFirm | CapitalFirm') -> None:
        """
        Add firm to list of deposit firms.
        
        Parameters
        ----------
            firm : Firm | ConsumptionFirm | CapitalFirm
                firm object 
        """
        if firm not in self.deposit_firms:
            self.deposit_firms.append(firm)

    def remove_deposit_firm(self, firm: 'ConsumptionFirm | CapitalFirm') -> None:
        """
        Remove firm from list of deposit firms.
        
        Parameters
        ----------
            firm : Firm | ConsumptionFirm | CapitalFirm
                firm object 
        """
        if firm in self.deposit_firms:
            self.deposit_firms.remove(firm)

    def add_loan_firm(self, firm: 'ConsumptionFirm | CapitalFirm') -> None:
        """
        Add firm to list of loan firms.
        
        Parameters
        ----------
            firm : Firm | ConsumptionFirm | CapitalFirm
                firm object 
        """
        if firm not in self.loan_firms:
            self.loan_firms.append(firm)

    def remove_loan_firm(self, firm: 'ConsumptionFirm | CapitalFirm') -> None:
        """
        Remove firm from list of loan firms.
        
        Parameters
        ----------
            firm : Firm | ConsumptionFirm | CapitalFirm
                firm object 
        """
        if firm in self.loan_firms:
            self.loan_firms.remove(firm)

    def update_loan_firms(self, firms: list['ConsumptionFirm | CapitalFirm']) -> None:
        """
        Update list of loan firms.
        
        Parameters
        ----------
            firms : list[ConsumptionFirm | CapitalFirm]
                list of all firms with loans
        """
        for firm in firms:
            firm_loans = firm.compute_bank_loans(self.id)
            if firm_loans <= 1e-6:
                self.remove_loan_firm(firm)
            else:
                self.add_loan_firm(firm)
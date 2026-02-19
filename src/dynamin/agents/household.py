### Import Libraries ###
import numpy as np
from numpy.typing import NDArray

### Import Agents for Typing ###
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from dynamin.agents.consumption_firm import ConsumptionFirm
    from dynamin.agents.capital_firm import CapitalFirm

### Household Class ###
class Household:
    
    """
    Household Class
    ===============
    Household agent class used to simulate households in the model.
    
    Parameters
    ----------
        id : int
            unique id
        
        init_wage : float
            initial market wage rate
        
        params : dict
            model parameters
    
    Attributes
    ----------
        id : int
            unique id
    
        steps : int
            number of time steps per year
        
        time : int
            total number of time steps 
    
        dt : float
            time delta, inverse steps
    
        mpc_deposits : float
            marginal propensity to consume out of deposits
    
        num_firms : int
            number of applications sent on the labour market
    
        num_cfirms : int
            number of C-firms visited on the consumption market
    
        deposit_interest : float
            bank deposit interest rate
    
        employed : bool
            employment status
    
    Data
    ----
        wage : NDArray
            wage rate
            
        income : NDArray
            income (wage + deposit interest) 
    
        deposits : NDArray
            deposits held at the bank
        
        expenditure : NDArray
            expenditure on consumption goods
        
        desired_expenditure : NDArray
            desired expenditure on consumption goods
    
    Methods
    -------
        determine_income(self, t: int) -> None
        
        determine_desired_consumption(self, t: int) -> None
        
        determine_consumption(self, cfirms: list['ConsumptionFirm'], probabilities: list[float], t: int) -> None
        
        determine_applications(self, firms: list['ConsumptionFirm | CapitalFirm'], probabilities: list[float], t: int) -> None
        
        determine_deposits(self, t: int) -> None
    """
    
    def __init__(self, id: int, init_wage: float, init_deposits: float, params: dict) -> None:
        """
        Household class initialisation.
        
        Parameters
        ----------
            id : int
                unique id
            
            init_wage : float
                initial wage rate
            
            params : dict
                model parameters
        """
        # Parameters 
        self.id:                    int   = id
        self.steps:                 int   = params['simulation']['steps']
        self.time:                  int   = (params['simulation']['years'] + params['simulation']['start'])*self.steps + 1
        self.dt:                    float = 1/self.steps
        self.mpc_income:            float = params['household']['mpc_income']
        self.mpc_deposits:          float = params['household']['mpc_deposits']
        self.num_firms:             int   = params['household']['num_firms']
        self.num_cfirms:            int   = params['household']['num_cfirms']
        self.deposit_interest:      float = params['bank']['deposit_interest'] * self.dt
        self.employed:              bool  = False
        # Data  
        self.wage:                  NDArray = np.zeros(shape=self.time)
        self.income:                NDArray = np.zeros(shape=self.time)
        self.deposits:              NDArray = np.zeros(shape=self.time)
        self.expenditure:           NDArray = np.zeros(shape=self.time)
        self.desired_expenditure:   NDArray = np.zeros(shape=self.time)
        # Initial values    
        self.wage[0]                = init_wage
        self.deposits[0]            = init_deposits
        self.income[0]              = init_wage + self.deposit_interest * self.deposits[0]
    
    def __repr__(self) -> str:
        """
        Returns printable unique household agent representaton.
        
        Returns
        -------
            representation : str
        """
        return f'Household: {self.id}'
    
    def __str__(self) -> str:
        """ 
        Returns printable agent type.
        
        Returns
        -------
            agent type : str
        """
        return 'Household'
    
    def determine_income(self, t: int) -> None:
        """
        Households income.
        
        Parameters
        ----------
            t : int
                time period
        """
        self.income[t] = self.wage[t] + self.deposit_interest * self.deposits[t-1]

    def determine_desired_consumption(self, t: int) -> None:
        """
        Desired household consumption.
        
        Parameters
        ----------
            t : int
        """
        # desired expenditure as income plus deposit (savings) expenditure marginal propensities
        self.desired_expenditure[t] = self.mpc_income * self.income[t] + self.mpc_deposits * self.deposits[t-1]

    def determine_consumption(self, cfirms: list['ConsumptionFirm'], probabilities: list[float], t: int) -> None:
        """
        Household visit C-firms with probability given by probabilities list and consumes consumption goods.
        
        Parameters
        ----------
            cfirms : list[ConsumptionFirm]
                list of all C-firms in the market
            
            probabilities : list[float]
                list of probabilities that household visits a given C-firm
            
            t : int 
                time period
        """
        # randomly visits cfirms (index)
        visited_cfirms_indices: list[np.int32] = list(np.random.choice(np.arange(len(cfirms)), size=self.num_cfirms, replace=False, p=probabilities))
        # sorts cfirms indices by price
        sorted_visited_cfirms_indices = sorted(visited_cfirms_indices, key=lambda i: cfirms[i].price[t], reverse=False)
        # consumption from cfirms 
        for i in sorted_visited_cfirms_indices:
            # cfirm at index i
            cfirm = cfirms[i]
            # amount of goods demanded 
            goods_demanded = self.desired_expenditure[t] / cfirm.price[t]
            # amount of goods purchased, constrained by cfirm inventories
            goods_purchased = min(cfirm.inventories[t], goods_demanded)
            # cost of purchased goods
            goods_cost = goods_purchased * cfirm.price[t]
            # update cfirm demand
            cfirm.demand[t] += goods_demanded
            # update cfirm inventories
            cfirm.inventories[t] -= goods_purchased
            # update cfirm quantity sold
            cfirm.quantity[t] += goods_purchased
            # update expenditure
            self.expenditure[t] += goods_cost
            # update desired expenditure
            self.desired_expenditure[t] -= goods_cost
            # household stops consuming if desired expenditure is exceeded
            if self.desired_expenditure[t] <= 1e-6:
                break
    
    def determine_applications(self, firms: list['ConsumptionFirm | CapitalFirm'], probabilities: list[float], t: int) -> None:
        """
        Unemployed household sends job applications to firms with open vacancies.
        
        Parameters
        ----------
            firms : list[ConsumptionFirm | CapitalFirm]
                list of all firms (C-firms + K-firms) in the market
            
            probabilities : list[float]
                list of probabilities that household sends an application to a given firm
        
            t : int
                time period
        """
        # only applies to jobs if unemployed
        if not self.employed:
            # randomly visits firms
            selected_firms_indices: list[np.int32] = list(np.random.choice(np.arange(len(firms)), size=self.num_firms, replace=False, p=probabilities))
            # sorts firms by wage
            sorted_selected_firms_indices = sorted(selected_firms_indices, key=lambda i: firms[i].wage[t], reverse=True)
            # applies to first firm with highest wage and open vacancies
            for i in sorted_selected_firms_indices:
                # firm at index i
                firm = firms[i]
                # check if firm has vacancies
                if firm.vacancies[t] > 0:
                    # append to firm applications
                    firm.applications.append(self)

    def determine_deposits(self, t: int) -> None:
        """
        Calculates household deposits in next period.
        
        Parameters
        ----------
            t : int
                time period
        """
        # updates deposits (savings) as earned income minus any expenditure
        self.deposits[t] = self.deposits[t-1] + self.income[t] - self.expenditure[t]
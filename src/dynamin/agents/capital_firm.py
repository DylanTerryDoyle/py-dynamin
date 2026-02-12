
### Import Parent AbstractFirm for Inheritance ###
from dynamin.agents.abstract_firm import AbstractFirm

### Capital Firm Class ###
class CapitalFirm(AbstractFirm):
    
    """
    CapitalFirm Class
    =================
    Capital firm (K-firm) class used to simulate K-firms in the model, inherits from Firm class. 
    
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
            
        excess_output : float
            desired percentage of excess output over demand 
    
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
        
        assets : numpy.ndarray
            value of assets
        
        liabilities : numpy.ndarray
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
        
        determine_bankruptcy(self, t: int) -> None
        
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
        CapitalFirm class initialisation, inherits from Firm class.
        
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
        super().__init__(id, initial_output, initial_wage, params)
        # Parameters
        self.excess_output:             float = params['kfirm']['excess_output']
        # Initial values
        self.market_share[0]            = 1 / params['simulation']['num_kfirms']
        self.desired_inventories[0]     = self.excess_output * initial_output
        self.wage_bill[0]               = self.wage[0] * self.desired_labour[0]
        self.profits[0]                 = initial_output - self.wage_bill[0]
        self.profit_share[0]            = self.profits[0] / initial_output
        self.deposits[0]                = self.profits[0]
        self.equity[0]                  = self.deposits[0]

    def __repr__(self) -> str:
        """
        Returns printable unique K-firm agent representaton.
        
        Returns
        -------
            representation : str
        """
        return f'Kfirm: {self.id}'
    
    def __str__(self) -> str:
        """ 
        Returns printable agent type.
        
        Returns
        -------
            agent type : str
        """
        return 'CapitalFirm'

    def determine_output(self, t: int) -> None:
        """
        Calculate K-firm output as a linear production function of labour productivity and the number of employees.
        
        Parameters
        ----------
            t : int
                time period
        """
        # calculate total labour 
        self.labour[t] = len(self.employees)
        # output as total labour times the productivity they make goods
        self.output[t] = self.productivity[t] * self.labour[t]

    def determine_inventories(self, t: int) -> None:
        """
        Calculate K-firm inventories.
        
        Parameters
        ----------
            t : int
                time period
        """
        # update total inventories from new output and previous inventories
        self.inventories[t] = self.output[t] + self.inventories[t-1]
        # desired level of inventories as a ration of current output
        self.desired_inventories[t] = self.output[t] * self.excess_output
    
    def determine_desired_output(self, t: int) -> None:
        """
        Calculate K-firm desired output.
        
        Parameters
        ----------
            t : int
                time period
        """
        # desired output next period
        self.desired_output[t] = max(self.expected_demand[t] * (1 + self.excess_output) - self.inventories[t], self.expected_productivity[t])
    
    def determine_desired_labour(self, t: int) -> None:
        """
        Calculate K-firm desired labour.
        
        Parameters
        ----------
            t : int
                time period
        """
        # desired amount of labour next period
        self.desired_labour[t] = int(self.desired_output[t] / self.expected_productivity[t])

    def determine_desired_loan(self, t: int) -> None:
        """
        Calculate K-firm desired loan.
        
        Parameters
        ----------
            t : int
                time period
        """
        # desired loan next period
        self.desired_loan[t] = max(self.wage_buffer * self.wage_bill[t] - self.profits[t] - self.deposits[t-1], 0)
        
    def determine_deposits(self, t: int) -> None:
        """
        Calculate K-firm deposits.
        
        Parameters
        ----------
            t : int
                time period
        """
        # update deposits by accounting identity 
        self.deposits[t] = self.deposits[t-1] + self.profits[t] + self.loans[t] - self.total_repayment[t]
        
    def determine_balance_sheet(self, t: int) -> None:
        """
        Calculate K-firm balance sheet, assets and liabilities.
        
        Parameters
        ----------
            t : int
                time period
        """
        # calculate assets
        self.assets[t] = self.deposits[t]
        # calculate liabilities
        self.liabilities[t] = self.debt[t]
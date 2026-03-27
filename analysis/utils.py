import re
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from scipy import stats
from pathlib import Path
from sqlalchemy import Engine

def format_scenario_name(name: str) -> str:
    if name.startswith("zero_growth"):
        number = re.search(r"\d+", name).group()
        return f"Zero-Growth {number}"
    elif name.startswith("growth"):
        number = re.search(r"\d+", name).group()
        return f"Growth {number}"
    else:
        # fallback
        return name

def load_scenarios(engine: Engine) -> pd.DataFrame:
    """
    Load scenarios table to dataframe.

    Parameters
    ----------
        engine : Engine
            database engine connector
    """
    query = """
        SELECT * FROM scenarios;
    """
    return pd.read_sql_query(query, engine)

def load_macro_data(engine: Engine, scenario_id: int, params: dict):
    """Load macro_data table from the SQL database into a pandas DataFrame and calculate variables."""
    start = params["simulation"]["start"]
    steps = params["simulation"]["steps"]
    # load table into pandas DataFrame
    data = pd.read_sql_query(
        """
            SELECT T3.scenario_id, T2.simulation_index, T1.* 
            
            FROM macro_data AS T1
            
            -- join simulations and scenarios to get scenario id
            JOIN simulations AS T2 ON T2.simulation_id = T1.simulation_id
            JOIN scenarios AS T3 ON T3.scenario_id = T2.scenario_id
            
            -- filter by scenario_id
            WHERE T3.scenario_id = %(scenario_id)s
            
            -- order by scenario, simulation, time
            ORDER BY T3.scenario_index, T2.simulation_index, T1.time ASC
            ;
        """,
        engine,
        params={
            "scenario_id": scenario_id
        }
    )
    ### define new time series ###
    # year
    data["year"] = (data["time"] // steps).astype(int)
    # turn real_gdp from quarterly to annual
    data["real_gdp"] = data.groupby("simulation_id")["real_gdp"].transform(lambda x: x.rolling(window=steps).sum())
    # turn nominal_gdp from quarterly to annual
    data["nominal_gdp"] = data.groupby("simulation_id")["nominal_gdp"].transform(lambda x: x.rolling(window=steps).sum())
    # real gdp growth 
    data["rgdp_growth"] = data.groupby("simulation_id")["real_gdp"].transform(lambda x: np.log(x) - np.log(x.shift(steps)))
    # standardised real gdp growth 
    data["std_rgdp_growth"] = data.groupby("simulation_id")["rgdp_growth"].transform(lambda x: (x - x.mean()) / x.std())
    # turn real_consumption from quarterly to annual
    data["real_consumption"] = data.groupby("simulation_id")["real_consumption"].transform(lambda x: x.rolling(window=steps).sum())
    # consumption growth 
    data["consumption_growth"] = data.groupby("simulation_id")["real_consumption"].transform(lambda x: np.log(x) - np.log(x.shift(steps)))
    # turn real_investment from quarterly to annual
    data["real_investment"] = data.groupby("simulation_id")["real_investment"].transform(lambda x: x.rolling(window=steps).sum())
    # investment growth 
    data["investment_growth"] = data.groupby("simulation_id")["real_investment"].transform(lambda x: np.log(x) - np.log(x.shift(steps)))
    # price index (cfirms)
    data["price_index"] = data["nominal_consumption"] / data["real_consumption"]
    # inflation
    data["inflation"] = data.groupby("simulation_id")["price_index"].transform(lambda x: np.log(x) - np.log(x.shift(steps)))
    # turn wages from quarterly to annual 
    data["wages"] = data.groupby("simulation_id")["wages"].transform(lambda x: x.rolling(window=steps).sum())
    # wage inflation
    data["wage_inflation"] = data.groupby("simulation_id")["wages"].transform(lambda x: np.log(x) - np.log(x.shift(steps)))
    # ESL / nGDP
    data["esl_gdp"] = data["esl"] / data["nominal_gdp"]
    # debt ratio
    data["debt_ratio"] = data["debt"] / data["nominal_gdp"]
    # wage share
    data["wage_share"] = data["wages"] / data["nominal_gdp"]
    # turn profits from quarterly to annual 
    data["profits"] = data.groupby("simulation_id")["profits"].transform(lambda x: x.rolling(window=steps).sum())
    # profit share
    data["profit_share"] = data["profits"] / data["nominal_gdp"]
    # productivity level
    data["productivity_level"] = data["real_gdp"] / data["employment"]
    # normalised productivity to start date
    data["productivity"] = data.groupby("simulation_id")["productivity_level"].transform(lambda x: x / x.iloc[start])
    # productivity growth
    data["productivity_growth"] = data.groupby("simulation_id")["productivity"].transform(lambda x: np.log(x) - np.log(x.shift(steps)))
    # credit: credit (change in debt) as a percentage of GDP
    data["credit_rate"] = data.groupby("simulation_id")["debt"].transform(lambda x: x - x.shift(steps)) / data["nominal_gdp"]
    # debt growth (log difference)
    data["debt_growth"] = data.groupby("simulation_id")["debt"].transform(lambda x: np.log(x) - np.log(x.shift(steps)))
    # change unemployment
    data["change_unemployment"] = data.groupby("simulation_id")["unemployment_rate"].transform(lambda x: x - x.shift(steps))
    
    ### recession indicator ###
    # 1. negative GDP growth indicator
    data["neg_rgdp"] = (data["rgdp_growth"] < 0).astype(int)
    # 2. identify consecutive spells (per simulation)
    data["neg_spell"] = (data.groupby("simulation_id")["neg_rgdp"].transform(lambda x: (x != x.shift()).cumsum()))
    # 3. spell lengths
    spell_lengths = (data[data["neg_rgdp"] == 1].groupby(["simulation_id", "neg_spell"]).size().reset_index(name="spell_length"))
    # 4. keep only spells with length >= 3
    valid_spells = spell_lengths[spell_lengths["spell_length"] >= 2]
    # merge back
    data = data.merge(valid_spells[["simulation_id", "neg_spell"]],on=["simulation_id", "neg_spell"],how="left",indicator=True)
    # recession indicator
    data["recession"] = (data["_merge"] == "both").astype(int)
    # cleanup
    data.drop(columns=["neg_rgdp", "neg_spell", "_merge"], inplace=True)

    ### crisis frequency (annual) ###
    # 1. Crisis flag per period
    data["crises"] = (data["rgdp_growth"] < -0.03).astype(int)
    # 2. Crisis occurrence per year per simulation
    yearly_crisis = (data.groupby(["simulation_id", "year"])["crises"].max().reset_index(name="year_crisis"))
    # 3. Crisis probability per simulation
    crisis_prob_sim = (yearly_crisis.groupby("simulation_id")["year_crisis"].mean().reset_index(name="crisis_probability"))
    # merge back
    data = data.merge(crisis_prob_sim, on="simulation_id", how="left")
    
    ### crisis severity (cumulative shortfall) ###
    # 1. Identify crisis spells
    data["crisis_spell"] = (data.groupby("simulation_id")["crises"].transform(lambda x: (x != x.shift()).cumsum()))
    # 2. Compute shortfall below -3%
    data["growth_shortfall"] = np.where(data["crises"] == 1,-0.03 - data["rgdp_growth"],0)
    # 3. Sum shortfall within each crisis spell
    crisis_severity = (data[data["crises"] == 1].groupby(["simulation_id", "crisis_spell"])["growth_shortfall"].sum().reset_index(name="crisis_severity"))
    # 4. Average severity per simulation
    severity_sim = (crisis_severity.groupby("simulation_id")["crisis_severity"].mean().reset_index())
    # merge back
    data = data.merge(severity_sim, on="simulation_id", how="left")

    ### tail risk (VaR & ES of std real growth growth rates) ###
    # tail
    alpha = 0.05
    # 1. compute value at risk (VaR 5% quantile) per simulation
    var_df = (data.groupby("simulation_id")["std_rgdp_growth"].quantile(alpha).reset_index(name="value_at_risk"))
    # merge back for ES calculation
    data = data.merge(var_df, on="simulation_id", how="left")

    # 2. compute expected shortfall (ES => mean below VaR)
    es_df = (data[data["std_rgdp_growth"] <= data["value_at_risk"]].groupby("simulation_id")["std_rgdp_growth"].mean().reset_index(name="expected_shortfall"))
    # merge back
    data = data.merge(es_df, on="simulation_id", how="left")

    # make VaR & ES positive 
    data["value_at_risk"] = -data["value_at_risk"]
    data["expected_shortfall"] = -data["expected_shortfall"]

    # drop transient data
    data = data.loc[data["time"] > start * steps]
    # reset index
    data.reset_index(inplace=True, drop=True)
    return data

def load_micro_data(engine: Engine, scenario_id: int, params: dict):
    """Load macro_data table from the SQL database into a pandas DataFrame and calculate variables."""
    start = params["simulation"]["start"]
    steps = params["simulation"]["steps"]
    # load table into pandas DataFrame
    data = pd.read_sql_query(
        """
            SELECT T3.scenario_id, T2.simulation_index, T1.* 
            
            FROM macro_data AS T1
            
            -- join simulations and scenarios to get scenario id
            JOIN simulations AS T2 ON T2.simulation_id = T1.simulation_id
            JOIN scenarios AS T3 ON T3.scenario_id = T2.scenario_id
            
            -- filter by scenario_id
            WHERE T3.scenario_id = %(scenario_id)s
            
            -- order by scenario, simulation, time
            ORDER BY T3.scenario_index, T2.simulation_index, T1.time ASC
            ;
        """,
        engine,
        params={
            "scenario_id": scenario_id
        }
    ) 
    ### define new time series ###
    # year
    data["year"] = (data["time"] // steps).astype(int)
    # turn real_gdp from quarterly to annual
    data["real_gdp"] = data.groupby("simulation_id")["real_gdp"].transform(lambda x: x.rolling(window=steps).sum())
    # turn nominal_gdp from quarterly to annual
    data["nominal_gdp"] = data.groupby("simulation_id")["nominal_gdp"].transform(lambda x: x.rolling(window=steps).sum())
    # ESL / nGDP
    data["esl_gdp"] = data["esl"] / data["nominal_gdp"]
    # real gdp growth 
    data["rgdp_growth"] = (data.groupby("simulation_id")["real_gdp"].transform(lambda x: np.log(x) - np.log(x.shift(steps))))
    # crisis flag per period
    data["crises"] = (data["rgdp_growth"] < -0.03).astype(int)
    ### bankruptcy probability conditioned on crisis ###
    # 1. quarterly default probability
    data["cfirm_probability_default"] = data["cfirm_defaults"] / params["simulation"]["num_cfirms"]
    data["kfirm_probability_default"] = data["kfirm_defaults"] / params["simulation"]["num_kfirms"]
    data["bank_probability_default"]  = data["bank_defaults"]  / params["simulation"]["num_banks"]
    # 2. average per year default probability, per simulation, split by crisis 
    yearly_probs = (data.groupby(["simulation_id","year","crises"])[["cfirm_probability_default","kfirm_probability_default","bank_probability_default"]].mean().reset_index())
    # 3. average across years within each simulation, conditional on there being a crisis
    cond_probs = (yearly_probs.groupby(["simulation_id","crises"])[["cfirm_probability_default","kfirm_probability_default","bank_probability_default"]].mean().reset_index())
    # 4. pivot to get separate columns for crisis=1 (true) and crisis=0 (false)
    cond_probs = cond_probs.pivot(index=["simulation_id"], columns="crises")
    cond_probs.columns = [f"{var}_crises{c}" for var,c in cond_probs.columns]
    cond_probs = cond_probs.reset_index()
    # merge back
    data = data.merge(cond_probs, on=["simulation_id"], how="left")
    # crisis = 1 values only in crisis periods
    data["cfirm_probability_default_crises1"] = np.where(data["crises"] == 1, data["cfirm_probability_default_crises1"], np.nan)
    data["kfirm_probability_default_crises1"] = np.where(data["crises"] == 1, data["kfirm_probability_default_crises1"], np.nan)
    data["bank_probability_default_crises1"] = np.where(data["crises"] == 1, data["bank_probability_default_crises1"], np.nan)
    # crisis = 0 values only in non-crisis periods
    data["cfirm_probability_default_crises0"] = np.where(data["crises"] == 0, data["cfirm_probability_default_crises0"], np.nan)
    data["kfirm_probability_default_crises0"] = np.where(data["crises"] == 0, data["kfirm_probability_default_crises0"], np.nan)
    data["bank_probability_default_crises0"] = np.where(data["crises"] == 0, data["bank_probability_default_crises0"], np.nan)
    # drop transient data
    data = data.loc[data["time"] > start * steps]
    # reset index
    data.reset_index(inplace=True, drop=True)

    return data

def box_plot_scenarios(
    plot_data: dict[str,pd.DataFrame],
    variables: dict[str,str],
    scenarios_short_names: list[str],
    figsize: tuple[float,float] = (16,10),
    ncols: int = 2,
    fontsize: int = 18,
    colours: list[str] | None = None,
    ylabel_dict: dict[str,str] | None = None,
    yticks_dict: dict[str,list[float]] | None = None,
    ylim_dict: dict[str,tuple[float,float]] | None = None,
    whis: tuple[float,float]=(5,95),
    sub_title_depth: float = 0.15,
    wspace: float = 0.2,
    hspace: float = 0.3,
    figure_path: Path | str | None = None,
    figure_name: str | None = None,
    dp: int = 3,
    dpi: int = 500
):
    """
    Create one figure with multiple subplots, where each subplot is a box plot
    for all scenarios for a single variable.
    """
    
    n_vars = len(variables)
    
    fig = plt.figure(figsize=figsize)
    outer = gs.GridSpec(n_vars // ncols, ncols, figure=fig, wspace=wspace, hspace=hspace)
    
    # loop over variables
    for i, (var_name, var_key) in enumerate(variables.items()):
        row = i // ncols
        col = i % ncols
        
        ax = fig.add_subplot(outer[row, col])
        
        # prepare data
        plot_vals = [plot_data[scenario][var_key].dropna().to_numpy() for scenario in scenarios_short_names]
        
        bplot = ax.boxplot(
            plot_vals,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", linewidth=2),
            whis=whis
        )
        
        # set box colours
        if colours:
            for patch, color in zip(bplot["boxes"], colours):
                patch.set_facecolor(color)
                patch.set_alpha(0.5)
        
        # x-axis
        ax.set_xticks(range(1, len(scenarios_short_names)+1))
        ax.set_xticklabels(scenarios_short_names, fontsize=fontsize, ha="right")

        # y-axis 
        if ylim_dict and var_key in ylim_dict:
            ax.set_ylim(ylim_dict[var_key])
        if yticks_dict and var_key in yticks_dict:
            ax.set_yticks(yticks_dict[var_key])
        if ylabel_dict and var_key in ylabel_dict:
            ax.set_ylabel(ylabel_dict[var_key], fontsize=fontsize)
        
        ax.tick_params(axis='y', labelsize=fontsize)
        
        # title below the subplot
        ax.text(0.5, -sub_title_depth, f"({chr(97+i)}) {var_name}", ha='center', va='top', fontsize=fontsize+5, transform=ax.transAxes)
    
    if figure_path and figure_name:
        figure_path = Path(figure_path)
        figure_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(figure_path / figure_name, dpi=dpi, bbox_inches="tight")
    elif figure_name:
        plt.savefig(figure_name, dpi=dpi, bbox_inches="tight")

def sub_boxplot(
    ax: plt.Axes,
    plot_data: dict[str,pd.DataFrame], 
    variable: str,
    fontsize: int | None=None,
    xlabels: list[str] | None=None, 
    xticks: list[int] | None=None, 
    yticks: list[float] | None=None,
    ylim: tuple[float, float] | None=None,
    colours: list[str] | None=None,
    figure_path: Path | str | None=None,
    dp: int = 3,
    whis: tuple[float,float]=(5,95)
    ):
    
    ### create plot data ###
    # copy plot data
    plot_data = plot_data.copy()
    # get all values across scenarios for variable
    for scenario in plot_data.keys():
        vals = plot_data[scenario][variable].dropna()
        plot_data[scenario] = vals.to_numpy().ravel()
    
    ### box plot ###
    bplot = ax.boxplot(plot_data.values(), patch_artist=True, showfliers=False, medianprops=dict(color="black", linewidth=2), whis=whis)
    
    ### set box colour ###
    if colours:
        for patch, color in zip(bplot["boxes"], colours):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
            
    ### figure settings ###
    # y axis ticks
    ax.set_yticks(yticks)
    # y axis limits
    ax.set_ylim(ylim)
    # x axis ticks
    ax.set_xticks(xticks)
    # tick size
    ax.tick_params(axis='both', labelsize=fontsize)
    
def normality_tests(data: pd.Series, significance: float=0.01):
    # ignore scipy warnings 
    warnings.filterwarnings("ignore")

    # Kolmogorov-Smirnov Test for Nomaliy
    ks_result = stats.kstest(data, "norm")
    # print test outcome
    print("  - Kolmogorov-Smirnov Test")
    if ks_result.pvalue < significance:
        print(f"    - Reject null hypothesis => not normally distributed ({significance*100}% significance level)")
    else:
        print(f"    - Accept null hypothesis => normally distributed ({significance*100}% significance level)")
    # print results
    print("    - Resutls:")
    print(f"      - test stat = {round(ks_result.statistic, 3)}")
    print(f"      - p-value = {round(ks_result.pvalue, 3)}")

    # Shapiro-Wilk Test for Nomaliy
    sw_result = stats.shapiro(data)
    # print test outcome
    print("  - Shapiro-Wilk Test")
    if sw_result.pvalue < significance:
        print(f"    - Reject null hypothesis => not normally distributed ({significance*100}% significance level)")
    else:
        print(f"    - Accept null hypothesis => normally distributed ({significance*100}% significance level)")
    # print results
    print("    - Results:")
    print(f"      - test stat = {round(sw_result.statistic, 3)}")
    print(f"      - p-value = {round(sw_result.pvalue, 3)}")

    # Anderson-Darling Test for Nomaliy
    ad_result = stats.anderson(data, "norm")
    # AD p-value
    if ad_result.statistic >= .6:
        ad_pvalue = np.exp(1.2937 - 5.709*ad_result.statistic - .0186*(ad_result.statistic**2))
    elif ad_result.statistic >=.34:
        ad_pvalue = np.exp(.9177 - 4.279*ad_result.statistic - 1.38*(ad_result.statistic**2))
    elif ad_result.statistic >.2:
        ad_pvalue = 1 - np.exp(-8.318 + 42.796*ad_result.statistic - 59.938*(ad_result.statistic**2))
    else:
        ad_pvalue = 1 - np.exp(-13.436 + 101.14*ad_result.statistic - 223.73*(ad_result.statistic**2))
    # print test outcome
    print("  - Anderson-Darling Test")
    # print results
    if ad_pvalue < significance:
        print(f"    - Reject null hypothesis => not normally distributed ({significance*100}% significance level)")
    else:
        print(f"    - Accept null hypothesis => normally distributed ({significance*100}% significance level)")
    print("    - Results:")
    print(f"      - test stat = {round(ad_result.statistic, 3)}")
    print(f"      - p-value = {round(ad_pvalue, 3)}")

def subplot_autocorrelation(
    ax: plt.Axes,
    simulated: np.typing.ArrayLike, 
    empirical: np.typing.ArrayLike, 
    fontsize: int, 
    lags: int=20, 
    lamda: int=1600
):
    """
    Plots the autocorrelation function (ACF) of the simulated time-series with a 95% confidence interval (CI), calculated over all simulations,
    and also plots the empirical time-series ACF for a given feature vector of the two time-series for a given number of lags.
    
    Parameters
    ----------
        simulated : pandas.DataFrame
            model simulated time-series with s simulations
        
        empirical : pandas.DataFrame
            single empirical time-series
        
        feature : str
            name of column in both simulated and empirical DataFrame
        
        figsize : tuple (int, int) 
            size of figure (x-axis, y-axis)
            
        fontsize : int
            fontsize of legend and ticks
        
        lags: int
            number of autocorrelation lags
        
        lamda : int
            Hodrick-Prescott filter lambda parameter (quarterly => lambda=1600)
            
        savefig : str
            path and figure name 
    """
    # array to store autocorrelation for each simulation
    # calculate autocorrelation for each simulation
    sim_autocorr = np.zeros(shape=(simulated.shape[0], lags + 1))
    for s in range(simulated.shape[0]):
        # decompose trend and cycle component using Hodrick-Prescott filter
        sim_cycle, sim_trend = sm.tsa.filters.hpfilter(simulated[s,:], lamda)
        # autocorrelation 
        sim_autocorr[s,:] = sm.tsa.acf(sim_cycle, nlags=lags)
    # median autocorrelation
    sim_autocorr_median = np.quantile(sim_autocorr, 0.5, axis=0)
    # upper 5% quantile
    sim_autocorr_upper = np.quantile(sim_autocorr, 0.95, axis=0)
    # lower 5% quantile
    sim_autocorr_lower = np.quantile(sim_autocorr, 0.05, axis=0)
    # decompose trend and cycle component of empirical time-series using Hodrick-Prescott filter
    emp_cycle, emp_trend = sm.tsa.filters.hpfilter(empirical, lamda)
    # empirical autocorrelation
    emp_autocorr = sm.tsa.acf(emp_cycle, nlags=lags)
    # x values (lags)
    x = np.arange(0, lags+1)
    # plot empirical autocorrelation 
    ax.plot(emp_autocorr, color="k", linestyle="--", linewidth=1, label="Empirical")
    # plot simulated autocorrelation median
    ax.plot(sim_autocorr_median, color="k", linewidth=1, label="Simulated")
    # plot confidence interval
    ax.fill_between(x, sim_autocorr_median, sim_autocorr_upper, color="grey", alpha=0.2, label="90% IPR")
    ax.fill_between(x, sim_autocorr_median, sim_autocorr_lower, color="grey", alpha=0.2)
    # plot 0 line
    ax.axhline(0, color="k")
    # figure ticks
    ax.set_yticks([1.00,0.75,0.50,0.25,0.00,-0.25,-0.50,-0.75,-1.00], ["1.00","0.75","0.50","0.25","0.00","–0.25","–0.50","–0.75","–1.00"], fontsize=fontsize)
    ax.set_xticks([0,5,10,15,20], [0,5,10,15,20], fontsize=fontsize)
    ax.set_xlabel("Lags", fontsize=fontsize + 2)

def cross_correlation(
    xsimulated: np.typing.ArrayLike,
    ysimulated: np.typing.ArrayLike,
    xempirical: np.typing.ArrayLike,
    yempirical: np.typing.ArrayLike,
    lags: int=10, 
    lamda: int=1600
):
    """
    Calculate the cross correlation (xcorr) of the simulated time-series for feature xfeature and yfeature with a 90% confidence interval (CI), calculated over all simulations,
    each for a given number of lags.
    
    Parameters
    ----------
        simulated : pandas.DataFrame
            model simulated time-series with s simulations
        
        empirical : pandas.DataFrame
            single empirical time-series
        
        xfeature : str
            name of the x feature column in both simulated and empirical DataFrames
        
        yfeature : str
            name of the y feature column in both simulated and empirical DataFrames
        
        lags: int
            number of correlation lags, in range (-lags, lags)
        
        lamda : int
            Hodrick-Prescott filter lambda parameter (quarterly => lambda=1600)
    """
    plt.figure()
    assert xsimulated.shape == ysimulated.shape
    # array to store autocorrelation for each simulation
    sim_xcorr = np.zeros(shape=(xsimulated.shape[0], lags*2 + 1))
    # calculate autocorrelation for each simulation
    for s in range(xsimulated.shape[0]):
        # decompose trend and cycle component using Hodrick-Prescott filter
        sim_xcycle, sim_xtrend = sm.tsa.filters.hpfilter(xsimulated[s,:], lamda)
        sim_ycycle, sim_ytrend = sm.tsa.filters.hpfilter(ysimulated[s,:], lamda)
        # autocorrelation
        sim_xcorr[s,:] = plt.xcorr(sim_xcycle, sim_ycycle, maxlags=lags)[1]
    # median autocorrelation
    sim_xcorr_median = np.quantile(sim_xcorr, 0.5, axis=0)
    # upper 5% quantile
    sim_xcorr_upper = np.quantile(sim_xcorr, 0.95, axis=0)
    # lower 5% quantile
    sim_xcorr_lower = np.quantile(sim_xcorr, 0.05, axis=0)
    # decompose trend and cycle component of empirical time-series using Hodrick-Prescott filter
    emp_xcycle, emp_xtrend = sm.tsa.filters.hpfilter(xempirical, lamda)
    emp_ycycle, emp_ytrend = sm.tsa.filters.hpfilter(yempirical, lamda)
    # empirical autocorrelation
    emp_xcorr = plt.xcorr(emp_xcycle, emp_ycycle, maxlags=lags)[1]
    plt.close()
    return sim_xcorr_median, sim_xcorr_upper, sim_xcorr_lower, emp_xcorr

def subplot_cross_correlation(
    ax: plt.Axes,
    xsimulated: np.typing.ArrayLike,
    ysimulated: np.typing.ArrayLike,
    xempirical: np.typing.ArrayLike,
    yempirical: np.typing.ArrayLike,
    fontsize: int, 
    lags: int=10, 
    lamda: int=1600
):
    """
    Plots the cross correlation (xcorr) of the simulated time-series for feature xfeature and yfeature with a 95% confidence interval (CI), calculated over all simulations,
    and also plots the empirical time-series xcorr for a given xfeature and yfeature vector, each for a given number of lags.
    
    Parameters
    ----------
        simulated : pandas.DataFrame
            model simulated time-series with s simulations
        
        empirical : pandas.DataFrame
            single empirical time-series
        
        xfeature : str
            name of the x feature column in both simulated and empirical DataFrames
        
        yfeature : str
            name of the y feature column in both simulated and empirical DataFrames
            
        fontsize : int
            fontsize of legend and ticks
        
        lags: int
            number of correlation lags, in range (-lags, lags)
        
        lamda : int
            Hodrick-Prescott filter lambda parameter (quarterly => lambda=1600)
    """
    # cross correlations
    sim_xcorr_median, sim_xcorr_upper, sim_xcorr_lower, emp_xcorr = cross_correlation(xsimulated, ysimulated, xempirical, yempirical, lags, lamda)
    # x values (lags)
    x = np.arange(0, lags*2 + 1)
    # plot empirical xcorr
    ax.plot(emp_xcorr, color="k", linestyle="--", linewidth=1, label="Empirical")
    # plot median simulated xcorr
    ax.plot(sim_xcorr_median, color="k", linewidth=1, label="Simulated")
    # plot confidence interval
    ax.fill_between(x, sim_xcorr_median, sim_xcorr_upper, color="grey", alpha=0.2, label="90% IPR")
    ax.fill_between(x, sim_xcorr_median, sim_xcorr_lower, color="grey", alpha=0.2)
    # plot 0 line
    ax.axhline(0, color="k")
    # figure ticks
    ax.set_yticks([1.00,0.75,0.50,0.25,0.00,-0.25,-0.50,-0.75,-1.00], ["1.00","0.75","0.50","0.25","0.00","–0.25","–0.50","–0.75","–1.00"], fontsize=fontsize)
    ax.set_xticks([0,int(0.5*lags),int(lags),int(1.5*lags),int(2*lags)], [f"–{lags}",f"–{int(0.5*lags)}",0,int(0.5*lags),lags], fontsize=fontsize)
    ax.set_xlabel("Lags", fontsize=fontsize + 2)

def fit_powerlaw(data: np.ndarray) -> tuple[float, float]:
    """
    Estimates power-law exponent (alpha) and xmin by minimising the
    KS statistic between the empirical tail and the fitted Pareto CDF,
    following Clauset, Shalizi & Newman (2009).

    Returns
    -------
    alpha : float   Power-law exponent
    xmin  : float   Lower cutoff
    """
    data = np.sort(data)
    n = len(data)

    # Search over candidate xmin values (up to 90th percentile)
    max_idx = int(0.99 * n)
    candidates = data[1:max_idx]  # skip the very first point

    best_ks   = np.inf
    best_xmin = candidates[0]
    best_alpha = 2.0

    for xmin_candidate in candidates:
        tail = data[data >= xmin_candidate]
        if len(tail) < 20:          # need enough points for a stable fit
            break

        # MLE for Pareto alpha given xmin
        # alpha_hat = 1 + n_tail / sum(log(x / xmin))
        n_tail = len(tail)
        alpha_hat = 1 + n_tail / np.sum(np.log(tail / xmin_candidate))

        # KS statistic between empirical and theoretical CDF
        empirical_cdf = np.arange(1, n_tail + 1) / n_tail
        theoretical_cdf = 1 - (xmin_candidate / tail) ** (alpha_hat - 1)
        ks = np.max(np.abs(empirical_cdf - theoretical_cdf))

        if ks < best_ks:
            best_ks    = ks
            best_xmin  = xmin_candidate
            best_alpha = alpha_hat

    return best_alpha, best_xmin

def subplot_ccdf(
    ax: plt.Axes,
    data: np.typing.ArrayLike,
    fontsize: int,
    ylim: tuple[float, float] | None=None,
    dp: int = 0,
) -> None:
    """
    Plots the complementary cumulative distribution function (CCDF) for a given
    continuous dataset, estimates a power-law tail with bounded xmin search,
    and compares against a lognormal distribution.
    """

    ### clean data ###
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    data = data[data > 0.01]

    if data.size < 100:
        raise ValueError("Too few positive samples for reliable power-law fitting.")

    ### power-law fit ###
    alpha, xmin = fit_powerlaw(data)

    # alpha = results.alpha
    # xmin = results.xmin

    ### lognormal fit ###
    ln_shape, ln_loc, ln_scale = stats.lognorm.fit(data, floc=0)

    ### empirical CCDF ###
    x = np.sort(data)
    cdf = np.arange(1, len(x) + 1) / len(x)
    ccdf = 1.0 - cdf

    ### plot fits ###

    # plot empirical CCDF
    ax.scatter(x, ccdf, color="lightgrey", edgecolors="k", alpha=0.7, s=30, label="Empirical CCDF",)
    # power-law CCDF (rescaled at xmin)
    idx = np.searchsorted(x, xmin)
    rescale = ccdf[idx]
    powerlaw_ccdf = np.where(x >= xmin, (xmin / x) ** (alpha - 1) * rescale, np.nan)
    # plot power-law fit
    ax.plot(x, powerlaw_ccdf, color="tab:cyan", linewidth=3, label=rf"Power law ($\alpha$ = {alpha:.2f})",)
    # xmin line
    ax.axvline(xmin, color="k", linestyle=":", label=rf"$x_{{\min}}$ = {round(xmin, dp)}")
    # lognormal CCDF
    ln_ccdf = 1.0 - stats.lognorm.cdf(x, ln_shape, ln_loc, ln_scale)
    ax.plot( x, ln_ccdf, color="tab:red", linestyle="--", linewidth=2, label="Log-normal")

    # plot settings
    ax.loglog()
    ax.legend(loc="lower left", fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize)
    ax.set_ylim(ylim)

    ### results summary ###
    print(f" - Power-law exponent (alpha) = {alpha:.4f}")
    print(f" - Power-law xmin = {xmin:.4f}\n")

def minskyan_test(
    data: pd.DataFrame,
    min_obs: int = 20,
    window: int = 8,
    min_consecutive: int = 2,
):
    """
    Event study test for whether credit_rate and debt_ratio peak before recessions.
 
    For each simulation:
      - Identifies recession start dates (min_consecutive periods of negative rgdp_growth).
      - Extracts pre/post windows of length `window` around each recession start.
      - Records the lag of the peak credit_rate and debt_ratio in the pre-recession window.
      - Tests whether the peak occurs before the recession start (peak_lag < 0).
 
    Parameters
    ----------
    data            : DataFrame with columns simulation_id, rgdp_growth,
                      credit_rate, debt_ratio.
    min_obs         : Minimum observations required per simulation.
    window          : Number of periods before and after recession start to extract.
    min_consecutive : Consecutive negative growth periods to define a recession.
    """
 
    results_list            = []
    all_credit_rate_windows = []   # (n_episodes, 2*window+1) — for aggregate plot
    all_debt_ratio_windows  = []
 
    total_sims   = 0
    dropped_obs  = 0
    no_recession = 0
 
    # helper: find all recession start dates in a growth series
    def find_recession_starts(growth, min_consecutive):
        starts = []
        count  = 0
        for t, v in enumerate(growth):
            count = count + 1 if v < 0 else 0
            if count == min_consecutive:
                # start is the first period of the negative streak
                starts.append(t - min_consecutive + 1)
        return starts
 
    for sim_id, df in data.groupby("simulation_id"):
 
        total_sims += 1
        df = df[["rgdp_growth", "credit_rate", "debt_ratio"]].dropna().reset_index(drop=True)
 
        if len(df) < min_obs:
            dropped_obs += 1
            continue
 
        recession_starts = find_recession_starts(df["rgdp_growth"].values, min_consecutive)
 
        if not recession_starts:
            no_recession += 1
            continue
 
        for t0 in recession_starts:
 
            # require full window on both sides
            if t0 - window < 0 or t0 + window >= len(df):
                continue
 
            credit_window = df["credit_rate"].iloc[t0 - window: t0 + window + 1].values
            debt_window   = df["debt_ratio"].iloc[t0 - window: t0 + window + 1].values
 
            # lag of peak within window — negative means peak before recession start
            # window index `window` corresponds to lag 0 (recession start)
            credit_peak_idx = np.argmax(credit_window)
            debt_peak_idx   = np.argmax(debt_window)
 
            # convert index to lag relative to recession start
            credit_peak_lag = credit_peak_idx - window
            debt_peak_lag   = debt_peak_idx   - window
 
            # peak precedes recession start
            credit_peaks_before = credit_peak_lag < 0
            debt_peaks_before   = debt_peak_lag   < 0
 
            # minsky condition: both credit and debt peak before recession
            minsky_condition = credit_peaks_before and debt_peaks_before
 
            all_credit_rate_windows.append(credit_window)
            all_debt_ratio_windows.append(debt_window)
 
            results_list.append({
                "simulation_id":       sim_id,
                "recession_start":     t0,
                "credit_peak_lag":     credit_peak_lag,
                "debt_peak_lag":       debt_peak_lag,
                "credit_peaks_before": credit_peaks_before,
                "debt_peaks_before":   debt_peaks_before,
                "minsky":              minsky_condition,
            })
    
    if not results_list:
        raise ValueError("No recession episodes found.")
 
    results_df = pd.DataFrame(results_list)
 
    ### formal test: is mean peak lag significantly negative? ###
    from scipy.stats import ttest_1samp
    credit_tstat, credit_pvalue = ttest_1samp(results_df["credit_peak_lag"], popmean=0)
    debt_tstat,   debt_pvalue   = ttest_1samp(results_df["debt_peak_lag"],   popmean=0)
 
    ### summary results ###
    summary = pd.DataFrame([{
        "total_simulations":         total_sims,
        "total_episodes":            len(results_list),
 
        # mean peak lags (negative => peaks before recession)
        "mean_credit_peak_lag":      results_df["credit_peak_lag"].mean(),
        "mean_debt_peak_lag":        results_df["debt_peak_lag"].mean(),
 
        # share peaking before recession
        "share_credit_peaks_before": results_df["credit_peaks_before"].mean(),
        "share_debt_peaks_before":   results_df["debt_peaks_before"].mean(),
        "share_minsky":              results_df["minsky"].mean(),
 
        # formal t-test: mean peak lag < 0
        "credit_peak_tstat":         credit_tstat,
        "credit_peak_pvalue":        credit_pvalue,
        "debt_peak_tstat":           debt_tstat,
        "debt_peak_pvalue":          debt_pvalue,
    }])
 
    return results_df, summary, all_credit_rate_windows, all_debt_ratio_windows

def debt_deflation_test(
    data: pd.DataFrame,
    min_obs: int = 20,
    window: int = 8,
    min_consecutive: int = 2,
):
    """
    Event study test for whether deflation and recession occur at the same time,
    consistent with Fisher's (1933) debt-deflation theory.
 
    For each simulation:
      - Identifies recession start dates (min_consecutive periods of negative rgdp_growth).
      - Checks whether inflation is negative during the recession episode (deflation present).
      - Tests whether debt_ratio peaks before the recession start.
      - Fisher condition: deflation present during recession and debt peaks before.
 
    Parameters
    ----------
    data            : DataFrame with columns simulation_id, rgdp_growth,
                      debt_ratio, inflation.
    min_obs         : Minimum observations required per simulation.
    window          : Number of periods before and after recession start to extract
                      (used for debt_ratio peak detection and aggregate plot).
    min_consecutive : Consecutive negative growth periods to define a recession.
    """
 
    results_list           = []
    all_debt_ratio_windows = []   # (n_episodes, 2*window+1) — for aggregate plot
    all_inflation_windows  = []
 
    total_sims   = 0
    dropped_obs  = 0
    no_recession = 0
 
    # helper: find all recession start dates in a growth series
    def find_recession_starts(growth, min_consecutive):
        starts = []
        count  = 0
        for t, v in enumerate(growth):
            count = count + 1 if v < 0 else 0
            if count == min_consecutive:
                # start is the first period of the negative streak
                starts.append(t - min_consecutive + 1)
        return starts
 
    for sim_id, df in data.groupby("simulation_id"):
 
        total_sims += 1
        df = df[["rgdp_growth", "debt_ratio", "inflation"]].dropna().reset_index(drop=True)
 
        if len(df) < min_obs:
            dropped_obs += 1
            continue
 
        recession_starts = find_recession_starts(df["rgdp_growth"].values, min_consecutive)
 
        if not recession_starts:
            no_recession += 1
            continue
 
        for t0 in recession_starts:
 
            # require full window on both sides
            if t0 - window < 0 or t0 + window >= len(df):
                continue
 
            debt_window      = df["debt_ratio"].iloc[t0 - window: t0 + window + 1].values
            inflation_window = df["inflation"].iloc[t0 - window: t0 + window + 1].values
 
            # debt ratio: lag of peak — negative means peak before recession start
            debt_peak_idx   = np.argmax(debt_window)
            debt_peak_lag   = debt_peak_idx - window
            debt_peaks_before = debt_peak_lag < 0
 
            # deflation: inflation is negative during the recession episode
            recession_inflation = df["inflation"].iloc[t0: t0 + min_consecutive].values
            deflation_present   = bool((recession_inflation < 0).any())
 
            # fisher condition: deflation present during recession and debt peaked before
            fisher_condition = deflation_present and debt_peaks_before
 
            all_debt_ratio_windows.append(debt_window)
            all_inflation_windows.append(inflation_window)
 
            results_list.append({
                "simulation_id":      sim_id,
                "recession_start":    t0,
                "debt_peak_lag":      debt_peak_lag,
                "debt_peaks_before":  debt_peaks_before,
                "deflation_present":  deflation_present,
                "fisher":             fisher_condition,
            })
     
    if not results_list:
        raise ValueError("No recession episodes found.")
 
    results_df = pd.DataFrame(results_list)
 
    ### formal test: is mean debt peak lag significantly negative? ###
    from scipy.stats import ttest_1samp
    debt_tstat, debt_pvalue = ttest_1samp(results_df["debt_peak_lag"], popmean=0)
 
    ### summary results ###
    summary = pd.DataFrame([{
        "total_simulations":              total_sims,
        "total_episodes":                 len(results_list),
 
        # mean debt peak lag (negative => peaks before recession)
        "mean_debt_peak_lag":             results_df["debt_peak_lag"].mean(),
 
        # share peaking before / deflation present
        "share_debt_peaks_before":        results_df["debt_peaks_before"].mean(),
        "share_deflation_during_recession": results_df["deflation_present"].mean(),
        "share_fisher":                   results_df["fisher"].mean(),
    }])
 
    return results_df, summary, all_inflation_windows

# def debt_deflation_test(
#     data: pd.DataFrame,
#     min_obs: int = 20,
#     max_lags: int = 4,
#     horizon: int = 12,
#     alpha: float = 0.05,
# ):
#     warnings.filterwarnings("ignore", module="statsmodels")

#     # recession impulse response helper
#     def irf_recession(series, min_consecutive=2):
#         count = 0
#         for v in series:
#             count = count + 1 if v < 0 else 0
#             if count >= min_consecutive:
#                 return True
#         return False

#     # time to recession helper
#     def time_to_recession(series, min_consecutive=2):
#         count = 0
#         for t, v in enumerate(series):
#             count = count + 1 if v < 0 else 0
#             if count >= min_consecutive:
#                 return t - min_consecutive + 1
#         return np.nan

#     # storage for per-simulation IRF arrays (for aggregate plot)
#     all_inflation_irfs = []
#     all_growth_irfs    = []

#     results_list      = []
#     total_sims        = 0
#     dropped_obs       = 0
#     dropped_var_fit   = 0
#     dropped_whiteness = 0
#     valid_sims        = 0

#     for sim_id, df in data.groupby("simulation_id"):

#         total_sims += 1
#         df = df[["debt_ratio", "inflation", "rgdp_growth"]].dropna().reset_index(drop=True)

#         if len(df) < min_obs:
#             dropped_obs += 1
#             continue

#         ### fit VAR with AIC lag selection ###
#         try:
#             model      = VAR(df)
#             lag_select = model.select_order(max_lags)
#             p          = max(lag_select.aic, 1)
#             res        = model.fit(p)
#         except Exception:
#             dropped_var_fit += 1
#             continue

#         valid_sims += 1

#         ### orthogonalised IRF — shock scaled to one SD of the structural shock ###
#         # orth=True applies the Cholesky decomposition of the residual covariance
#         # matrix, giving a shock equal to a typical (one-SD) movement in debt_ratio
#         # rather than an arbitrary one-unit impulse
#         irf_obj = res.irf(horizon)

#         # response of inflation to orthogonalised debt shock
#         inflation_irf = irf_obj.orth_irfs[:, 1, 0]
#         # response of growth to orthogonalised debt shock
#         growth_irf    = irf_obj.orth_irfs[:, 2, 0]

#         # store IRFs for aggregate plot
#         all_inflation_irfs.append(inflation_irf)
#         all_growth_irfs.append(growth_irf)

#         # sign flags (point estimate)
#         deflation_sig = bool(inflation_irf.min() < 0)
#         recession_sig = irf_recession(growth_irf, min_consecutive=2)

#         # magnitudes and timing
#         peak_deflation = float(inflation_irf.min())
#         trough_growth  = float(growth_irf.min())
#         t_deflation    = int(np.argmin(inflation_irf))
#         t_recession    = time_to_recession(growth_irf, min_consecutive=2)

#         ### signed Granger causality ###
#         try:
#             gc_debt_infl   = res.test_causality("inflation",   ["debt_ratio"], kind="f")
#             gc_debt_growth = res.test_causality("rgdp_growth", ["debt_ratio"], kind="f")
#             gc_infl_growth = res.test_causality("rgdp_growth", ["inflation"],  kind="f")

#             # granger significance and correct sign direction
#             debt_causes_infl   = (gc_debt_infl.pvalue   < alpha) and (inflation_irf.min() < 0)
#             debt_causes_growth = (gc_debt_growth.pvalue < alpha) and (growth_irf.min()    < 0)
#             infl_causes_growth = (gc_infl_growth.pvalue < alpha) and (growth_irf.min()    < 0)

#         except Exception:
#             debt_causes_infl = debt_causes_growth = infl_causes_growth = False

#         # significant Granger chain: debt -> deflation -> recession
#         fisher_granger = debt_causes_infl and debt_causes_growth and infl_causes_growth

#         # full Fisher mechanism: deflation + recession + Granger chain
#         fisher_mechanism = deflation_sig and recession_sig and fisher_granger

#         ### heterogeneity: initial debt level ###
#         initial_debt = df["debt_ratio"].iloc[0]

#         results_list.append({
#             "simulation_id":      sim_id,
#             "initial_debt_ratio": initial_debt,

#             # IRF sign flags (point estimate)
#             "deflation_irf": deflation_sig,
#             "recession_irf": not np.isnan(t_recession),

#             # full Fisher mechanism
#             "fisher_irf": deflation_sig and (not np.isnan(t_recession)) and fisher_granger,

#             # Granger chain
#             "granger_debt_infl":   debt_causes_infl,
#             "granger_debt_growth": debt_causes_growth,
#             "granger_infl_growth": infl_causes_growth,
#             "fisher_granger":      fisher_granger,

#             # magnitudes
#             "peak_deflation": peak_deflation,
#             "trough_growth":  trough_growth,

#             # timing (periods)
#             "time_to_deflation": t_deflation,
#             "time_to_recession": t_recession,
#         })

#     # diagnostics — printed before the validity check so they're always visible
#     print(f"  [diagnostics] total: {total_sims} | drop obs: {dropped_obs} | drop fit: {dropped_var_fit} | drop whiteness: {dropped_whiteness} | valid: {valid_sims}")

#     if not results_list:
#         raise ValueError("No valid VAR estimations after diagnostics.")

#     results_df = pd.DataFrame(results_list)

#     ### heterogeneity: split by median initial debt ###
#     median_debt = results_df["initial_debt_ratio"].median()
#     high_debt   = results_df[results_df["initial_debt_ratio"] >= median_debt]
#     low_debt    = results_df[results_df["initial_debt_ratio"] <  median_debt]

#     ### summary results ###
#     def nanmean(s):
#         return s.dropna().mean() if len(s.dropna()) else np.nan

#     summary = pd.DataFrame([{
#         "total_simulations": total_sims,
#         "valid_estimations": valid_sims,

#         # point-estimate share
#         "share_deflation_irf": results_df["deflation_irf"].mean(),
#         "share_recession_irf": results_df["recession_irf"].mean(),
#         "share_fisher_irf":    results_df["fisher_irf"].mean(),

#         # Granger chain
#         "share_granger_debt_infl":   results_df["granger_debt_infl"].mean(),
#         "share_granger_debt_growth": results_df["granger_debt_growth"].mean(),
#         "share_granger_infl_growth": results_df["granger_infl_growth"].mean(),
#         "share_fisher_granger":      results_df["fisher_granger"].mean(),

#         # magnitudes
#         "mean_peak_deflation": nanmean(results_df["peak_deflation"]),
#         "mean_trough_growth":  nanmean(results_df["trough_growth"]),

#         # timing (quarters)
#         "mean_time_to_deflation": nanmean(results_df["time_to_deflation"]),
#         "mean_time_to_recession": nanmean(results_df["time_to_recession"]),

#         # heterogeneity by debt level
#         "fisher_sig_high_debt": high_debt["fisher_irf"].mean() if len(high_debt) else np.nan,
#         "fisher_sig_low_debt":  low_debt["fisher_irf"].mean()  if len(low_debt)  else np.nan,
#     }])

#     return results_df, summary, all_inflation_irfs, all_growth_irfs

# def pre_recession_peak_test(
#     data: pd.DataFrame,
#     min_obs: int = 20,
#     window: int = 8,
#     min_consecutive: int = 2,
# ):
#     """
#     Event study test for whether credit_rate, debt_ratio, and inflation peak
#     before recessions.
 
#     For each simulation:
#       - Identifies recession start dates (min_consecutive periods of negative rgdp_growth).
#       - Extracts pre/post windows of length `window` around each recession start.
#       - Records the lag of the peak credit_rate and debt_ratio in the pre-recession window.
#       - Tests whether the peak occurs before the recession start (peak_lag < 0).
 
#     Parameters
#     ----------
#     data            : DataFrame with columns simulation_id, rgdp_growth,
#                       credit_rate, debt_ratio, inflation.
#     min_obs         : Minimum observations required per simulation.
#     window          : Number of periods before and after recession start to extract.
#     min_consecutive : Consecutive negative growth periods to define a recession.
#     """
 
#     results_list = []
#     all_credit_rate_windows = []   # (n_episodes, 2*window+1) — for aggregate plot
#     all_debt_ratio_windows  = []
#     all_inflation_windows   = []
 
#     total_sims    = 0
#     dropped_obs   = 0
#     no_recession  = 0
 
#     # helper: find all recession start dates in a growth series
#     def find_recession_starts(growth, min_consecutive):
#         starts = []
#         count  = 0
#         for t, v in enumerate(growth):
#             count = count + 1 if v < 0 else 0
#             if count == min_consecutive:
#                 # start is the first period of the negative streak
#                 starts.append(t - min_consecutive + 1)
#         return starts
 
#     for sim_id, df in data.groupby("simulation_id"):
 
#         total_sims += 1
#         df = df[["rgdp_growth", "credit_rate", "debt_ratio", "inflation"]].dropna().reset_index(drop=True)
 
#         if len(df) < min_obs:
#             dropped_obs += 1
#             continue
 
#         recession_starts = find_recession_starts(df["rgdp_growth"].values, min_consecutive)
 
#         if not recession_starts:
#             no_recession += 1
#             continue
 
#         for t0 in recession_starts:
 
#             # require full window on both sides
#             if t0 - window < 0 or t0 + window >= len(df):
#                 continue
 
#             credit_window    = df["credit_rate"].iloc[t0 - window: t0 + window + 1].values
#             debt_window      = df["debt_ratio"].iloc[t0 - window: t0 + window + 1].values
#             inflation_window = df["inflation"].iloc[t0 - window: t0 + window + 1].values
 
#             # lag of peak within window — negative means peak before recession start
#             # window index `window` corresponds to lag 0 (recession start)
#             credit_peak_idx    = np.argmax(credit_window)
#             debt_peak_idx      = np.argmax(debt_window)
#             deflation_peak_idx = np.argmin(inflation_window)
 
#             # convert index to lag relative to recession start
#             credit_peak_lag    = credit_peak_idx    - window
#             debt_peak_lag      = debt_peak_idx      - window
#             deflation_peak_lag = deflation_peak_idx - window
 
#             # peak precedes recession start
#             credit_peaks_before    = credit_peak_lag    < 0
#             debt_peaks_before      = debt_peak_lag      < 0
#             deflation_peaks_before = deflation_peak_lag < 0
 
#             # minsky condition: both credit and debt peak before recession
#             minsky_condition = credit_peaks_before and debt_peaks_before 

#             # fisher condition: deflation peaks before recession and debt ratio peaks before recession
#             fisher_condition = deflation_peaks_before and debt_peaks_before
 
#             all_credit_rate_windows.append(credit_window)
#             all_debt_ratio_windows.append(debt_window)
#             all_inflation_windows.append(inflation_window)
 
#             results_list.append({
#                 "simulation_id":        sim_id,
#                 "recession_start":      t0,
#                 "credit_peak_lag":      credit_peak_lag,
#                 "debt_peak_lag":        debt_peak_lag,
#                 "deflation_peak_lag":   deflation_peak_lag,
#                 "credit_peaks_before":  credit_peaks_before,
#                 "debt_peaks_before":    debt_peaks_before,
#                 "deflation_peaks_before": deflation_peaks_before,
#                 "minsky":               minsky_condition,
#                 "fisher":               fisher_condition,
#             })
 
#     print(f"  [diagnostics] total: {total_sims} | drop obs: {dropped_obs} | no recession: {no_recession} | episodes: {len(results_list)}")
 
#     if not results_list:
#         raise ValueError("No recession episodes found.")
 
#     results_df = pd.DataFrame(results_list)
 
#     ### formal test: is mean peak lag significantly negative? ###
#     from scipy.stats import ttest_1samp
#     credit_tstat,    credit_pvalue    = ttest_1samp(results_df["credit_peak_lag"],    popmean=0)
#     debt_tstat,      debt_pvalue      = ttest_1samp(results_df["debt_peak_lag"],      popmean=0)
#     deflation_tstat, deflation_pvalue = ttest_1samp(results_df["deflation_peak_lag"], popmean=0)
 
#     ### summary results ###
#     summary = pd.DataFrame([{
#         "total_simulations":           total_sims,
#         "total_episodes":              len(results_list),
 
#         # mean peak lags (negative => peaks before recession)
#         "mean_credit_peak_lag":        results_df["credit_peak_lag"].mean(),
#         "mean_debt_peak_lag":          results_df["debt_peak_lag"].mean(),
#         "mean_deflation_peak_lag":     results_df["deflation_peak_lag"].mean(),
 
#         # share peaking before recession
#         "share_credit_peaks_before":    results_df["credit_peaks_before"].mean(),
#         "share_debt_peaks_before":      results_df["debt_peaks_before"].mean(),
#         "share_minsky":               results_df["minsky"].mean(),
#         "share_deflation_peaks_before": results_df["deflation_peaks_before"].mean(),
#         "share_fisher":                 results_df["fisher"].mean(),
 
#         # formal t-test: mean peak lag < 0
#         "credit_peak_tstat":           credit_tstat,
#         "credit_peak_pvalue":          credit_pvalue,
#         "debt_peak_tstat":             debt_tstat,
#         "debt_peak_pvalue":            debt_pvalue,
#         "deflation_peak_tstat":        deflation_tstat,
#         "deflation_peak_pvalue":       deflation_pvalue,
#     }])
 
#     return results_df, summary, all_credit_rate_windows, all_debt_ratio_windows, all_inflation_windows
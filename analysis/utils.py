import numpy as np
import pandas as pd
import powerlaw as pl
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats
from pathlib import Path
from sqlalchemy import Engine

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
    data['year'] = (data['time'] // steps).astype(int)
    # real gdp growth 
    data['rgdp_growth'] = (data.groupby('simulation_id')['real_gdp'].transform(lambda x: np.log(x) - np.log(x.shift(steps))))
    # standardised real gdp growth 
    data['std_rgdp_growth'] = (data.groupby('simulation_id')['rgdp_growth'].transform(lambda x: (x - x.mean()) / x.std()))
    # consumption growth 
    data['consumption_growth'] = (data.groupby('simulation_id')['real_consumption'].transform(lambda x: np.log(x) - np.log(x.shift(steps))))
    # investment growth 
    data['investment_growth'] = (data.groupby('simulation_id')['real_investment'].transform(lambda x: np.log(x) - np.log(x.shift(steps))))
    # price index (cfirms)
    data['price_index'] = data['nominal_consumption'] / data['real_consumption']
    # inflation
    data['inflation'] = (data.groupby('simulation_id')['price_index'].transform(lambda x: np.log(x) - np.log(x.shift(steps))))
    # wage index 
    data['wage_index'] = data['wages'] / data['employment']
    # wage inflation
    data['wage_inflation'] = (data.groupby('simulation_id')['wage_index'].transform(lambda x: np.log(x) - np.log(x.shift(steps))))
    # ESL / nGDP
    data['esl_gdp'] = data['esl'] / data['nominal_gdp']
    # debt ratio
    data['debt_ratio'] = data['debt'] / data['nominal_gdp']
    # wage share
    data['wage_share'] = data['wages'] / data['nominal_gdp']
    # profit share
    data['profit_share'] = data['profits'] / data['nominal_gdp']
    # normalised productivity to start date
    data['productivity'] = (data['real_gdp'] / data['employment']) / (data['real_gdp'][start] / data['employment'][start])
    # productivity growth
    data['productivity_growth'] = (data.groupby('simulation_id')['productivity'].transform(lambda x: np.log(x) - np.log(x.shift(steps))))
    # credit: credit (change in debt) as a percentage of GDP
    data['credit_gdp'] = (data.groupby('simulation_id')['debt'].transform(lambda x: x - x.shift(steps)) / data['nominal_gdp'])
    # change unemployment
    data['change_unemployment'] = data['unemployment_rate'] - data['unemployment_rate'].shift(steps)
    
    ### probability of at least one crisis in a given year ###
    # 1. Crisis flag per quarter
    data['crises'] = (data['rgdp_growth'] < -0.03).astype(int)

    # 2. Crisis probability PER YEAR (across simulations) - ONE LINE
    yearly_crisis_prob = (
        data.groupby(['year', 'simulation_id'])['crises']
        .max()  # any crisis that year?
        .groupby('year')
        .mean()  # % simulations with crisis
        .reset_index(name='yearly_crisis_prob')
    )

    # 3. Merge back to main data
    data = data.merge(yearly_crisis_prob, on='year', how='left')
    
    # quarterly default probability
    data['cfirm_probability_default'] = data['cfirm_defaults'] / params['simulation']['num_cfirms']
    data['kfirm_probability_default'] = data['kfirm_defaults'] / params['simulation']['num_kfirms']
    data['bank_probability_default']  = data['bank_defaults']  / params['simulation']['num_banks']
    
    # average per year default probability, per simulation, split by crisis 
    yearly_probs = (
        data.groupby(['simulation_id','year','crises'])[['cfirm_probability_default','kfirm_probability_default','bank_probability_default']]
        .mean()
        .reset_index()
    )
    
    # average across years within each simulation, conditional on there being a crisis
    cond_probs = (
        yearly_probs.groupby(['simulation_id','crises'])[['cfirm_probability_default','kfirm_probability_default','bank_probability_default']]
        .mean()
        .reset_index()
    )

    # Pivot to get separate columns for crisis=1 (true) and crisis=0 (false)
    cond_probs = cond_probs.pivot(index='simulation_id', columns='crises')
    cond_probs.columns = [f"{var}_crisis{c}" for var,c in cond_probs.columns]
    cond_probs = cond_probs.reset_index()

    # merge back into main data so each row has the conditional averages
    data = data.merge(cond_probs, on='simulation_id', how='left')
        
    # recession indicator 
    # 1. negative GDP growth indicator
    data['neg_rgdp'] = (data['rgdp_growth'] < 0).astype(int)

    # 2. identify consecutive spells (per simulation)
    data['neg_spell'] = (
        data.groupby('simulation_id')['neg_rgdp']
        .transform(lambda x: (x != x.shift()).cumsum())
    )

    # 3. spell lengths
    spell_lengths = (
        data[data['neg_rgdp'] == 1]
        .groupby(['simulation_id', 'neg_spell'])
        .size()
        .reset_index(name='spell_length')
    )

    # 4. keep only spells with length >= 3
    valid_spells = spell_lengths[spell_lengths['spell_length'] >= 2]

    # 5. merge back
    data = data.merge(
        valid_spells[['simulation_id', 'neg_spell']],
        on=['simulation_id', 'neg_spell'],
        how='left',
        indicator=True
    )

    # 6. recession indicator
    data['recession'] = (data['_merge'] == 'both').astype(int)

    # cleanup
    data.drop(columns=['neg_rgdp', 'neg_spell', '_merge'], inplace=True)
    
    # drop transient data
    data = data.loc[data['time'] > start * steps]
    # reset index
    data.reset_index(inplace=True, drop=True)
    return data

def box_plot_scenarios(
    plot_data: dict[str,pd.DataFrame], 
    variable: str,
    figsize: tuple[float,float] | None=None,
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
        plot_data[scenario] = plot_data[scenario][variable].to_numpy().ravel()
    
    ### box plot ###
    plt.figure(figsize=figsize)
    bplot = plt.boxplot(plot_data.values(), patch_artist=True, showfliers=False, medianprops=dict(color='black', linewidth=2), whis=whis)
    
    ### set box colour ###
    if colours:
        for patch, color in zip(bplot['boxes'], colours):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
            
    ### figure settings ###
    # decimal places 
    if yticks is not None:
        plt.yticks(yticks, [f"{y:.{dp}f}" for y in yticks], fontsize=fontsize)
    else:
        plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter(f'%.{dp}f'))
    # y axis ticks
    plt.yticks(yticks, fontsize=fontsize)
    # y axis limits
    plt.ylim(ylim)
    # x axis ticks
    plt.xticks(xticks, xlabels, fontsize=fontsize)
    # save figure
    plt.savefig(figure_path / f"box_plot_{variable}", bbox_inches='tight')
    
def normality_tests(data: pd.Series, significance: float=0.01):
    # Kolmogorov-Smirnov Test for Nomaliy
    ks_result = stats.kstest(data, 'norm')
    # print test outcome
    print('\n- Kolmogorov-Smirnov Test:')
    if ks_result.pvalue < significance:
        print(f"  - Reject null hypothesis => not normally distributed ({significance*100}% significance level)")
    else:
        print(f"  - Accept null hypothesis => normally distributed ({significance*100}% significance level)")
    # print results
    print(f'  - KS test resutls: \n{ks_result}\n')

    # Shapiro-Wilk Test for Nomaliy
    sw_result = stats.shapiro(data)
    # print test outcome
    print('- Shapiro-Wilk Test:')
    if sw_result.pvalue < significance:
        print(f"  - Reject null hypothesis => not normally distributed ({significance*100}% significance level)")
    else:
        print(f"  - Accept null hypothesis => normally distributed ({significance*100}% significance level)")
    # print results
    print(f'  - SW test results: \n{sw_result}\n')

    # Anderson-Darling Test for Nomaliy
    ad_result = stats.anderson(data, 'norm')
    # print test outcome
    print('- Anderson-Darling Test')
    # print results
    print(f'  - AD test results: \n{ad_result}')
    if ad_result.statistic >= .6:
        p = np.exp(1.2937 - 5.709*ad_result.statistic - .0186*(ad_result.statistic**2))
    elif ad_result.statistic >=.34:
        p = np.exp(.9177 - 4.279*ad_result.statistic - 1.38*(ad_result.statistic**2))
    elif ad_result.statistic >.2:
        p = 1 - np.exp(-8.318 + 42.796*ad_result.statistic - 59.938*(ad_result.statistic**2))
    else:
        p = 1 - np.exp(-13.436 + 101.14*ad_result.statistic - 223.73*(ad_result.statistic**2))
    print("  - AD p-value = ", p)
    if p < significance:
        print(f"  - Reject null hypothesis => not normally distributed ({significance*100}% significance level)")
    else:
        print(f"  - Accept null hypothesis => normally distributed ({significance*100}% significance level)")

def plot_autocorrelation(
    simulated: np.typing.ArrayLike, 
    empirical: np.typing.ArrayLike, 
    figsize: tuple[int,int], 
    fontsize: int, 
    savefig: str, 
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
    # plot results
    plt.figure(figsize=figsize)
    # x values (lags)
    x = np.arange(0, lags+1)
    # plot empirical autocorrelation 
    plt.plot(emp_autocorr, color='k', linestyle='--', linewidth=1, label='Empirical')
    # plot simulated autocorrelation median
    plt.plot(sim_autocorr_median, color='k', linewidth=1, label='Simulated')
    # plot confidence interval
    plt.fill_between(x, sim_autocorr_median, sim_autocorr_upper, color='grey', alpha=0.2, label='95% CI')
    plt.fill_between(x, sim_autocorr_median, sim_autocorr_lower, color='grey', alpha=0.2)
    # plot 0 line
    plt.axhline(0, color='k')
    # figure ticks
    plt.yticks([1.00,0.75,0.50,0.25,0.00,-0.25,-0.50,-0.75,-1.00], ['1.00','0.75','0.50','0.25','0.00','–0.25','–0.50','–0.75','–1.00'], fontsize=fontsize)
    plt.xticks([0,5,10,15,20], [0,5,10,15,20], fontsize=fontsize)
    # legend
    plt.legend(fontsize=fontsize, loc='upper right')
    # save figure
    plt.savefig(savefig, bbox_inches='tight')
    plt.close()

def plot_cross_correlation(
    xsimulated: np.typing.ArrayLike,
    ysimulated: np.typing.ArrayLike,
    xempirical: np.typing.ArrayLike,
    yempirical: np.typing.ArrayLike,
    figsize: tuple[int,int], 
    savefig: str,
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
        
        figsize : tuple (int, int) 
            size of figure (x-axis, y-axis)
            
        fontsize : int
            fontsize of legend and ticks
        
        lags: int
            number of correlation lags, in range (-lags, lags)
        
        lamda : int
            Hodrick-Prescott filter lambda parameter (quarterly => lambda=1600)
            
        savefig : str
            path and figure name 
    """
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
    # clear figure
    plt.clf()
    # start figure
    plt.figure(figsize=figsize)
    # x values (lags)
    x = np.arange(0, lags*2 + 1)
    # plot empirical xcorr
    plt.plot(emp_xcorr, color='k', linestyle='--', linewidth=1, label='Empirical')
    # plot median simulated xcorr
    plt.plot(sim_xcorr_median, color='k', linewidth=1, label='Simulated')
    # plot confidence interval
    plt.fill_between(x, sim_xcorr_median, sim_xcorr_upper, color='grey', alpha=0.2, label='95% CI')
    plt.fill_between(x, sim_xcorr_median, sim_xcorr_lower, color='grey', alpha=0.2)
    # plot 0 line
    plt.axhline(0, color='k')
    # figure ticks
    plt.yticks([1.00,0.75,0.50,0.25,0.00,-0.25,-0.50,-0.75,-1.00], ['1.00','0.75','0.50','0.25','0.00','–0.25','–0.50','–0.75','–1.00'], fontsize=fontsize)
    plt.xticks([0,int(0.5*lags),int(lags),int(1.5*lags),int(2*lags)], [f'–{lags}',f'–{int(0.5*lags)}',0,int(0.5*lags),lags], fontsize=fontsize)
    # legend
    plt.legend(fontsize=fontsize, loc='upper right')
    # save figure
    plt.savefig(savefig, bbox_inches='tight')
    plt.close()

def plot_ccdf(
    data: np.typing.ArrayLike,
    figsize: tuple[int, int],
    fontsize: int,
    ylim: tuple[float, float] | None=None,
    savefig: str | None=None,
    dp: int = 0,
) -> None:
    """
    Plots the complementary cumulative distribution function (CCDF) for a given
    continuous dataset, estimates a power-law tail with bounded xmin search,
    and compares against a lognormal distribution.
    """

    # --- basic data hygiene ---
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    data = data[data > 1]

    if data.size < 100:
        raise ValueError("Too few positive samples for reliable power-law fitting.")

    # --- power-law fit ---
    results = pl.Fit(
        data,
        discrete=False,
        verbose=False,
    )

    alpha = results.alpha
    xmin = results.xmin

    n_tail = np.sum(data >= xmin)
    if n_tail < 50:
        print("Warning: very few points in the power-law tail (n < 50).")

    # --- empirical CCDF ---
    x = np.sort(data)
    cdf = np.arange(1, len(x) + 1) / len(x)
    ccdf = 1.0 - cdf

    plt.figure(figsize=figsize)
    plt.scatter(
        x,
        ccdf,
        color="skyblue",
        edgecolors="k",
        alpha=0.7,
        s=30,
        label="Empirical CCDF",
    )

    # --- power-law CCDF (rescaled at xmin) ---
    idx = np.searchsorted(x, xmin)
    rescale = ccdf[idx]

    powerlaw_ccdf = np.where(
        x >= xmin,
        (xmin / x) ** (alpha - 1) * rescale,
        np.nan,
    )

    plt.plot(
        x,
        powerlaw_ccdf,
        color="limegreen",
        linewidth=3,
        label=rf"Power law ($\alpha$ = {alpha:.2f})",
    )

    # --- xmin marker ---
    plt.axvline(
        xmin,
        color="k",
        linestyle=":",
        label=rf"$x_{{\min}}$ = {round(xmin, dp)}",
    )

    # --- lognormal comparison ---
    ln_shape, ln_loc, ln_scale = stats.lognorm.fit(data, floc=0)
    ln_ccdf = 1.0 - stats.lognorm.cdf(x, ln_shape, ln_loc, ln_scale)

    plt.plot(
        x,
        ln_ccdf,
        color="r",
        linestyle="--",
        linewidth=2,
        label="Log-normal",
    )

    # --- plot cosmetics ---
    plt.loglog()
    plt.legend(loc="lower left", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.ylim(ylim)

    plt.savefig(savefig, bbox_inches="tight")
    plt.close()

    # --- diagnostics ---
    print(f"Power-law exponent (alpha) = {alpha:.4f}")
    print(f"Power-law xmin = {xmin:.4f}")
    print(f"Tail size (n >= xmin) = {n_tail}")
    print(
        "Distribution comparison (PL vs LN) =",
        results.distribution_compare("power_law", "lognormal"),
        "\n",
    )
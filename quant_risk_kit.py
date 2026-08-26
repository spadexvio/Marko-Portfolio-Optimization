import pandas as pd
import scipy.stats
import numpy as np
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def annualized_returns(r , periods_per_year):
    """
    Loads the data and calculates annualized returns 
    """
    compounded_growth = (r+1).prod()
    n_months = r.shape[0]
    return compounded_growth**(periods_per_year/n_months) - 1

def annualized_volatility(r, periods_per_year):
    """
    Loads the data and returns the annualized volatility
    """
    vol = r.std()
    return vol*np.sqrt(periods_per_year)

def sharpe_ratio(r, risk_free_rate, periods_per_year):
    """
    Loads the dataand returns the sharpe ratio as the ratio of excess return to volatility
    """
    excess_return = annualized_returns(r, periods_per_year) - risk_free_rate
    return excess_return/annualized_volatility(r, periods_per_year)

def drawdown(return_series : pd.Series):
    """
    Takes a time series of asset returns
    Computes and returns a data frame which contains:
    Wealth Index, Drawdown, Max Drawdown
    """
    wealth_index=1000*(1+return_series).cumprod()
    previous_peaks=wealth_index.cummax()
    drawdowns=(wealth_index-previous_peaks)/previous_peaks
    return pd.DataFrame({
        "Wealth": wealth_index,
        "Peaks": previous_peaks,
        "Drawdown": drawdowns
    })

def get_ffme_returns():
    """
    Load the Fama-French dataset for the returns of the Top and Bottom deciles by Market Cap
    """
    me_m=pd.read_csv("Portfolios_Formed_on_ME_monthly_EW.csv",
                   header=0, index_col=0, parse_dates=True, date_format='%Y%m', na_values=-99.99)
    rets=me_m[['Lo 10', 'Hi 10']]
    rets.columns=['SmallCap', 'LargeCap']
    rets.index=rets.index.to_period('M')
    rets=rets/100
    return rets

def get_hfi_returns():
    """
    Load and format the EDHEC Hedge Funds Index returns
    """
    hfi=pd.read_csv("edhec-hedgefundindices.csv",
                   header=0, index_col=0, parse_dates=True, na_values=-99.99)
    hfi.index = pd.to_datetime(hfi.index, dayfirst=True)
    hfi.index=hfi.index.to_period('M')
    hfi=hfi/100
    return hfi

def get_xl_returns():
    """
    Load and format the EDHEC Hedge Funds Index returns
    """
    sector_tickers = ['XLK', 'XLF', 'XLE', 'XLV', 'XLU']
    df_sectors = yf.download(sector_tickers, start="2006-01-01", end="2010-12-31")['Close']
    rets = df_sectors.pct_change().dropna()
    rets.index = pd.to_datetime(rets.index, dayfirst=True)
    rets.index=rets.index.to_period('M')
    return rets

def get_ind_returns():
    """
    Get the Ken French 30 industry returns
    """
    ind= pd.read_csv("ind30_m_vw_rets.csv",
                 header=0, index_col=0, parse_dates=True, date_format= '%Y%m', na_values=-99.99)
    ind.index=pd.to_datetime(ind.index).to_period('M')
    ind=ind/100
    ind.columns = ind.columns.str.strip()
    return ind

def get_ind_size():
    """
    Get the Ken French 30 industry size
    """
    ind= pd.read_csv("ind30_m_size.csv",
                 header=0, index_col=0, parse_dates=True, date_format= '%Y%m', na_values=-99.99)
    ind.index=pd.to_datetime(ind.index).to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind

def get_ind_nfirms():
    """
    Get the Ken French 30 industry size
    """
    ind= pd.read_csv("ind30_m_nfirms.csv",
                 header=0, index_col=0, parse_dates=True, date_format= '%Y%m', na_values=-99.99)
    ind.index=pd.to_datetime(ind.index).to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind

def semideviation(r : pd.Series):
    """
    Calculates deviation of the negative returns data
    """
    return r[r<0].std(ddof=0)

def skewness(r : pd.Series):
    """
    Load the return series and return its skewness
    """
    demeaned_r=r-r.mean()
    exp=r.std(ddof=0)
    skew=((demeaned_r)**3).mean()/(exp**3)
    return skew

def kurtosis(r : pd.Series):
    """
    Load the return series and return its skewness
    """
    demeaned_r=r-r.mean()
    exp=r.std(ddof=0)
    kurt=((demeaned_r)**4).mean()/(exp**4)
    return kurt

def is_normal(r: pd.Series , level=0.01):
    """
    Applies the Jarque_Bera test to decide if the series is normal or not
    Test is applied to the 1% level by default
    Returns True if the hypothesis of normality is accepted or returns False
    """
    statistic,p_value = scipy.stats.jarque_bera(r)
    return p_value>level

def var_historic(r , level=5):
    """
    Calculates VaR from Historical data
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("Expected r to be Seies or DataFrame")

def var_gaussian(r, level=5, modified=False):
    """
    Returns the parametric/modified gaussian VaR for a series or a Dataframe 
    """
    z=norm.ppf(level/100)
    if modified:
        s=skewness(r)
        k=kurtosis(r)
        sz = ((z**2-1)*5/6)-((2*z**3-5*z)*s**2)/36
        sk = (z**3-3*z)*(k-3)/24
        z = z+sz+sk
    return -(r.mean() + z*r.std(ddof=0))

def cvar_historic(r , level=5):
    """
    Calculates VaR from Historical data
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    elif isinstance(r, pd.Series):
        is_beyond = r < -var_historic(r, level=level)
        return -r[is_beyond].mean()
    else:
        raise TypeError("Expected r to be Seies or DataFrame")

def portfolio_return(weights, returns):
    """
    Weights ---> returns
    """
    return weights.T @ returns

def portfolio_vola(weights, covmat):
    """
    Weights ---> Volatility
    """
    return (weights.T @ covmat @ weights)**0.5

def plot_eff_front_2(r, covmat, n_points, style):
    """
    Loads the 2 assets data and plot the Efficient Frontier curve
    """
    weights = [np.array([w,1-w]) for w in np.linspace(0,1,n_points)]
    rets = [portfolio_return(w, r) for w in weights]
    vol = [portfolio_vola(w, covmat) for w in weights]
    eff=pd.DataFrame({"Return":rets,"Risk":vol})
    eff.plot.line(x="Risk",y="Return", style=style)

def minimize_vol(er, covmat, target_return):
    """
    Target Return ----> Weight Vector
    """
    n= er.shape[0]
    init_guess= np.repeat(1/n, n)
    bounds = ((0.0, 1.0),)*n
    return_is_target = {
        'type': 'eq',
        'args': (er,),
        'fun' : lambda weights, er: target_return - portfolio_return(weights,er)
    }
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights : np.sum(weights) -1
    }
    results = minimize(portfolio_vola, init_guess, args=(covmat,), method = "SLSQP",
                      options={'disp': False},
                      constraints=(return_is_target, weights_sum_to_1),
                      bounds=bounds)
    return results.x

def optimal_weights(n_points, er, covmat):
    """
    --> list of weights to run the optimizer on to minimize the vol
    """
    target_rets = np.linspace(er.min(), er.max(), n_points) 
    weights = [minimize_vol(er, covmat, target_return) for target_return in target_rets]
    return weights

def gmv(covmat):
    """
    Returns the weights of the Global Minimum Volatility portfolio
    given a covariance matrix
    """
    n = covmat.shape[0]
    return msr(np.repeat(1, n), covmat, 0)
    
def plot_eff_front_N(er, covmat, n_points, show_cml=False, risk_free_rate=0, show_ew= False, show_gmv=False):
    """
    Loads the N assets data and plot the Efficient Frontier curve
    """
    weights = optimal_weights(n_points, er, covmat)
    rets = [portfolio_return(w, er) for w in weights]
    vol = [portfolio_vola(w, covmat) for w in weights]
    eff=pd.DataFrame({"Return":rets,"Risk":vol})
    ax = eff.plot.line(x="Risk",y="Return", style= ".-")
    if show_ew: # display EW
        n=er.shape[0]
        w_ew=np.repeat(1/n,n) 
        vol_ew = portfolio_vola(w_ew, covmat)
        r_ew= portfolio_return(w_ew, er)
        ax.plot([vol_ew], [r_ew], color="goldenrod", marker="o", markersize=12)
    if show_gmv:
        #vol_gmv = min(vol)    
        #r_gmv= rets[vol.index(min(vol))]     
        #wt= weights[vol.index(min(vol))]
        w_gmv = gmv(covmat)
        r_gmv = portfolio_return(w_gmv, er)
        vol_gmv = portfolio_vola(w_gmv, covmat)
        ax.plot([vol_gmv], [r_gmv], color="purple", marker="o", markersize=6)
        #return [vol_gmv, r_gmv]
    if show_cml:
        rf=risk_free_rate
        w_msr = msr(er, covmat, rf)
        r_msr= portfolio_return(w_msr, er)
        vol_msr= portfolio_vola(w_msr, covmat)
        # Add CML
        ax.set_xlim(left=0)
        cml_x = [0, vol_msr]
        cml_y = [rf, r_msr]
        return [(w_gmv), (vol_gmv, r_gmv), ax.plot(cml_x, cml_y, color="green", marker="o", linestyle="dashed")]
    else:
        return ax

def msr(er, covmat, risk_free_rate):
    """
    Risk free return and Maxm Sharpe Ratio ---> Weights
    """
    n= er.shape[0]
    init_guess= np.repeat(1/n, n)
    bounds = ((0.0, 1.0),)*n
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights : np.sum(weights) -1
    }
    def neg_sharpe_ratio(weights, er, covmat, risk_free_rate):
        r = portfolio_return(weights, er)
        vol = portfolio_vola(weights, covmat)
        return -((r-risk_free_rate)/vol)
        
    results = minimize(neg_sharpe_ratio, init_guess, args=(er, covmat, risk_free_rate), method = "SLSQP",
                      options={'disp': False},
                      constraints=(weights_sum_to_1),
                      bounds=bounds)
    return results.x

def get_tot_mkt_rets():
    """
    Return the total market index returns
    """
    ind_return = get_ind_returns()
    ind_nfirms = get_ind_nfirms()   # No. of firms in each sector
    ind_size = get_ind_size() 
    ind_mktcap = ind_nfirms * ind_size
    total_mktcap = ind_mktcap.sum(axis=1)
    ind_capweight = ind_mktcap.divide(total_mktcap, axis=0)
    total_market_return = (ind_capweight * ind_return).sum(axis = 1)
    total_market_index = drawdown(total_market_return).Wealth
    return total_market_return

def run_cppi(risky_r, safe_r = None, m=3, start= 1000, floor = 0.8, risk_free_rate= 0.03, drawdown = None):
    """
    Runs a backtest of CPPI Strategy - Returns a dict of Asset Value History, Risk Budget History, Risky Weight History
    """
    dates = risky_r.index
    n_steps = len(dates)
    account_value = start
    floor_value = start * floor
    peak = start

    if isinstance(risky_r, pd.Series):
        risky_r = pd.DataFrame(risky_r)

    if safe_r is None:
        safe_r = pd.DataFrame().reindex_like(risky_r)
        safe_r = pd.DataFrame(risk_free_rate/12, index=risky_r.index, columns=risky_r.columns)
    account_history = pd.DataFrame().reindex_like(risky_r)
    cushion_history = pd.DataFrame().reindex_like(risky_r)
    risky_w_history = pd.DataFrame().reindex_like(risky_r)

    for step in range(n_steps):
        if drawdown is not None:
            peak = np.maximum(peak, account_value)
            floor_value = (1 - drawdown) * peak
        cushion = (account_value - floor_value) / account_value
        risky_w = m * cushion
        risky_w = np.minimum(risky_w, 1)
        risky_w = np.maximum(risky_w, 0)
        safe_w = 1 - risky_w
        risky_alloc = risky_w * account_value
        safe_alloc = safe_w * account_value
        # update the accnt value for this time stamp
        account_value = risky_alloc * (1+risky_r.iloc[step]) + safe_alloc * (1+safe_r.iloc[step])
        # save the values for history
        cushion_history.iloc[step] = cushion
        risky_w_history.iloc[step] = risky_w
        account_history.iloc[step] = account_value
    risky_wealth = start * (1+risky_r).cumprod()
    backtest_result = {
        "Wealth": account_history,
        "Risky_Wealth": risky_wealth,
        "Risk Budget" : cushion_history,
        "Risky allocation" : risky_w_history,
        "m" : m,
        "start": start,
        "floor" : floor,
        "risky_rate" : risky_r,
        "safe_rate" : safe_r
    }
    return backtest_result

def summary_stats(r, riskfree_rate=0.03):
    """
    Return a DataFrame that contains aggregated summary stats for the returns in the columns of r
    """
    ann_r = r.aggregate(annualized_returns, periods_per_year=12)
    ann_vol = r.aggregate(annualized_volatility, periods_per_year=12)
    ann_sr = r.aggregate(sharpe_ratio, risk_free_rate=riskfree_rate, periods_per_year=12)
    dd = r.aggregate(lambda r: drawdown(r).Drawdown.min())
    skew = r.aggregate(skewness)
    kurt = r.aggregate(kurtosis)
    cf_var5 = r.aggregate(var_gaussian, modified=True)
    hist_cvar5 = r.aggregate(cvar_historic)
    return pd.DataFrame({
        "Annualized Return": ann_r,
        "Annualized Vol": ann_vol,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Cornish-Fisher VaR (5%)": cf_var5,
        "Historic CVaR (5%)": hist_cvar5,
        "Sharpe Ratio": ann_sr,
        "Max Drawdown": dd
    })

def gbm(n_years = 10 , n_scenarios = 1000, mu = 0.7, sigma = 0.15, prices= True, steps_per_year= 12, s_0=100):  
    """
    Evolution of Geometric Brownian Motion trajectories, such as for Stock Prices through Monte Carlo
    :param n_years:  The number of years to generate data for
    :param n_paths: The number of scenarios/trajectories
    :param mu: Annualized Drift, e.g. Market Return
    :param sigma: Annualized Volatility
    :param steps_per_year: granularity of the simulation
    :param s_0: initial value
    :return: a numpy array of n_paths columns and n_years*steps_per_year rows
    """
    # Derive per-step Model Parameters from User Specifications
    dt = 1/steps_per_year
    n_steps = int(n_years*steps_per_year) + 1
    # the standard way ...
    # rets_plus_1 = np.random.normal(loc=mu*dt+1, scale=sigma*np.sqrt(dt), size=(n_steps, n_scenarios))
    # without discretization error ...
    rets_plus_1 = np.random.normal(loc=(1+mu)**dt, scale=(sigma*np.sqrt(dt)), size=(n_steps, n_scenarios))
    rets_plus_1[0] = 1
    ret_val = s_0*pd.DataFrame(rets_plus_1).cumprod() if prices else rets_plus_1-1
    rets_plus_1[0] = 1
    ret_val = s_0*pd.DataFrame(rets_plus_1).cumprod() if prices else rets_plus_1-1
    return pd.DataFrame(ret_val)

def discount(t, r):
    """
    Compute the price of a pure discount bond that pays a dollar at time period t
    and r is the per-period interest rate
    returns a |t| x |r| Series or DataFrame
    r can be a float, Series or DataFrame
    returns a DataFrame indexed by t
    """
    discounts = pd.DataFrame([(r+1)**-i for i in t])
    discounts.index = t
    return discounts

def pv(flows, r):
    """
    Compute the present value of a sequence of cash flows given by the time (as an index) and amounts
    r can be a scalar, or a Series or DataFrame with the number of rows matching the num of rows in flows
    """
    dates = flows.index
    discounts = discount(dates, r)
    return discounts.multiply(flows, axis='rows').sum()

def funding_ratio(assets, liabilities, r):
    """
    Computes the funding ratio of a series of liabilities, based on an interest rate and current value of assets
    """
    return pv(assets, r)/pv(liabilities, r)

import math
def bond_cash_flows(maturity, principal=100, coupon_rate=0.03, coupons_per_year=12):
    """
    Returns the series of cash flows generated by a bond,
    indexed by the payment/coupon number
    """
    n_coupons = round(maturity*coupons_per_year)
    coupon_amt = principal*coupon_rate/coupons_per_year
    coupon_times = np.arange(1, n_coupons+1)
    cash_flows = pd.Series(data=coupon_amt, index=coupon_times)
    cash_flows.iloc[-1] += principal # add the principal to the last payment
    return cash_flows

def bond_price(maturity, principal=100, coupon_rate=0.03, coupons_per_year=12, discount_rate=0.03):
    """
    Computes the price of a bond that pays regular coupons until maturity
    at which time the principal and the final coupon is returned
    This is not designed to be efficient, rather,
    it is to illustrate the underlying principle behind bond pricing!
    If discount_rate is a DataFrame, then this is assumed to be the rate on each coupon date
    and the bond value is computed over time.
    i.e. The index of the discount_rate DataFrame is assumed to be the coupon number
    """
    if isinstance(discount_rate, pd.DataFrame):
        pricing_dates = discount_rate.index
        prices = pd.DataFrame(index=pricing_dates, columns=discount_rate.columns)
        for t in pricing_dates:
            prices.loc[t] = bond_price(maturity-t/coupons_per_year, principal, coupon_rate, coupons_per_year,
                                      discount_rate.loc[t])
        return prices
    else: # base case ... single time period
        if maturity <= 0: return principal+principal*coupon_rate/coupons_per_year
        cash_flows = bond_cash_flows(maturity, principal, coupon_rate, coupons_per_year)
        return pv(cash_flows, discount_rate/coupons_per_year)

def macaulay_duration(flows, discount_rate):
    """
    Computes the Macaulay duration of a sequence of cash flows
    """
    discounted_flows = discount(flows.index, discount_rate)*flows
    weights = discounted_flows/(discounted_flows).sum()
    return ((flows.index)*weights).sum()

def match_durations(cf_t, cf_s, cf_l, discount_rate):
    """
    Returns the weight W in cf_s that, along with (1-W) in cf_l will have an effective
    duration that matches cf_t
    """
    d_t = macaulay_duration(cf_t, discount_rate)
    d_s = macaulay_duration(cf_s, discount_rate)
    d_l = macaulay_duration(cf_l, discount_rate)
    return (d_l - d_t)/(d_l - d_s)

def inst_to_ann(r):
    """
    Convert an instantaneous interest rate to an annual interest rate
    """
    return np.expm1(r)

def ann_to_inst(r):
    """
    Convert an instantaneous interest rate to an annual interest rate
    """
    return np.log1p(r)

import math
def cir(n_years = 10, n_scenarios=1, a=0.05, b=0.03, sigma=0.05, steps_per_year=12, r_0=None):
    """
    Generate random interest rate evolution over time using the CIR model
    b and r_0 are assumed to be the annualized rates, not the short rate
    and the returned values are the annualized rates as well
    """
    if r_0 is None: r_0 = b 
    r_0 = ann_to_inst(r_0)
    dt = 1/steps_per_year
    num_steps = int(n_years*steps_per_year) +1 # because n_years might be a float
    
    shock = np.random.normal(0, scale=np.sqrt(dt), size=(num_steps, n_scenarios))
    rates = np.empty_like(shock)
    rates[0] = r_0

    ## For Price Generation
    h = math.sqrt(a**2 + 2*sigma**2)
    prices = np.empty_like(shock)
    ####

    def price(ttm, r):
        _A = ((2*h*math.exp((h+a)*ttm/2))/(2*h+(h+a)*(math.exp(h*ttm)-1)))**(2*a*b/sigma**2)
        _B = (2*(math.exp(h*ttm)-1))/(2*h + (h+a)*(math.exp(h*ttm)-1))
        _P = _A*np.exp(-_B*r)
        return _P
    prices[0] = price(n_years, r_0)
    ####
    
    for step in range(1, num_steps):
        r_t = rates[step-1]
        d_r_t = a*(b-r_t)*dt + sigma*np.sqrt(r_t)*shock[step]
        rates[step] = abs(r_t + d_r_t)
        # generate prices at time t as well ...
        prices[step] = price(n_years-step*dt, rates[step])

    rates = pd.DataFrame(data=inst_to_ann(rates), index=range(num_steps))
    ### for prices
    prices = pd.DataFrame(data=prices, index=range(num_steps))
    ###
    return rates, prices

def bond_total_return(monthly_prices, principal, coupon_rate, coupons_per_year):
    """
    Computes the total return of a Bond based on monthly bond prices and coupon payments
    Assumes that dividends (coupons) are paid out at the end of the period (e.g. end of 3 months for quarterly div)
    and that dividends are reinvested in the bond
    """
    coupons = pd.DataFrame(data = 0.0, index=monthly_prices.index, columns=monthly_prices.columns)
    t_max = monthly_prices.index.max()
    pay_date = np.linspace(12/coupons_per_year, t_max, int(coupons_per_year*t_max/12), dtype=int)
    coupons.iloc[pay_date] = principal*coupon_rate/coupons_per_year
    total_returns = (monthly_prices + coupons)/monthly_prices.shift()-1
    return total_returns.dropna()

def terminal_values(rets):
    """
    Computes the terminal values from a set of returns supplied as a T x N DataFrame
    Return a Series of length N indexed by the columns of rets
    """
    return (rets+1).prod()

def terminal_stats(rets, floor = 0.8, cap=np.inf, name="Stats", dd=0.25):
    """
    Produce Summary Statistics on the terminal values per invested dollar
    across a range of N scenarios
    rets is a T x N DataFrame of returns, where T is the time-step (we assume rets is sorted by time)
    Returns a 1 column DataFrame of Summary Stats indexed by the stat name 
    """
    terminal_wealth = (rets+1).prod()
    breach = terminal_wealth < floor
    reach = terminal_wealth >= cap
    max_drawdowns = summary_stats(rets)["Max Drawdown"]
    drawdown_breaches = max_drawdowns < -dd  # True for paths breaching 25% max DD
    #peaks = drawdown(rets[drawdown_breaches])["Peaks"]
    # Breach rate for Max Drawdown constraint:
    p_dd_breach = drawdown_breaches.mean() if drawdown_breaches.any() else np.nan
    #dd_e_short = (-(drawdown_breaches + dd)* peaks).mean() if drawdown_breaches.any() else np.nan
    p_breach = breach.mean() if breach.sum() > 0 else np.nan
    p_reach = reach.mean() if reach.sum() > 0 else np.nan
    e_short = (floor-terminal_wealth[breach]).mean() if breach.sum() > 0 else np.nan
    e_surplus = (-cap+terminal_wealth[reach]).mean() if reach.sum() > 0 else np.nan
    sum_stats = pd.DataFrame.from_dict({
        "mean": terminal_wealth.mean(),
        "std" : terminal_wealth.std(),
        "p_breach": p_breach,
        "e_short":e_short,
        "p_reach": p_reach,
        "e_surplus": e_surplus,
        "dd_breach": p_dd_breach,
        #"dd_e_short": dd_e_short
    }, orient="index", columns=[name])
    return sum_stats

def bt_mix(r1, r2, allocator, **kwargs):
    """
    Runs a back test (simulation) of allocating between a two sets of returns
    r1 and r2 are T x N DataFrames or returns where T is the time step index and N is the number of scenarios.
    allocator is a function that takes two sets of returns and allocator specific parameters, and produces
    an allocation to the first portfolio (the rest of the money is invested in the GHP) as a T x 1 DataFrame
    Returns a T x N DataFrame of the resulting N portfolio scenarios
    """
    if not r1.shape == r2.shape:
        raise ValueError("r1 and r2 need to be of same shape")
    weights = allocator(r1, r2, **kwargs)
    if not weights.shape == r1.shape:
        raise  ValueError("Allocator returned weights that don't match r1")
    r_mix = weights*r1 + (1-weights)*r2
    return r_mix

def glidepath_allocator (r1, r2, start_glide=1, end_glide=0):
    """
    Simulates a Target date Fund Style gradual move from r1 to r2
    """
    n_points = r1.shape[0]
    n_col = r1.shape[1]
    path = pd.Series(data=np.linspace(start_glide, end_glide, num=n_points))
    paths = pd.concat([path]*n_col, axis=1)
    paths.index= r1.index
    paths.columns = r1.columns
    return paths

def fixedmix_allocator(r1, r2, w1, **kwargs):
    """
    Produces a time series over T steps of allocations between the PSP and GHP across N scenarios
    PSP and GHP are T x N DataFrames that represent the returns of the PSP and GHP such that:
     each column is a scenario
     each row is the price for a timestep
    Returns an T x N DataFrame of PSP Weights
    """
    return pd.DataFrame(data=w1, index=r1.index, columns=r1.columns)

def floor_allocator(psp_r, ghp_r, floor, zc_prices, m=3):
    """
    Allocate between PSP and GHP with the goal to provide exposure to the upside
    of the PSP without going violating the floor.
    Uses a CPPI-style dynamic risk budgeting algorithm by investing a multiple
    of the cushion in the PSP
    Returns a DataFrame with the same shape as the psp/ghp representing the weights in the PSP
    """
    if zc_prices.shape != psp_r.shape:
        raise ValueError("PSP and ZC Prices must have the same shape")
    n_steps, n_scenarios = psp_r.shape
    account_value = np.repeat(1, n_scenarios)
    floor_value = np.repeat(1, n_scenarios)
    w_history = pd.DataFrame(index=psp_r.index, columns=psp_r.columns)
    for step in range(n_steps):
        floor_value = floor*zc_prices.iloc[step] ## PV of Floor assuming today's rates and flat YC
        cushion = (account_value - floor_value)/account_value
        psp_w = (m*cushion).clip(0, 1) # same as applying min and max
        ghp_w = 1-psp_w
        psp_alloc = account_value*psp_w
        ghp_alloc = account_value*ghp_w
        # recompute the new account value at the end of this step
        account_value = psp_alloc*(1+psp_r.iloc[step]) + ghp_alloc*(1+ghp_r.iloc[step])
        w_history.iloc[step] = psp_w
    return w_history

def drawdown_allocator(psp_r, ghp_r, maxdd, m=3):
    """
    Allocate between PSP and GHP with the goal to provide exposure to the upside
    of the PSP without going violating the floor.
    Uses a CPPI-style dynamic risk budgeting algorithm by investing a multiple
    of the cushion in the PSP
    Returns a DataFrame with the same shape as the psp/ghp representing the weights in the PSP
    """
    n_steps, n_scenarios = psp_r.shape
    account_value = np.repeat(1, n_scenarios)
    floor_value = np.repeat(1, n_scenarios)
    ### For MaxDD
    peak_value = np.repeat(1, n_scenarios)
    w_history = pd.DataFrame(index=psp_r.index, columns=psp_r.columns)
    for step in range(n_steps):
        ### For MaxDD
        floor_value = (1-maxdd)*peak_value ### Floor is based on Prev Peak
        cushion = (account_value - floor_value)/account_value
        psp_w = (m*cushion).clip(0, 1) # same as applying min and max
        ghp_w = 1-psp_w
        psp_alloc = account_value*psp_w
        ghp_alloc = account_value*ghp_w
        # recompute the new account value at the end of this step
        account_value = psp_alloc*(1+psp_r.iloc[step]) + ghp_alloc*(1+ghp_r.iloc[step])
        ### For MaxDD
        peak_value = np.maximum(peak_value, account_value) ### For MaxDD
        w_history.iloc[step] = psp_w
    return w_history

def glidem_floor_allocator(psp_r, ghp_r, floor, zc_prices, m_start=3, m_end=1):
    """
    Allocate between PSP and GHP with the goal to provide exposure to the upside
    of the PSP without going violating the floor.
    Uses a CPPI-style dynamic risk budgeting algorithm by investing a multiple
    of the cushion in the PSP
    Returns a DataFrame with the same shape as the psp/ghp representing the weights in the PSP
    """
    if zc_prices.shape != psp_r.shape:
        raise ValueError("PSP and ZC Prices must have the same shape")
    n_steps, n_scenarios = psp_r.shape
    account_value = np.repeat(1, n_scenarios)
    floor_value = np.repeat(1, n_scenarios)
    w_history = pd.DataFrame(index=psp_r.index, columns=psp_r.columns)
    m = glidepath_allocator(psp_r, ghp_r, m_start, m_end)
    for step in range(n_steps):
        floor_value = floor*zc_prices.iloc[step] ## PV of Floor assuming today's rates and flat YC
        cushion = (account_value - floor_value)/account_value
        psp_w = (m.iloc[step]*cushion).clip(0, 1) # same as applying min and max
        ghp_w = 1-psp_w
        psp_alloc = account_value*psp_w
        ghp_alloc = account_value*ghp_w
        # recompute the new account value at the end of this step
        account_value = psp_alloc*(1+psp_r.iloc[step]) + ghp_alloc*(1+ghp_r.iloc[step])
        w_history.iloc[step] = psp_w
    return w_history

def glidem_drawdown_allocator(psp_r, ghp_r, maxdd, m_start = 3, m_end = 1):
    """
    Allocate between PSP and GHP with the goal to provide exposure to the upside
    of the PSP without going violating the floor.
    Uses a CPPI-style dynamic risk budgeting algorithm by investing a multiple
    of the cushion in the PSP
    Returns a DataFrame with the same shape as the psp/ghp representing the weights in the PSP
    """
    n_steps, n_scenarios = psp_r.shape
    account_value = np.repeat(1, n_scenarios)
    floor_value = np.repeat(1, n_scenarios)
    ### For MaxDD
    peak_value = np.repeat(1, n_scenarios)
    w_history = pd.DataFrame(index=psp_r.index, columns=psp_r.columns)
    m = glidepath_allocator(psp_r, ghp_r, m_start, m_end)
    for step in range(n_steps):
        ### For MaxDD
        single_crash = psp_r.iloc[step].mean()<=-0.30
        double_crash = step>0 and psp_r.iloc[step-1].mean()<-0.15 and psp_r.iloc[step].mean() <= -0.15
        #if single_crash or double_crash :
        #    m.iloc[step]=np.maximum(1.0, m.iloc[step]-4.0)
        floor_value = (1-maxdd)*peak_value ### Floor is based on Prev Peak
        cushion = (account_value - floor_value)/account_value
        psp_w = (m.iloc[step]*cushion).clip(0, 1) # same as applying min and max
        if single_crash or double_crash :
            psp_w = psp_w * 0.05
        ghp_w = 1-psp_w
        psp_alloc = account_value*psp_w
        ghp_alloc = account_value*ghp_w
        # recompute the new account value at the end of this step
        account_value = psp_alloc*(1+psp_r.iloc[step]) + ghp_alloc*(1+ghp_r.iloc[step])
        ### For MaxDD
        peak_value = np.maximum(peak_value, account_value) ### For MaxDD
        w_history.iloc[step] = psp_w
    return w_history

def ldi_funding_ratio(asset_returns, liabilities, zc_prices, a_0, rolling = False):
    """
    Takes the asset returns, liabilities and computes the funding ratio, returns the mean/rolling window of 1 year
    """
    assets = a_0 * (1+ asset_returns).cumprod()
    fr = pd.DataFrame(index=liabilities.index, columns=liabilities.columns, dtype=float)
    for t in range(len(assets)):
        sub = liabilities.iloc[t]
        fr.iloc[t] = (assets.iloc[t] - sub).divide((liabilities[t-1:]*zc_prices[t-1:]).sum(axis=0))
    ret = fr_mean=fr.mean(axis=1)
    if rolling: 
        ret = fr_mean.rolling(window = 12).mean()
    return ret
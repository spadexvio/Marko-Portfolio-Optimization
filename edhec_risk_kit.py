import pandas as pd
import scipy.stats
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

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
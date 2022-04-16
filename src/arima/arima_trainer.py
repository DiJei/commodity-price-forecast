import time
import warnings
import itertools
import numpy as np
import pandas as pd
import pmdarima as pm
from statsmodels.tsa.stattools import acf, pacf

def sqdist(vector):
    return sum(x*x for x in vector)

def hyper_param_sarima_trainer(data, p_values = [0,1], q_values = [0,1], p_s_values = [0,1], q_s_values = [0,1], m = 12):
    """Hyperparam search for sarimax
    Args:
        data (Dataframe): Dataframe with only one column
        p_values (list):  init and last value for p param
        q_values (list):  init and last value for q param
        p_s_values (list):  init and last value for P param
        q_s_values (list):  init and last value for Q param
    Returns:
        best_model (model_fit): Best model fit
        log_df (Dataframe): log of all trains results
    """
    ## ORDER (p,d,q)
    hyper_param_order = []
    base_q = []
    base_p = []
    for n in range(p_values[0], p_values[1] + 1):
        base_q.append(n)
    for n in range(q_values[0], q_values[1] + 1):
        base_p.append(n)    
    for q,p in itertools.product(base_q,base_p):
        hyper_param_order.append([q, p])
    # Sorting
    hyper_param_order.sort(key=sqdist)
    
    ## SEASONAL ORDER (p,d,q)
    hyper_param_order_s = []
    base_q = []
    base_p = []
    for n in range(p_s_values[0], p_s_values[1] + 1):
        base_q.append(n)
    for n in range(q_s_values[0], q_s_values[1] + 1):
        base_p.append(n)    
    for q,p in itertools.product(base_q,base_p):
        hyper_param_order_s.append([q, p])
    # Sorting
    hyper_param_order_s.sort(key=sqdist)
    
    # Estimate d, D
    d = pm.arima.ndiffs(data)
    D = pm.arima.nsdiffs(data, m = m, max_D = 3, test='ocsb')
    
    log_df = pd.DataFrame()
    best_model = None
    flag = True
    count = 1
    ant = -1
    max_train = len(hyper_param_order)*len(hyper_param_order_s)
    for order in hyper_param_order:
        for order_s in hyper_param_order_s:
            row = {}
            start_time = time.time()
            try:
                model = pm.arima.ARIMA((order[0],d,order[1]), seasonal_order=(order_s[0], D, order_s[1], m), method='lbfgs', maxiter = 80, 
                                   suppress_warnings=True, scoring='mse')
                res = model.fit(data)
            except:
                print('fit error')
                row['order(p,d,q)'] = str((order[0],d,order[1]))
                row['seasonal(p,d,q)'] = str((order_s[0], D, order_s[1], m))
                row['AIC'] = -1
                row['BIC'] = -1
                row['train_time'] = np.round(time.time() - start_time, 2)
                log_df = log_df.append(row, ignore_index = True)
                continue
                
            row['order(p,d,q)'] = str((order[0],d,order[1]))
            row['seasonal(p,d,q)'] = str((order_s[0], D, order_s[1], m))
            row['AIC'] = model.aic()
            row['BIC'] = model.aic()
            row['train_time'] = np.round(time.time() - start_time, 2)
            log_df = log_df.append(row, ignore_index = True)
            if flag:
                best_model = model
                flag = False
                print(row)
            else:
                if best_model.aic() > model.aic():
                    best_model = model
                    print(row)
        count +=1
        per = int((count/max_train)*100)
        if per % 5 == 0 and per != ant:
            print(str(per) +"%")
            ant = per
    return best_model, log_df

def hyper_param_arimax_trainer(data, p_values = [0,5], q_values = [0,5], exo_data = pd.DataFrame()):
    """Hyperparam search for arimax
    Args:
        data (Dataframe): Dataframe with only one column
        p_values (list):  init and last value for p param
        q_values (list):  init and last value for q param
        exo_data (Dataframe):   Dataframe with exogenous vrabile
    Returns:
        best_model (model_fit): Best model fit
        log_df (Dataframe): log of all trains results
    """
    count = 1
    
    ## ORDER
    
    # Estimate d
    d = pm.arima.ndiffs(data, max_d = 4, test = 'adf')
    
    # Select range of p and q
    hyper_param_order = []
    base_q = select_good_values_q(data.diff(d).dropna(), q_values[-1])
    base_p = select_good_values_p(data.diff(d).dropna(), p_values[-1])    
    for q,p in itertools.product(base_q,base_p):
        hyper_param_order.append([q, p])
    # Sorting
    hyper_param_order.sort(key=sqdist)
    
    max_train  = len(hyper_param_order)

    ant = -1
    log_df = pd.DataFrame()
    best_model = None
    flag = True
    for order in hyper_param_order:
        row = {}
        start_time = time.time()
        try:
            model = pm.arima.ARIMA((order[0],d,order[1]), method='lbfgs', maxiter = 80, 
                                   suppress_warnings=True, scoring='mse')
            if len(exo_data) == 0:
                res = model.fit(data)
            else:
                res = model.fit(data, exo_data)
        except:
            print('fit error')
            row['order(p,d,q)'] = str((order[0],d,order[1]))
            row['AIC'] = -1
            row['BIC'] = -1
            row['train_time'] = np.round(time.time() - start_time, 2)
            log_df = log_df.append(row, ignore_index = True)
            continue
        row['order(p,d,q)'] = str((order[0],d,order[1]))
        row['AIC'] = model.aic()
        row['BIC'] = model.bic()
        row['train_time'] = np.round(time.time() - start_time, 2)
        log_df = log_df.append(row, ignore_index = True)
        if flag:
            best_model = model
            flag = False
            print(row)
        else:
            if best_model.aic() > model.aic():
                best_model = model
                print(row)
        count +=1
        per = int((count/max_train)*100)
        if per % 5 == 0 and per != ant:
            print(str(per) +"%")
            ant = per
    return best_model, log_df

def select_good_values_q(data, maxlags):
    """Calculate Auto Correlation Function values and filter ones
    outside aplha value
    Args:
        data (Array): Timeseries data, must be stationary
        maxlags (int):  max value of q
    Returns:
        good_values (list): list of good estimation of q param
    """
    value, inter = acf(data, nlags = maxlags, alpha = 0.05, fft=True)
    
    good_values = []
    for x in range(0, len(value)):
        if np.abs(value[x]) >= inter[x].std():
            good_values.append(x)    
    return good_values

def select_good_values_p(data, maxlags):
    """Calculate Partial Auto Correlation Function values and filter ones
    outside aplha value
    Args:
        data (Array): Timeseries data, must be stationary
        maxlags (int):  max value of p
    Returns:
        good_values (list): list of good estimation of p param
    """
    value, inter = pacf(data, nlags = maxlags, alpha = 0.05)
    good_values = []
    for x in range(0, len(value)):
        if np.abs(value[x]) >= inter[x].std():
            good_values.append(x)    
    return good_values


def fill_values_missing_dates(base_df, base_cols, target_df, target_cols):
    """
    """
    
    # get dates not in traget_df
    new_date = base_df[~base_df.index.isin(target_df.index)].sort_index().copy()
    new_date[base_cols] = np.nan
    
    for x in range(0, len(base_cols)):
        new_date.rename(columns = {base_cols[x]: target_cols[x]})
    
    new_target_df = target_df[target_cols].append(new_date)
    new_target_df = new_target_df.sort_index()
    new_target_df = new_target_df[target_cols].interpolate(method='linear', limit_direction='forward', axis=0)
    
    return new_target_df
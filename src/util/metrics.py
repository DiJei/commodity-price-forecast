import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error


def mean_forecast_bias_error(df_original, df_predict, cols = ['Open', 'High', 'Low', 'Close']):
    """Measuare Mean forecast error or Forecast Bias, df_a, df_b must have the same lenght and dates
    Args:
        df_a (Dataframe): Dataframe with daily values of prices
        df_b (Dataframe): Dataframe with daily values of prices
        cols (list): Name of column with feature names 
    Returns:
        result  (dict): Key/Value with the accuracy for each col (in the same order of cols param)
    """
    result = {}

    for col in cols:
        forecast_errors = [df_original[col].values[i]-df_predict[col].values[i] for i in range(len(df_original))]
        bias = sum(forecast_errors) * 1.0/len(df_original)
        result[col] = bias    
    return result


def value_mape(df_a, df_b, cols = ['Open', 'High', 'Low', 'Close']):
    """Measuare Mean Square Error of prices, df_a, df_b must have the same lenght and dates
    Args:
        df_a (Dataframe): Dataframe with daily values of prices
        df_b (Dataframe): Dataframe with daily values of prices
        cols (list): Name of column with feature names 
    Returns:
        result  (dict): Key/Value with the accuracy for each col (in the same order of cols param)
    """
    result = {}
    
    for col in cols:
        result[col] = np.round(mean_absolute_percentage_error(df_a[col].values , df_b[col].values), 4)    
    return result

def per_change_mape(df_a, df_b, cols = ['Open', 'High', 'Low', 'Close']):
    """Measuare Mean Square Error of % of changes , df_a, df_b must have the same lenght and dates
    Args:
        df_a (Dataframe): Dataframe with daily values of prices
        df_b (Dataframe): Dataframe with daily values of prices
        cols (list): Name of column with feature names 
    Returns:
        result  (dict): Key/Value with the accuracy for each col (in the same order of cols param)
    """
    diff_pirecs_a = df_a[cols].diff()[1:]
    diff_pirecs_a = diff_pirecs_a/ df_a[cols][1:] * 100

    diff_pirecs_b = df_b[cols].diff()[1:]
    diff_pirecs_b = diff_pirecs_b/ df_b[cols][1:] * 100

    result = {}
    for col in cols:
        result[col] = np.round(mean_absolute_percentage_error(diff_pirecs_a[col].values , diff_pirecs_b[col].values), 2)
    return result

def acc_weekly_trend_direction(df_a, df_b,  cols = ['Open', 'High', 'Low', 'Close']):
    """Measuare % of weeks the model predict the right trend (upper, down),
    df_a, df_b must have the same lenght and dates
    Args:
        df_a (Dataframe): Dataframe with daily values of prices
        df_b (Dataframe): Dataframe with daily values of prices
        cols (list): Name of column with feature names 
    Returns:
        result  (dict): Key/Value with the accuracy for each col (in the same order of cols param)
    """
    df_a['year'] = df_a['Date'].dt.year
    df_a['week'] = df_a['Date'].dt.week
    df_b['year'] = df_b['Date'].dt.year
    df_b['week'] = df_b['Date'].dt.week
    
    # Build week dicts
    week_dict = {}
    for year in df_a['year'].unique():
        week_dict[year] = df_a.query('year == {}'.format(year))['week'].unique()
    
    # Check values
    res_df_a = pd.DataFrame()
    res_df_b = pd.DataFrame()
    for year in week_dict:
        for week in week_dict[year]:
            temp_a = df_a.query("year == {} and week == {}".format(year,week))[cols + ['Date']]
            temp_a_mean = temp_a[cols].mean().values
            temp_a_day = temp_a[cols][:1].values
            res_cmp = pd.DataFrame(temp_a_mean > temp_a_day, columns = cols)
            res_df_a = pd.concat([res_df_a, res_cmp],ignore_index = True)
            
            temp_b = df_b.query("year == {} and week == {}".format(year,week))[cols + ['Date']]
            temp_b_mean = temp_b[cols].mean().values
            temp_b_day = temp_b[cols][:1].values
            res_cmp = pd.DataFrame(temp_b_mean > temp_b_day, columns = cols)
            res_df_b = pd.concat([res_df_b, res_cmp],ignore_index = True)
    result = {}
    for col in cols:
        result[col] = np.round((np.sum(res_df_a[col].values == res_df_b[col].values)/len(res_df_a))*100, 2)
    return result

def acc_monthly_trend_direction(df_a, df_b,  cols = ['Open', 'High', 'Low', 'Close']):
    """Measuare % of months the model predict the right trend (upper, down),
    df_a, df_b must have the same lenght and dates
    Args:
        df_a (Dataframe): Dataframe with daily values of prices
        df_b (Dataframe): Dataframe with daily values of prices
        cols (list): Name of column with feature names 
    Returns:
        result  (dict): Key/Value with the accuracy for each col (in the same order of cols param)
    """
    df_a['year'] = df_a['Date'].dt.year
    df_a['month'] = df_a['Date'].dt.month
    df_b['year'] = df_b['Date'].dt.year
    df_b['month'] = df_b['Date'].dt.month
    
    # Build week dicts
    month_dict = {}
    for year in df_a['year'].unique():
        month_dict[year] = df_a.query('year == {}'.format(year))['month'].unique()
    
    # Check values
    res_df_a = pd.DataFrame()
    res_df_b = pd.DataFrame()
    for year in month_dict:
        for week in month_dict[year]:
            temp_a = df_a.query("year == {} and month == {}".format(year,week))[cols + ['Date']]
            temp_a_mean = temp_a[cols].mean().values
            temp_a_day = temp_a[cols][:1].values
            res_cmp = pd.DataFrame(temp_a_mean > temp_a_day, columns = cols)
            res_df_a = pd.concat([res_df_a, res_cmp],ignore_index = True)
            
            temp_b = df_b.query("year == {} and month == {}".format(year,week))[cols + ['Date']]
            temp_b_mean = temp_b[cols].mean().values
            temp_b_day = temp_b[cols][:1].values
            res_cmp = pd.DataFrame(temp_b_mean > temp_b_day, columns = cols)
            res_df_b = pd.concat([res_df_b, res_cmp],ignore_index = True)
    result = {}
    for col in cols:
        result[col] = np.round((np.sum(res_df_a[col].values == res_df_b[col].values)/len(res_df_a))*100, 2)
    return result

def acc_daily_trend_direction(df_a, df_b,  cols = ['Open', 'High', 'Low', 'Close']):
    """Measuare % of days the model predict the right trend (upper, down),
    df_a, df_b must have the same lenght and dates
    Args:
        df_a (Dataframe): Dataframe with daily values of prices
        df_b (Dataframe): Dataframe with daily values of prices
        cols (list): Name of column with feature names 
    Returns:
        result  (dict): Key/Value with the accuracy for each col (in the same order of cols param)
    """
    temp_a = df_a[cols].diff().dropna()
    temp_a = temp_a.reset_index()
    temp_a = temp_a[cols] > 0
    temp_a = temp_a.astype(int)
    
    temp_b = df_b[cols].diff().dropna()
    temp_b = temp_b.reset_index()
    temp_b = temp_b[cols] > 0
    temp_b = temp_b.astype(int)
    
    result = {}
    for col in cols:
        result[col] = np.round((np.sum(temp_a[col].values == temp_b[col].values)/len(temp_a))*100, 2)
    return result


def acc_dir_trend(df_a, col = 'price_sac', dif = 1):
    """
    """
    
    temp_a =  df_a.sort_index(ascending = False).copy()
    temp_a = temp_a.diff(dif).dropna()
    
    temp_a = temp_a.sort_index(ascending = True)
    temp_a['trend'] = (temp_a[col] > 0).astype(int)
    
    temp_a['trend'].replace({1: -1}, inplace=True)
    temp_a['trend'].replace({0: 1}, inplace=True)
    temp_a['price_sac'] *= -1

    return temp_a


def acc_nday_trend_direction(df_a, df_b,  col = ['price_sac'], dif = 1):
    """Measuare % of days the model predict the right trend (upper, down),
    df_a, df_b must have the same lenght and dates
    Args:
        df_a (Dataframe): Dataframe with daily values of prices
        df_b (Dataframe): Dataframe with daily values of prices
        cols (list): Name of column with feature names 
    Returns:
        result  (dict): Key/Value with the accuracy for each col (in the same order of cols param)
    """
    temp_a =  acc_dir_trend(df_a, col,  dif)
    temp_b =  acc_dir_trend(df_b, col,  dif)
    
    result = np.round(np.sum(temp_a['trend'] == temp_b['trend'])/len(temp_b['trend']), 4)
    
    return result
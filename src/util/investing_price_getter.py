import pandas as pd
import investpy
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class PriceLoader():

    def __init__(self, commodity = 'US Soybeans', country = 'United States'):
        self.commodity = commodity
        self.country = country
    
    def get_price_from_until(self, start, end):
        try:
            res = investpy.get_commodity_historical_data(commodity = self.commodity, 
                                                         country   = self.country, 
                                                         from_date = start,
                                                         to_date   = end)
            res = res.reset_index()
        except:
            res = pd.DataFrame()
        
        return res
    
    
def data_to_torch(df_prices, lookback, lookforward):   
    # remove zero values
    for col in ['Open','High','Low','Close']:
        for inde in df_prices[df_prices[col] == 0].index.values:
            df_prices.loc[inde, col] = (df_prices.loc[inde - 1, col] + df_prices.loc[inde + 1, col])/2
    
    
    scaler = MinMaxScaler(feature_range=(-1, 1))
    price = pd.DataFrame(scaler.fit_transform(df_prices[['Open','High','Low','Close']]))
    
    data_raw = price.to_numpy()
    data_x = []
    data_y = []
    
    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - lookback - lookforward + 1): 
        data_x.append(data_raw[index: index + lookback])
        data_y.append(data_raw[index + lookback: index + lookback + lookforward])
    
    data_x = np.array(data_x);
    data_y = np.array(data_y);
    
    ## Convert to tensor
    data_x = torch.from_numpy(data_x).type(torch.Tensor)
    data_y = torch.from_numpy(data_y).type(torch.Tensor)
    
    ## only for one day
    if lookforward == 1:
        y_train_lstm = torch.unsqueeze(y_train_lstm, dim = 1)
        y_test_lstm  = torch.unsqueeze(y_test_lstm, dim = 1)
    
    return data_x, data_y, scaler


def transform_to_seq(df_prices, lookback, lookforward):
    data_raw =  df_prices[['Open','High','Low','Close']].to_numpy()
    
    df_seq = pd.DataFrame()
    # create all possible sequences of length seq_len
    
    for index in range(len(data_raw) - lookback - lookforward + 1):
        temp = pd.DataFrame(data_raw[index: index + lookback + lookforward].transpose())
        temp['seq'] = ['Open','High','Low','Close']
        date = str(df_prices.loc[index, ['Date']].dt.date.values[0])
        temp['date'] = [date,date,date,date]
        df_seq = pd.concat([df_seq, temp])
    
    return df_seq

def seq_to_torch(df_seq, lookback, lookforward):
    temp = np.array(df_seq.values[:,0:lookback + lookforward]).astype('float16')
    # to torch
    data_x = torch.from_numpy(temp[:,0:lookback]).type(torch.Tensor)
    data_y = torch.from_numpy(temp[:,lookback:lookback + lookforward]).type(torch.Tensor)
    
    
    data_x = torch.reshape(data_x, (int(len(data_x)/4),4,data_x.shape[1])).transpose(1,2)
    data_y = torch.reshape(data_y, (int(len(data_y)/4),4,data_y.shape[1])).transpose(1,2)
    
    return data_x, data_y
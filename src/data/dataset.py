import torch
import pandas as pd
import numpy as np
import logging
from torch.utils.data import Dataset, DataLoader


class PriceDataset(Dataset):
    def __init__(self, df, lookback = 20, lookforward = 5, input_dim = 4, price_cols = ['Open','High','Low','Close'], date_col = 'Date', date_format = '%Y-%m-%d'):
        
        self.lookback = lookback
        self.lookforward = lookforward
        self.input_dim = input_dim
        
        df[date_col] = pd.to_datetime(df[date_col] , format = date_format)
        df_seq = self.transform_to_seq(df, price_cols, date_col)
        
        self.ids = df[0:len(df) - self.lookback - self.lookforward][date_col].astype(str)
        self.data_x, self.data_y = self.seq_to_torch(df_seq)
        logging.info(f'Creating dataset with {len(self.ids)} examples')
        
    def __len__(self):
        return len(self.ids)
        
    def __getitem__(self, idx):
        sample = {"date": self.ids[idx], "x": self.data_x[idx], "y":self.data_y[idx]}
        return sample

    def transform_to_seq(self, df_prices, price_cols, date_col):
        data_raw =  df_prices[price_cols].to_numpy()
    
        df_seq = pd.DataFrame()
        # create all possible sequences of length seq_len
        
        for index in range(len(data_raw) - self.lookback - self.lookforward + 1):
            temp = pd.DataFrame(data_raw[index: index + self.lookback + self.lookforward].transpose())
            df_seq = pd.concat([df_seq, temp])
        
        return df_seq

    def seq_to_torch(self, df_seq):
        temp = np.array(df_seq.values[:,0:self.lookback + self.lookforward]).astype('float32')
        # to torch
        data_x = torch.from_numpy(temp[:,0:self.lookback]).type(torch.Tensor)
        data_y = torch.from_numpy(temp[:,self.lookback:self.lookback + self.lookforward]).type(torch.Tensor)
        
        data_x = torch.reshape(data_x, (int(len(data_x)/self.input_dim),self.input_dim,data_x.shape[1])).transpose(1,2)
        data_y = torch.reshape(data_y, (int(len(data_y)/self.input_dim),self.input_dim,data_y.shape[1])).transpose(1,2)
        
        return data_x, data_y

def build_technical_feature(df, price_cols = ['Open','High','Low','Close']):
    
    df_temp = df.copy()
    # MOVING AVERAGE
    df_temp['MV5'] = df_temp[price_cols[-1]].rolling(window=5).mean()
    df_temp['MV10'] = df_temp[price_cols[-1]].rolling(window=10).mean()
    
    # Momentum
    df_temp['MOM6'] = df_temp[price_cols[-1]].diff(6)
    df_temp['MOM12'] = df_temp[price_cols[-1]].diff(12)
    
    # Moving Exponential Mean
    df_temp['EWM20'] = df_temp[price_cols[-1]].ewm(span=20, adjust = False).mean()
    
    # Average True Range
    df_atr = pd.DataFrame()
    df_atr['H-L'] = df_temp[price_cols[1]] - df_temp[price_cols[2]]
    df_atr['H-C'] = np.abs(df_temp[price_cols[1]] - df_temp[price_cols[-1]])
    df_atr['L-C'] = np.abs(df_temp[price_cols[2]] - df_temp[price_cols[-1]])
    df_temp['ATR14'] = df_atr.max(axis=1).rolling(window = 14).mean()
    
    # Moving Average Convergence Divergence
    df_temp['MACD'] = df_temp[price_cols[-1]].ewm(span=12, adjust = False).mean() - df_temp[price_cols[-1]].ewm(span=26, adjust = False).mean()
    
    # Price Rate of Change
    df_temp['PRC7'] =(df_temp[price_cols[-1]].diff(7) / df_temp[price_cols[-1]].tail(len(df_temp) - 7))*100
    
    df_temp = df_temp.dropna()
    
    return df_temp

def price_to_per(df, cols = ['Open', 'Low', 'High', 'Close'], date_col = 'Date'):
    
    diff_pirecs = df[cols].diff()[1:]
    diff_pirecs= diff_pirecs/ df[cols][1:] * 100
    diff_pirecs[date_col] =  df[date_col][1:].values
    diff_pirecs = diff_pirecs.reset_index()
    
    return  diff_pirecs


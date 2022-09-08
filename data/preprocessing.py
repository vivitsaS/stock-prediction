import pandas as pd
from pandas import read_csv
import yfinance as yf
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import math

# this should:
# 1. take input - (df, predict_col_index/predict_col_name, window)
# 2. give output in (op_tensor_X, op_tensor_Y)
# 3. between 1 and 2, it should extract features and perform required transformations

class CSVDataset(Dataset):

    def __init__(self):

        # sequence window length
        self.window = 56
        
    # gets df from path
    def get_df_from_path(self, path):
    
        self.df = read_csv(path)
        
        return self.df
    # gets df from yahoo finance API
    def get_df_from_yfin(self, ticker_name, start_date, end_end_date):
        
        self.df = yf.download(tickers = ticker_name, start=start_date,  end=end_end_date)
        
        return self.df
        
    # adds technical indicators
    def ti(self,df):

        # Simple moving average (SMA)
        df['SMA_w1'] = df['Adj Close'].rolling(7).mean()
        df['SMA_w2'] = df['Adj Close'].rolling(14).mean()
        df['SMA_ratio'] = df['SMA_w1'] / df['SMA_w2']

        # SMA-V(voulume)
        df['SMA_w1_Volume'] = df['Volume'].transform(lambda x: x.rolling(window=7).mean())
        df['SMA_w2_Volume'] = df['Volume'].transform(lambda x: x.rolling(window=14).mean())
        df['SMA_Volume_Ratio'] = df['SMA_w1_Volume'] / df['SMA_w2_Volume']
        
        #EWMA (Exponentially-weighted Moving Average )
        df['EMA_w1'] = df['Adj Close'].ewm(span = 7, min_periods = 6).mean()
        df['EMA_w2'] = df['Adj Close'].ewm(span = 14, min_periods = 13).mean()
        df['EMA_ratio'] = df['EMA_w1'] / df['EMA_w2']
        
        # Stochastic Osclillator
        df['Lowest_5D'] = df['Low'].transform(lambda x: x.rolling(window=7).min())
        df['High_5D'] = df['High'].transform(lambda x: x.rolling(window=14).max())
        df['Lowest_15D'] = df['Low'].transform(lambda x: x.rolling(window=14).min())
        df['High_15D'] = df['High'].transform(lambda x: x.rolling(window=14).max())

        df['Stochastic_5'] = ((df['Close'] - df['Lowest_5D']) / (
                df['High_5D'] - df['Lowest_5D'])) * 100
        df['Stochastic_15'] = ((df['Close'] - df['Lowest_15D']) / (
                df['High_15D'] - df['Lowest_15D'])) * 100

        df['Stochastic_%D_5'] = df['Stochastic_5'].rolling(window=7).mean()
        df['Stochastic_%D_15'] = df['Stochastic_5'].rolling(window=14).mean()

        df['Stochastic_Ratio'] = df['Stochastic_%D_5'] / df['Stochastic_%D_15']

        # Moving Average Convergence Divergence (MACD)
        df['7Ewm'] = df['Close'].transform(lambda x: x.ewm(span=7, adjust=False).mean())
        df['14Ewm'] = df['Close'].transform(lambda x: x.ewm(span=14, adjust=False).mean())
        df['MACD'] = df['14Ewm'] - df['7Ewm']

        df = df[["Adj Close", "Volume", "SMA_ratio", "EMA_ratio", "SMA_Volume_Ratio", "Stochastic_Ratio","MACD"]]
        df = df.replace(np.nan, 0)

        return df

    # X,y split
    def Xy(self, df):

        X, y = [], []
        X_true, y_true = [], []
        for i in range(len(df) - (2 * self.window) - 1):
            # all rows specified (in window period) for all columns
            a_true = df.iloc[(i-1 + self.window):(i-1 + (2 * self.window)), :]
            scaler = MinMaxScaler(feature_range=(-1, 1))
            a_scaled = scaler.fit_transform(a_true)
            X.append(a_scaled)
            X_true.append(a_true)
            b_true = df.iloc[(i+ self.window):(i + (2 * self.window)), 0]
            b_true = b_true.values.reshape(-1, 1)
            b_scaled = scaler.fit_transform(b_true)
            b_scaled = b_scaled[-1]
            b_true = b_true[-1]
            # b = a['Adj Close'][i + (2*self.window) + 1]
            # the window + 1th row for Adj Close column, which is basically one value
            y.append(b_scaled)
            y_true.append(b_true)
            # b should contain adj close value of the window+1th day, scaeld wrt to the adj
            # close values of set a for window days

        X = np.array(X)
        y = np.array(y)
        X_true = np.array(X_true)
        y_true = np.array(y_true)

        # reshape
        # shape should be = (rows/window)*(timestep)*(features=cols)
        d_feat = len(df.columns)
        X_x = X.shape[0]
        y_x = y.shape[0]
        X = X.reshape(X_x, self.window, d_feat)
        y = y.reshape(y_x, 1, 1)
        X_true = X_true.reshape(X_x, self.window, d_feat)
        y_true = y_true.reshape(y_x, 1, 1)

        return X, y, X_true, y_true

    def get_X_pred(self, df):

        X = []
        X_true = []
        for i in range(len(df) - (2 * self.window) - 1):
            # all rows specified (in window period) for all columns
            a_true = df.iloc[(i + self.window):(i + (2 * self.window)), :]
            scaler = MinMaxScaler(feature_range=(-1, 1))
            a_scaled = scaler.fit_transform(a_true)
            X.append(a_scaled)
            X_true.append(a_true)
            # b should contain adj close value of the window+1th day, scaeld wrt to the adj
            # close values of set a for window days

        X = np.array(X)
        X_true = np.array(X_true)

        # reshape
        # shape should be = (rows/window)*(time step)*(features=cols)
        d_feat = len(df.columns)
        X_x = X.shape[0]
        X = X.reshape(X_x, self.window, d_feat)
        X_true = X_true.reshape(X_x, self.window, d_feat)

        return X, X_true

    # splits data into train, test, val dfs
    def get_split(self, df, train_pct, val_pct, test_pct):

        total_size = len(df)
        train = df.iloc[:(math.ceil(train_pct * total_size)), :]
        val = df.iloc[(math.ceil(train_pct * total_size)) + 1:(math.ceil(train_pct * total_size)) + (
            math.ceil(val_pct * total_size)), :]
        test = df.iloc[(math.ceil(val_pct * total_size)) + 1:(math.ceil(val_pct * total_size)) + (
            math.ceil(test_pct * total_size)), :]

        return train, val, test

from pandas import read_csv
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import math


class CSVDataset(Dataset):

    def __init__(self, path):

        # loads the csv file as a dataframe, take columns: 5(Adj. close) and 6(Volume).
        self.df = read_csv(path, usecols=[2, 3, 4, 5, 6])

        # lstm window
        self.window = 30
        """# rolling volume
        self.df["Volume"] = self.rolling_zscore(self.df, "Volume")
        # pct change in adj close price
        self.df["Adj Close"] = self.df["Adj Close"].pct_change()"""

    # calculates rolling z score
    def rolling_zscore(self, df, column):

        x = df[column]
        r = x.rolling(window=self.window)
        m = r.mean().shift(1)
        s = r.std(ddof=0).shift(1)
        z = (x - m) / s

        return z

    # wilder smoothening for TIs
    def Wilder(self, data, periods):
        start = np.where(~np.isnan(data))[0][0]  # Check if nans present in beginning
        Wilder = np.array([np.nan] * len(data))
        Wilder[start + periods - 1] = data[start:(start + periods)].mean()  # Simple Moving Average
        for i in range(start + periods, len(data)):
            Wilder[i] = (Wilder[i - 1] * (periods - 1) + data[i]) / periods  # Wilder Smoothing

        return (Wilder)

    # adds technical indicators
    def ti(self):

        # Simple moving average (SMA)
        self.df['SMA_5'] = self.df['Close'].transform(lambda x: x.rolling(window=5).mean())
        self.df['SMA_15'] = self.df['Close'].transform(lambda x: x.rolling(window=15).mean())
        self.df['SMA_ratio'] = self.df['SMA_15'] / self.df['SMA_5']

        # SMA-V(voulume)
        self.df['SMA5_Volume'] = self.df['Volume'].transform(lambda x: x.rolling(window=5).mean())
        self.df['SMA15_Volume'] = self.df['Volume'].transform(lambda x: x.rolling(window=15).mean())
        self.df['SMA_Volume_Ratio'] = self.df['SMA5_Volume'] / self.df['SMA15_Volume']

        # Stochastic Osclillator
        self.df['Lowest_5D'] = self.df['Low'].transform(lambda x: x.rolling(window=5).min())
        self.df['High_5D'] = self.df['High'].transform(lambda x: x.rolling(window=5).max())
        self.df['Lowest_15D'] = self.df['Low'].transform(lambda x: x.rolling(window=15).min())
        self.df['High_15D'] = self.df['High'].transform(lambda x: x.rolling(window=15).max())

        self.df['Stochastic_5'] = ((self.df['Close'] - self.df['Lowest_5D']) / (
                    self.df['High_5D'] - self.df['Lowest_5D'])) * 100
        self.df['Stochastic_15'] = ((self.df['Close'] - self.df['Lowest_15D']) / (
                    self.df['High_15D'] - self.df['Lowest_15D'])) * 100

        self.df['Stochastic_%D_5'] = self.df['Stochastic_5'].rolling(window=5).mean()
        self.df['Stochastic_%D_15'] = self.df['Stochastic_5'].rolling(window=15).mean()

        self.df['Stochastic_Ratio'] = self.df['Stochastic_%D_5'] / self.df['Stochastic_%D_15']

        # Moving Average Convergence Divergence (MACD)
        self.df['5Ewm'] = self.df['Close'].transform(lambda x: x.ewm(span=5, adjust=False).mean())
        self.df['15Ewm'] = self.df['Close'].transform(lambda x: x.ewm(span=15, adjust=False).mean())
        self.df['MACD'] = self.df['15Ewm'] - self.df['5Ewm']

        self.df = self.df[["Adj Close", "Volume", "SMA_ratio", "SMA_Volume_Ratio", "Stochastic_Ratio", "MACD"]]
        self.df = self.df.replace(np.nan, 0)

        return self.df

    # X,y split
    def Xy(self, df):

        X, y = [], []
        X_true, y_true = [], []
        for i in range(len(df) - (2 * self.window) - 1):
            # all rows specified (in window period) for all columns
            a_true = df.iloc[(i + self.window):(i + (2 * self.window)), :]
            scaler = MinMaxScaler(feature_range=(-1, 1))
            a_scaled = scaler.fit_transform(a_true)
            X.append(a_scaled)
            X_true.append(a_true)
            b_true = df.iloc[(i + self.window):(i + (2 * self.window) + 2), 0]
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

    # splits data into train, test, val dfs
    def get_split(self, df, train_pct, val_pct, test_pct):

        total_size = len(self.df)
        train = df.iloc[:(math.ceil(train_pct * total_size)), :]
        val = df.iloc[(math.ceil(train_pct * total_size)) + 1:(math.ceil(train_pct * total_size)) + (
            math.ceil(val_pct * total_size)), :]
        test = df.iloc[(math.ceil(val_pct * total_size)) + 1:(math.ceil(val_pct * total_size)) + (
            math.ceil(test_pct * total_size)), :]

        return train, val, test

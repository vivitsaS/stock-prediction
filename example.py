from intelLib.models import GAT, ALSTM, GRU, LSTM
from intelLib.data import preprocessing as pp


# collect standard stock price data
ohlcv_data = pp.CSVDataset().get_df_from_yfin('AAPL', '2010-01-01', '2022-09-01')

# add technical indicators
df = pp.CSVDataset().ti(ohlcv_data)

# train, val, test split
train, val, test = pp.CSVDataset().get_split(df, 0.6, 0.15, 0.25)

# x,y split
X_train, y_train, X_train_true, y_train_true = pp.CSVDataset().Xy(train)
X_val, y_val, X_val_true, y_val_true = pp.CSVDataset().Xy(val)
X_test, y_test, X_test_true, y_test_true = pp.CSVDataset().Xy(test)

# run

# define the model

# model parameters
d_feat = 3
hidden_size = 64
num_layers = 3
dropout = 0.2
batch_size = 64
model = ALSTM.ALSTMModel(d_feat, hidden_size, num_layers, dropout)
print("model params:{}".format(model.parameters))

# hyperparameters
loss_type = "mse"
metric_type = ""
n_epochs = 30
lr = 0.002
early_stop = 10

# fit model to dataset
print("\n fitting...")
best_param, trained_model = ALSTM.Task(X_train, y_train, X_val, y_val, X_test, y_test, model, loss_type, metric_type,
                                       n_epochs, lr, early_stop).fit()

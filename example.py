# run

from intelLib.models import GAT, ALSTM, GRU,LSTM
from intelLib.run import task

path = 'intelLib\\datasets\\netflix.csv'

# define the model
# model parameters
d_feat = 6
hidden_size = 64
num_layers = 3
dropout = 0.2
batch_size = 96
model = ALSTM.ALSTMModel(d_feat, hidden_size, num_layers, dropout)
print("model params:{}".format(model.parameters))

# hyperparameters
loss_type = "mse"
metric_type = ""
n_epochs = 300
lr = 0.002
early_stop = 20

# fit model to dataset from path
print("\n fitting...")
y_pred_train = ALSTM.Task(model, path, loss_type, metric_type, n_epochs, lr, early_stop).fit()

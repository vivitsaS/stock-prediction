# model definition
# this model should take in train.size() number of window size*d_feat arrays
from __future__ import division
from __future__ import print_function

import torch.nn as nn

import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim


class Task:

    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test, model, loss_type, metric_type, n_epochs, lr, early_stop):

        self.model = model
        
        self.X_train, self.y_train= X_train, y_train
        self.X_val, self.y_val= X_val, y_val
        self.X_test, self.y_test = X_test, y_test
        
        self.loss_type = loss_type
        self.metric_type = metric_type
        self.n_epochs = n_epochs
        self.lr = lr
        self.early_stop = early_stop
    

    def mse(self, pred, label):
        loss = ((pred - label) ** 2)
        loss = torch.mean(loss)
        return loss

    def loss_fn(self, pred, label):

        if self.loss_type == "mse":
            return self.mse(pred, label)

        raise ValueError("unknown loss `%s`" % self.loss_type)

    def metric_fn(self, pred, label):

        if self.metric_type == "" or self.metric_type == "loss":
            return self.loss_fn(pred, label)

        raise ValueError("unknown metric `%s`" % self.metric_type)
    
    # iterated to train the model on input data, updates parameters w,b. 
    def train_epoch(self):

        train_optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        features = torch.from_numpy(self.X_train)
        label = torch.from_numpy(self.y_train)    
        pred = self.model(features.float())
        ax0 = pred.shape[0]
        pred = pred.reshape([ax0, 1, 1])
        loss = self.loss_fn(pred, label)
        train_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
        train_optimizer.step()
        
        return self.model
    
    # iterated to test the model for inputed data
    def test_epoch(self, X, y):

        self.model.eval()

        scores = []
        losses = []
        pred_true = []

        with torch.no_grad():
            feature = torch.from_numpy(X)
            pred = self.model(feature.float())
            pred = pred.reshape([len(pred), 1, 1])
            label = torch.from_numpy(y)
            loss = self.loss_fn(pred, label)
            losses.append(loss.item())

            score = (self.metric_fn(pred, label))
            scores.append(score.item())

        return np.mean(losses), np.mean(scores)

    def fit(self):

        stop_steps = 0
        best_score = 0
        best_epoch = 0
        evals_result_train = []
        evals_loss_val = []
        evals_result_val = []
        evals_loss_train = []
        pred_true = []

        # train

        for step in range(self.n_epochs):

            print("epoch number = ", step)

            model = self.train_epoch()

            train_loss, train_score = self.test_epoch(X_train, y_train)
            val_loss, val_score = self.test_epoch(X_val, y_val)

            evals_loss_train.append(train_loss)
            evals_loss_val.append(val_loss)
            evals_result_train.append(train_score)
            evals_result_val.append(val_score)
            best_param = copy.deepcopy(self.model.state_dict())
            

            # early stop number of times we tolerate divergent results in a row.
            diff_loss = val_loss - train_loss

            if (diff_loss < .05) and (
                    diff_loss < (evals_loss_val[step - 1] - evals_loss_train[step - 1])):

                best_score = 1 - val_loss
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    break

        print("best score = ",best_score)
        print("best epoch = ",best_epoch)
        
        # plot train, val loss vs epoch
        plt.figure(facecolor = 'white')
        plt.plot(evals_loss_train, label="train loss")
        plt.plot(evals_loss_val, label="validation loss")
        plt.title(label="Train and Val loss/epoch")
        plt.legend()
        plt.show()

        return best_param, model

    def test(self, model, X_test, y_test):
        
        model.eval()
        
        #test_pred = []
        losses = []

        with torch.no_grad():
            test_features = torch.from_numpy(X_test)
            test_pred = model(test_features.float())
            ax0 = test_pred.shape[0]
            test_pred = test_pred.reshape([ax0, 1, 1])
            label = torch.from_numpy(y_test)
            print("Label shape = ")
            #test_pred.append(test_pred)
            mean_loss = self.loss_fn(test_pred, label)
            #losses.append(loss.item())

        # plots predicted, actual values vs time
        plt.figure(facecolor = "white")
        plt.plot(y_test[:,0,0], label = "real close price")
        plt.plot(test_pred[:,0,0], label = "predicted close price")
        plt.title(label = "Model predictions vs real values")
        
        return test_pred, mean_loss
    
    def predict(self, X_pred):

        X_pred = torch.from_numpy(X_pred)
        self.model.eval()
        preds = []

        with torch.no_grad():
            pred = self.model(X_pred).detach().numpy()

        preds.append(pred)

        return pd.Series(np.concatenate(preds), )
    # plot actual vs predicted prices


# unscale fn should take [i:i+window,:] of X and target val- pred. Unscale them by picking the max val
# from this set

def unscale(X_true, y_pred):
    y_pred_true = np.zeros((len(X_true[:, 0, 0]), 1, 1))

    for i in range(len(X_true[:, 0, 0])):
        for j in range(X_true[0,0])
        unscale_factor = X_true[i, :, j].max() - X_true[i, :, j].min()
        min_val = X_true[i, :, 0].min()

        y_pred_true[i, :, :] = unscale_factor * y_pred[i, :, :] + min_val

    return y_pred_true


class ALSTMModel(nn.Module):
    def __init__(self, d_feat, hidden_size, num_layers, dropout, rnn_type="GRU"):
        super().__init__()
        self.hid_size = hidden_size
        self.input_size = d_feat
        self.dropout = dropout
        self.rnn_type = rnn_type
        self.rnn_layer = num_layers
        self._build_model()

    def _build_model(self):
        try:
            klass = getattr(nn, self.rnn_type.upper())
        except:
            raise ValueError("unknown rnn_type `%s`" % self.rnn_type)
        self.net = nn.Sequential()
        self.net.add_module("fc_in", nn.Linear(in_features=self.input_size, out_features=self.hid_size))
        self.net.add_module("act", nn.Tanh())
        self.rnn = klass(
            input_size=self.hid_size,
            hidden_size=self.hid_size,
            num_layers=self.rnn_layer,
            batch_first=True,
            dropout=self.dropout,
        )
        self.fc_out = nn.Linear(in_features=self.hid_size * 2, out_features=1)
        self.att_net = nn.Sequential()
        self.att_net.add_module(
            "att_fc_in",
            nn.Linear(in_features=self.hid_size, out_features=int(self.hid_size / 2)),
        )
        self.att_net.add_module("att_dropout", torch.nn.Dropout(self.dropout))
        self.att_net.add_module("att_act", nn.Tanh())
        self.att_net.add_module(
            "att_fc_out",
            nn.Linear(in_features=int(self.hid_size / 2), out_features=1, bias=False),
        )
        self.att_net.add_module("att_softmax", nn.Softmax(dim=1))

    def forward(self, inputs):
        rnn_out, _ = self.rnn(self.net(inputs))  # [batch, seq_len, num_directions * hidden_size]
        attention_score = self.att_net(rnn_out)  # [batch, seq_len, 1]
        out_att = torch.mul(rnn_out, attention_score)
        out_att = torch.sum(out_att, dim=1)
        out = self.fc_out(
            torch.cat((rnn_out[:, -1, :], out_att), dim=1)
        )  # [batch, seq_len, num_directions * hidden_size] -> [batch, 1]
        return out[..., 0]
class GATModel(nn.Module):
    def __init__(self, d_feat, hidden_size, num_layers, dropout, base_model="GRU"):
        super().__init__()

        if base_model == "GRU":
            self.rnn = nn.GRU(
                input_size=d_feat,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
        elif base_model == "LSTM":
            self.rnn = nn.LSTM(
                input_size=d_feat,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
        else:
            raise ValueError("unknown base model name `%s`" % base_model)

        self.hidden_size = hidden_size
        self.d_feat = d_feat
        self.transformation = nn.Linear(self.hidden_size, self.hidden_size)
        self.a = nn.Parameter(torch.randn(self.hidden_size * 2, 1))
        self.a.requires_grad = True
        self.fc = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc_out = nn.Linear(hidden_size, 1)
        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

    def cal_attention(self, x, y):
        x = self.transformation(x)
        y = self.transformation(y)

        sample_num = x.shape[0]
        dim = x.shape[1]
        e_x = x.expand(sample_num, sample_num, dim)
        e_y = torch.transpose(e_x, 0, 1)
        attention_in = torch.cat((e_x, e_y), 2).view(-1, dim * 2)
        self.a_t = torch.t(self.a)
        attention_out = self.a_t.mm(torch.t(attention_in)).view(sample_num, sample_num)
        attention_out = self.leaky_relu(attention_out)
        att_weight = self.softmax(attention_out)
        return att_weight

    def forward(self, x):
        out, _ = self.rnn(x)
        hidden = out[:, -1, :]
        att_weight = self.cal_attention(hidden, hidden)
        hidden = att_weight.mm(hidden) + hidden
        hidden = self.fc(hidden)
        hidden = self.leaky_relu(hidden)
        return self.fc_out(hidden).squeeze()

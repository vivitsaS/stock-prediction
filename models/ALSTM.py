# model definition
# this model should take in train.size() number of 30*6 arrays
from __future__ import division
from __future__ import print_function

import torch.nn as nn

import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from intelLib.data import preprocessing


class Task:

    def __init__(self, model, path, loss_type, metric_type, n_epochs, lr, early_stop):

        self.model = model
        self.path = path
        self.loss_type = loss_type
        self.metric_type = metric_type
        self.n_epochs = n_epochs
        self.lr = lr
        self.early_stop = early_stop

        df = preprocessing.CSVDataset(path).ti()
        train, val, test = preprocessing.CSVDataset(path).get_split(df, 0.6, 0.15, 0.25)
        # x,y split

        self.X_train, self.y_train, self.X_train_true, self.y_train_true = preprocessing.CSVDataset(path).Xy(train)
        self.X_val, self.y_val, self.X_val_true, self.y_val_true = preprocessing.CSVDataset(path).Xy(val)
        self.X_test, self.y_test, self.X_test_true, self.y_test_true = preprocessing.CSVDataset(path).Xy(test)

        # self.X_train = X_train

        """self.pred = pred
        self.pred_true = pred_true"""
        print("n_epochs = ", n_epochs)

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

    def train_epoch(self):

        train_optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        feature = torch.from_numpy(self.X_train)
        label = torch.from_numpy(self.y_train)
        pred = self.model(feature.float())
        ax0 = pred.shape[0]
        pred = pred.reshape([ax0, 1, 1])
        loss = self.loss_fn(pred, label)
        train_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
        train_optimizer.step()

    def test_epoch(self, X_test, y_test, X_test_true, y_test_true):

        self.model.eval()

        scores = []
        losses = []
        pred_true = []

        with torch.no_grad():
            feature = torch.from_numpy(X_test)
            pred = self.model(feature.float())
            ax0 = pred.shape[0]
            pred = pred.reshape([ax0, 1, 1])
            label = torch.from_numpy(y_test)
            loss = self.loss_fn(pred, label)
            losses.append(loss.item())

            score = (self.metric_fn(pred, label))
            scores.append(score.item())

            # to obtain true values for comparison plot
            y_pred = pred
            y_pred_true = unscale(X_test_true, y_pred)
            # print("y_pred_true shape",y_pred_true.shape)

        """# The plots should show true values vs predicted values.
        plt.plot(label[:, :, 0], label="true")
        plt.plot(pred[:, :, 0], label="predicted")
        plt.legend()
        plt.show()"""

        """plt.plot(y_test_true[:,0], label="true")
        plt.plot(y_pred_true[:,:,0], label="predicted") """

        return np.mean(losses), np.mean(scores), y_pred

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
            # didn't use test set!
            print("epoch number = ", step)

            self.train_epoch()

            train_loss, train_score, y_pred_train = self.test_epoch(self.X_train, self.y_train, self.X_train_true,
                                                                    self.y_train_true)
            val_loss, val_score, y_pred_val = self.test_epoch(self.X_val, self.y_val, self.X_val_true, self.y_val_true)

            evals_loss_train.append(train_loss)
            evals_loss_val.append(val_loss)
            evals_result_train.append(train_score)
            evals_result_val.append(val_score)
            best_param = copy.deepcopy(self.model.state_dict())

            """print("train loss = ", val_loss)
            print("validation loss = ", train_loss)"""

            # early stop
            # early stop doesn't mean the max step until which we tolerate bad results.
            # It means the number of times we tolerate bad results in a row.
            diff_loss = val_loss - train_loss
            """print("diff_loss = ",diff_loss)"""
            # print("evals_loss_train[step - 1] - evals_loss_val[step - 1]) = ", evals_loss_train[step - 1] - evals_loss_val[step - 1])
            # if for the past 7 epochs, the diff loss keeps increasing it means that it is now overfitting.
            # best parameters are the ones for which the diff loss was the lowest in the past 8 epochs.

            if (diff_loss < .05) and (
                    diff_loss < (evals_loss_val[step - 1] - evals_loss_train[step - 1])):

                #print("Val loss = ", val_loss)
                best_score = 1 - val_loss
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    break

        print("best score={}".format(best_score))
        print("best epoch={}".format(best_epoch))
        print("evals_result_train={}".format(evals_result_train))
        print("evals_result_val={}".format(evals_result_val))

        plt.plot(evals_loss_train, label="train loss")
        plt.plot(evals_loss_val, label="validation loss")
        plt.title(label="Train and Val loss/epoch")
        plt.legend()
        plt.show()

        return y_pred_train, best_param

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
        unscale_factor = X_true[i, :, 0].max() - X_true[i, :, 0].min()
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

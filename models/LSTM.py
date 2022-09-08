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

            train_loss, train_score = self.test_epoch(self.X_train,self.y_train)
            val_loss, val_score = self.test_epoch(self.X_val, self.y_val)

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
        plt.plot(y_test[:50,0,0], label = "real close price")
        plt.plot(test_pred[:50,0,0], label = "predicted close price")
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
    
class LSTMModel(nn.Module):
    def __init__(self, d_feat, hidden_size, num_layers, dropout):
        super().__init__()

        self.rnn = nn.LSTM(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc_out = nn.Linear(hidden_size, 1)

        self.d_feat = d_feat

    def forward(self, x):
        # x: [N, F*T]
        x = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x = x.permute(0, 2, 1)  # [N, T, F]
        out, _ = self.rnn(x)
        return self.fc_out(out[:, -1, :]).squeeze()

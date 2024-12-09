# Logistic Regression model using pytorch
# 
# This script includes our logistic regression implementation 
# The model uses cross-entropy loss as the loss function and 
# stochastic gradient descent as the optimization algorithm.
#
# The LRModel class is a pytorch module that defines our logistic regression model
# The LogisticRegression class extends from our Model class to train and predict using logistic regression
# - Initialize with number of inputs and number of classes

from Models.model import Model

import torch
import torch.nn as nn
import torch.optim as optim

# logistic regression model using pytorch
class LRModel(nn.Module):
    def __init__(self, n_inputs, n_classes):
        super(LRModel, self).__init__()
        self.linear = nn.Linear(n_inputs, n_classes)

    def forward(self, x):
        return self.linear(x).squeeze(-1) # squeeze to change shape from (n, 1) to (n,)

# wrapper class for LRModel
class LogisticRegression(Model): 
    def __init__(self, n_inputs, n_classes):
        self.n_inputs = n_inputs
        self.n_classes = n_classes

        # Create as a base for loading (otherwise will be overridden in training)
        self.model = LRModel(self.n_inputs, self.n_classes)
        
    def train(self, X, Y, learning_rate=0.1, epochs=1000):
        # initialize model, criterion, and optimizer
        self.model = LRModel(self.n_inputs, self.n_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        
        # convert numpy data to tensor
        X_ = torch.from_numpy(X).float()
        Y_ = torch.from_numpy(Y).long()
        
        # training loop
        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model.forward(X_).squeeze(-1)
            loss = criterion(outputs, Y_)
            loss.backward()
            optimizer.step()
            # if (epoch+1) % (epochs//10) == 0:
            #     print(f'Epoch {epoch+1}/{epochs}: loss = {loss.item():.6f}')

    def predict(self, X):
        # convert numpy data to tensor
        X_ = torch.from_numpy(X).float()

        # evaluate
        self.model.eval()
        with torch.no_grad():
            # get predictions
            outputs = self.model.forward(X_)
            _, predicted = torch.max(outputs.data, 1)
            return predicted 

    def save(self, fname):
        torch.save(self.model.state_dict(), fname)

    def load(self, fname):
        self.model.load_state_dict(torch.load(fname))

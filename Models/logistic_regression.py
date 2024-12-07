from Models.model import Model

import torch
import torch.nn as nn
import torch.optim as optim

class LogisticRegression(Model, nn.Module): 
    def __init__(self, n_inputs=31, n_classes=7):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_inputs, n_classes) 

    def forward(self, x):
        return self.linear(x).squeeze(-1) # squeeze to change shape from (n, 1) to (n,)

    def train_model(self, X, Y, learning_rate=0.1, epochs=1000):
        # criterions: 
        # nn.CrossEntropyLoss() - evaluate as classification (ordinal doesnt matter)
        # nn.MSELoss() - evaluate as regression (classes dont matter much)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=learning_rate)
        X_ = torch.from_numpy(X).float()
        Y_ = torch.from_numpy(Y).long()

        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            outputs = self.forward(X_).squeeze(-1)
            loss = criterion(outputs, Y_)
            loss.backward()
            optimizer.step()
            if (epoch+1) % (epochs//10) == 0:
                print(f'Epoch {epoch+1}/{epochs}: loss = {loss.item():.6f}')

    def predict(self, X):
        X_ = torch.from_numpy(X).float()
        self.eval()
        with torch.no_grad():
            outputs = self.forward(X_)
            _, predicted = torch.max(outputs.data, 1)
            return predicted 


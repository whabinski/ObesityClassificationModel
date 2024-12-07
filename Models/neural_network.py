from Models.model import Model
import torch
import torch.nn as nn
import torch.optim as optim

class NNClass(nn.Module):
    def __init__(self):
        super(NNClass, self).__init__()
        self.relu = nn.ReLU()
        self.f1 = nn.Linear(1, 1)

    def forward(self, x):
        return self.relu(self.f1(x));

class NeuralNetwork(Model):
    def __init__(self, learningRate):
        self.model = NNClass()
        self.optimizer = optim.SGD(self.model.parameters(), lr=learningRate)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, features, labels):
        pass

    def predict(self, features):
        pass
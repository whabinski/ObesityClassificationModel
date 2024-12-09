# Neural Network model using pytorch
#
# This script includes our neural network implementation
# The model uses cross-entropy loss as the loss function and
# stochastic gradient descent as the optimization algorithm.
#
# The NNChildClass class is a pytorch module that defines our neural network model
# The CustomDataset class is a custom helper class for loading the dataset
# The NeuralNetwork class extends from our Model class to train and predict using the neural network
# - Initialize with number of features and number of labels

from Models.model import Model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import Dataset, DataLoader

from Scripts.plots import plot_metrics

class NNChildClass(nn.Module):
    def __init__(self, feature_count, label_count):
        super(NNChildClass, self).__init__()

        # Regularization Technique
        self.droprate = 0.3
        self.dropout = nn.Dropout(self.droprate)

        # Activation Function
        self.relu = nn.ReLU()

        # Full Connected Architecture
        c = feature_count
        self.fc1 = nn.Linear(c, c*3)
        self.fc2 = nn.Linear(c*3, c*2)
        self.fc3 = nn.Linear(c*2, c)
        self.classify = nn.Linear(c, label_count)

        # Batch Normalization after each convolution
        self.bn1 = nn.BatchNorm1d(c * 3)
        self.bn2 = nn.BatchNorm1d(c * 2)
        self.bn3 = nn.BatchNorm1d(c)

    def forward(self, x):

        #Passthrough -- Convolute, Normalize, Activate, Drop.
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)

        # Classify (No RELU)
        x = self.classify(x)
        return x

# Helper Class for laoding the the dataset
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)  # Convert to PyTorch tensor
        self.labels = torch.tensor(labels, dtype=torch.float32)      # Convert to PyTorch tensor
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]



class NeuralNetwork(Model):
    def __init__(self, feature_count, label_count):

        # Hyper Parameters
        self.learning_rate = 0.02
        self.epochs = 1000
        self.batch_size = 64
        
        # Loading
        self.feature_count = feature_count
        self.label_count = label_count

    # Create 
    def create_data_loader(self, features, labels):
        dataset = CustomDataset(features, labels)
        return DataLoader(dataset, batch_size=self.batch_size)


    def train(self, features, labels):

        self.model = NNChildClass(self.feature_count, self.label_count)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        model = self.model
        model.train()

        losses = []

        data_loader = self.create_data_loader(features, labels)
        for epoch in range(self.epochs):

            # if epoch % 100 == 0:
            #     print(f'\nEpoch {epoch} ', end=' ')
            # elif epoch % 5 == 0:
            #     print('.', end='')

            totalLoss = 0
            for X, Y in data_loader:

                predicted = model(X)
                Y = Y.long()
                loss = self.criterion(predicted, Y)

                totalLoss += loss.item()

                # Backward pass and optimization
                self.optimizer.zero_grad()  # Clear previous gradients
                loss.backward()             # Backpropagate gradients
                self.optimizer.step()       # Update weights

            losses.append(totalLoss)

        # print('Finished Training')
        # print(f'Final Epoch Train Loss: {totalLoss:.4f}')
        # plot_metrics(losses, 'Loss', True)
    

    def predict(self, features):
        # Model switch
        self.model.eval()
        with torch.no_grad():

            # Convert and Predict
            features = torch.tensor(features, dtype=torch.float32)
            predictedLabels = self.model(features)

            # Convert to Predicted Score
            predictedLabels = torch.argmax(predictedLabels, dim=1)
            return predictedLabels.detach().numpy() if isinstance(predictedLabels, torch.Tensor) else predictedLabels
        
    def save(self, fname):
        torch.save(self.model.state_dict(), fname)

    def load(self, fname):
        self.model.load_state_dict(torch.load(fname))
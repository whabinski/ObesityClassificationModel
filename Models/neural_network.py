from Models.model import Model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from Scripts.plots import plot_metrics

class NNChildClass(nn.Module):
    def __init__(self, feature_count, label_count):
        super(NNChildClass, self).__init__()
        self.relu = nn.ReLU()
        
        c = feature_count
        self.fc1 = nn.Linear(c, c*3)
        self.fc2 = nn.Linear(c*3, c)
        self.classify = nn.Linear(c, label_count)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.classify(x))

        return x

import torch
from torch.utils.data import Dataset, DataLoader


# Helper Function for laoding the the dataset
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

        self.learning_rate = 0.02
        self.epochs = 800
        self.batch_size = 64

        self.model = NNChildClass(feature_count, label_count)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    def create_data_loader(self, features, labels):

        dataset = CustomDataset(features, labels)
        return DataLoader(dataset, batch_size=self.batch_size)


    def train(self, features, labels):
        model = self.model
        model.train()

        losses = []

        data_loader = self.create_data_loader(features, labels)
        for epoch in range(self.epochs):

            if epoch % 100 == 0:
                print(f'\nEpoch {epoch} ', end=' ')
            elif epoch % 5 == 0:
                print('.', end='')

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

        print('Finished Training')
        print(f'Final Epoch Train Loss: {totalLoss:.4f}')
        # plot_metrics(losses, 'Loss', True)
    

    def predict(self, features):
        return self.model(features)

    def evaluate(self, features, labels):

        # Model switch
        self.model.eval()
        with torch.no_grad():

            # Convert and Predict
            features = torch.tensor(features, dtype=torch.float32)
            predictedLabels = self.model(features)

            # Convert to Predicted Score
            predictedLabels = torch.argmax(predictedLabels, dim=1)

            # Convert to proper 
            labels = labels.numpy() if isinstance(labels, torch.Tensor) else labels
            predictedLabels = predictedLabels.detach().numpy() if isinstance(predictedLabels, torch.Tensor) else predictedLabels

            # Calculate Metrics
            accuracy = accuracy_score(labels, predictedLabels)
            precision = precision_score(labels, predictedLabels, average='weighted')
            recall = recall_score(labels, predictedLabels, average='weighted')
            f1 = f1_score(labels, predictedLabels, average='weighted')

            # Report
            print(f"Accuracy:   {accuracy*100:.2f}%")
            print(f"Precision:  {precision:.4f}")
            print(f"Recall:     {recall:.4f}")
            print(f"F1-Score:   {f1:.4f}")
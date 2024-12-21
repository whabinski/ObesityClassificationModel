#
# Functions
# load_data: loads in the data from the raw .csv file
# feature_engineering: engineer new features to the dataset
# create_feature_and_target: seperate features and labels
# sample_technique: samples labels
# split_data: splits the data into train and test sets
# add_bmi_column: adds a column (BMI) from weight and height
# numerical_correlation_analysis: funciton to perform correlation analysis on numerical features
# categorical_correlation_analysis: funciton to perform correlation analysis on categorical features using mutual information
# feature_selection: selects the best features using a threshold
# undersample_classes:  to create class sizes equal to the smallest class
# ordinalize: takes the label classes and puts them into the correct order (0, 1, 2 ...)
# preprocess_features: preprocesses input features
# plot_metrics: plot some metric over epochs, comes with defined table title.
# plot_losses: plot the training and validation losses of each model while training
# trainModels: trains the newly initialized models 
# evaluate_kfold: perform k fold cross validation for all models
# load_and_split: unction to load data, seperate features and labels, and split into training and testing sets
# eval_kfold: perform k-fold evaluation for some model
# savePickle: saves every model into its own pickle file.
# main: runs the training 'pipeline'

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import sklearn
import math
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from sklearn.svm import SVC

# Turn ON to regenerate graphs
SHOW_GRAPHS = False

#-------- Load Data ---------------------------------------------------------------------------------------------
# Load and Split Data Scripts

# function to load raw csv data
def load_data(filepath):
    return pd.read_csv(filepath)                        # read and load the dataset      

# engineer features to dataset
def feature_engineering(data):
    engineered_data = add_bmi_column(data)      # calculate and add bmi column to dataset
    return engineered_data

# function to seperate features and labels
def create_feature_and_target(data):
    engineered_data = feature_engineering(data)                    # add engineered features to data
    
    features = engineered_data.drop(columns=["NObeyesdad"])        # assign feature columns
    labels = engineered_data["NObeyesdad"]                         # assign label coloumn
    return features, labels

def sample_technique(features, labels):
    sampled_features, sampled_labels = undersample_classes(features,labels)
    return sampled_features, sampled_labels

# function to split dataset into training and testing sets
def split_data(features, labels, test_size):
    # split dataset into train and test sets
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=test_size, random_state=42)
    return train_features, test_features, train_labels, test_labels

#-------- Feature engineering  ----------------------------------------------------------------------------------
#  correlation analysis, feature augmentation, feature selection

# funciton to calculate bmi and add as new column to data
def add_bmi_column(data):
    data['BMI'] = (data['Weight'] / (data['Height'] ** 2)).round(2)     # BMI = weight (kg) / height (meters) squared
    return data

# function to perform correlation analysis on numerical features
def numerical_correlation_analysis(features, labels, threshold=0.1):
    
    data = features.copy()          # copy feature set for manipulation
    data['Target'] = labels         # add target variable to all columns

    corr_matrix = data.corr()                                                           # compute correlation matrix on columns
    target_corr = corr_matrix['Target'].drop('Target').sort_values(ascending=False)     # find all correlation values with the target
    selected_features = target_corr[abs(target_corr) >= threshold].index.tolist()       # select features with correlation greater than threshold

    # visualize scores via heatmap
    if SHOW_GRAPHS: 
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title(f"Correlation Matrix with Threshold = {threshold}")
        plt.show()
    
    return selected_features

# funciton to perform correlation analysis on categorical features using mutual information
def categorical_correlation_analysis(features, labels, threshold = 0.05):
    
    features_encoded = features.copy()                                      # copy feature set for manipulation
    for col in features.columns:                                            # iterate through all columns
        label_encoder = LabelEncoder()                                      # initialize label encoder for each column
        features_encoded[col] = label_encoder.fit_transform(features[col])  # convert column to numeric label

    
    mi = mutual_info_classif(features_encoded, labels)                                                  # calculate mutual information score between categorical feature and target using sklearns mutual_info_class_if method
    mi_df = pd.DataFrame({'Feature': features.columns, 'Mutual Information': mi})                       # create dataframe to store mi scores of each feature

    selected_categorical_features = mi_df[mi_df['Mutual Information'] > threshold]['Feature'].tolist()       # select features with correlation greater than threshold
    
    # visualization scores via barchart
    if SHOW_GRAPHS: 
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Mutual Information", y="Feature", data=mi_df)
        plt.title(f"Mutual Information Scores for Categorical Features with Threshold = {threshold}")
        plt.xlabel("Mutual Information Score")
        plt.ylabel("Features")
        plt.tight_layout()
        plt.show()
    
    return selected_categorical_features

# function to perform correlation analysis and return feature seletion
def feature_selection(categorical_columns, numerical_columns, train_features, test_features, train_labels):
    
    selected_numerical_features = numerical_correlation_analysis(train_features[numerical_columns], train_labels, threshold=0.25)            # correlation analysis on numerical features
    
    selected_categorical_features = categorical_correlation_analysis(train_features[categorical_columns], train_labels, threshold=0.1)     # correlation analysis on numerical features
    
    return selected_categorical_features, selected_numerical_features   # return feature columns

# function to create class sizes equal to the smallest class
def undersample_classes(features, labels):
 
    data = pd.concat([features, labels], axis=1)                    # combine features and labels into a single DataFrame
    label_column = labels.name                                      # name of the labels column

    class_counts = labels.value_counts()                            # counts for each class
    smallest_class_size = labels.value_counts().min()               # size of the smallest class

    balanced_data = []                                              # initialize empty list to store undersampled data

    for class_label in labels.unique():                             # iterate over all unique classes
        class_data = data[data[label_column] == class_label]                                # filter for rows of the current class
        balanced_data.append(class_data.sample(n=smallest_class_size, random_state=42))     # append reduced sample size to list of balanced data

    balanced_data = pd.concat(balanced_data, axis=0).reset_index(drop=True)         # concatenate all the balanced data
    
    balanced_features = balanced_data.drop(columns=[label_column])                  # separate the features
    balanced_labels = balanced_data[label_column]                                   # seperate labels
    
    total_removed = (class_counts.sum()) - (smallest_class_size * len(class_counts))  # Total samples removed
    print(f"Smallest class sample size: {smallest_class_size}, removed {total_removed} samples total\n")
    
    return balanced_features, balanced_labels

#-------- Preprocessing  ----------------------------------------------------------------------------------------

def ordinalize(train_labels, test_labels):
    
    # process labels to numeric format using label encoder
    label_encoder = LabelEncoder()                                                                  # initialize label encoder
    train = label_encoder.fit_transform(train_labels)                              # fit and apply label encoder to training set
    test = label_encoder.transform(test_labels)

    lookup = {
        2: 4,
        3: 5,
        4: 6,
        5: 2,
        6: 3,
    }
   
    # Lookup to transform the alphabetical order (from label_encoder) to the ordinal 
    train = np.array([lookup.get(x, x) for x in train])
    test = np.array([lookup.get(x, x) for x in test])

    return train, test

# function to preprocess train and test sets
def preprocess_features(train_features, test_features, train_labels, test_labels):
    
    # process labels to ordinal numeric format using label encoder                                                                  # initialize label encoder
    train_labels_processed, test_labels_processed = ordinalize(train_labels, test_labels)                                 # apply label encoder to test set
    
    categorical_columns = train_features.select_dtypes(include=['object', 'category']).columns.tolist()     # dynamically define categorical columns to be processed
    numerical_columns = train_features.select_dtypes(include=['number']).columns.tolist()                   # dynamically define numerical columns to be processed

    # perform correlation analysis, and feature selection
    selected_categorical_columns, selected_numerical_columns = feature_selection(categorical_columns, numerical_columns, train_features, test_features, train_labels_processed)

    print(f"Columns after feature engineering:")
    print(f"- Categorical: {len(selected_categorical_columns)} {selected_categorical_columns}")
    print(f"- Numerical: {len(selected_numerical_columns)} {selected_numerical_columns}")

    # process (nominal) categorical columns using one hot encoding
    onehot_encoder = OneHotEncoder(sparse_output=False)                                                         # initialize one hot encoding
    train_categorical_encoded = onehot_encoder.fit_transform(train_features[selected_categorical_columns])      # fit and apply encoder to training set
    test_categorical_encoded = onehot_encoder.transform(test_features[selected_categorical_columns])            # apply encoder to test set

    # process numerical columns using standard scalar or min max scalar
    #scaler = MinMaxScaler()                                                                                 # initialize min max scalar
    scaler = StandardScaler()                                                                               # initialize standard scalar
    train_numerical_scaled = scaler.fit_transform(train_features[selected_numerical_columns])               # fit and apply scalar to training set
    test_numerical_scaled = scaler.transform(test_features[selected_numerical_columns])                     # apply scalar to test set

    # recombine categorical and numerical columns
    train_features_processed = np.hstack((train_categorical_encoded, train_numerical_scaled))       # combine processed categorical and numerical train set columns
    test_features_processed = np.hstack((test_categorical_encoded, test_numerical_scaled))          # combine processed categorical and numerical test set columns
    
    # Display the number of relative amounts
    labelCount = {}
    for label in train_labels_processed:
        labelCount[label] = labelCount.get(label, 0) + 1

    for label in range(7):
        x = labelCount[label] / len(train_labels_processed)
        #print(f'Label {label}: {x*100:.3f}% of training data points')

    # Save to Numpy Files
    np.save('./Data/train_features.npy', train_features_processed);
    np.save('./Data/test_features.npy', test_features_processed);
    np.save('./Data/train_labels.npy', train_labels_processed);
    np.save('./Data/test_labels.npy', test_labels_processed);
    print('Saved Train_features, train_labels, test_features, test_labels to .npy files')

    return train_features_processed, test_features_processed, train_labels_processed, test_labels_processed

#-------- Models ------------------------------------------------------------------------------------------------

# Logistic Regression model using pytorch
# 
# The model uses cross-entropy loss as the loss function and 
# stochastic gradient descent as the optimization algorithm.

# logistic regression model using pytorch
class LRModel(nn.Module):
    def __init__(self, n_inputs, n_classes):
        super(LRModel, self).__init__()
        self.linear = nn.Linear(n_inputs, n_classes)

    def forward(self, x):
        return self.linear(x).squeeze(-1) # squeeze to change shape from (n, 1) to (n,)

# wrapper class for LRModel
class LogisticRegression(): 
    def __init__(self, n_inputs, n_classes):
        self.n_inputs = n_inputs
        self.n_classes = n_classes

        # Create as a base for loading (otherwise will be overridden in training)
        self.model = LRModel(self.n_inputs, self.n_classes)
        
    def train(self, X, Y, learning_rate=0.1, epochs=20000):
        # initialize model, criterion, and optimizer
        self.model = LRModel(self.n_inputs, self.n_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, weight_decay=0.001)
        
        # convert numpy data to tensor
        X_ = torch.from_numpy(X).float()
        Y_ = torch.from_numpy(Y).long()
        
        # training loop
        losses = []
        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model.forward(X_).squeeze(-1)
            loss = criterion(outputs, Y_)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            # if (epoch+1) % (epochs//10) == 0:
            #     print(f'Epoch {epoch+1}/{epochs}: loss = {loss.item():.6f}')
        plot_metrics(losses, 'Loss', "Linear Regression")

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
        with open(fname, 'wb') as f:
            pickle.dump({
                'model_state_dict': self.model.state_dict(),
                'n_inputs': self.n_inputs,
                'n_classes': self.n_classes,
                }, f),
            print(f'Saved Data to {fname}')

    @staticmethod
    def load(fname):
        with open(fname, 'rb') as f:
            data_loaded = pickle.load(f)

        weights = data_loaded['model_state_dict']
        n_inputs = data_loaded['n_inputs']
        n_classes = data_loaded['n_classes']

        # Reconstruct the model
        model = LRModel(n_inputs, n_classes)
        model.load_state_dict(weights)

        # Create the wrapper and populate its fields
        wrapper = LogisticRegression(n_inputs, n_classes)
        wrapper.model = model
        print(f"Wrapper model loaded from {fname}")
        return wrapper
    
# Neural Network model using pytorch
#
# This includes our neural network implementation
# The model uses cross-entropy loss as the loss function and
# stochastic gradient descent as the optimization algorithm.

# neural network using pytorch
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
        z = 60
        self.fc1 = nn.Linear(c, z)
        self.fc2 = nn.Linear(z, c)
        self.classify = nn.Linear(c, label_count)

        # Batch Normalization after each convolution
        self.bn1 = nn.BatchNorm1d(z)
        self.bn2 = nn.BatchNorm1d(c)
        # self.bn3 = nn.BatchNorm1d(c)

    def forward(self, x):

        #Passthrough -- Convolute, Normalize, Activate, Drop.
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        # x = self.relu(self.bn3(self.fc3(x)))
        # x = self.dropout(x)

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

# wrapper class for NNChildClass
class NeuralNetwork():
    def __init__(self, feature_count, label_count):

        # Hyper Parameters
        self.learning_rate = 0.1
        self.epochs = 1000
        self.batch_size = 128
        self.validationSize = 0.2

        self.meanLossWindow = 5
        self.deminishingReturnsCount = 30 # condition has to happen x times before we exit

        self.minEpochsBeforeStop = 50        
        self.lossThreshold = 0.005
        self.ejectDifference = 0.04 # if difference is less than this (bigger in negative), we exit training.

        # Loading
        self.feature_count = feature_count
        self.label_count = label_count

        # Creatr as a base for loading  (otherwise will be overridden in training)
        self.model = NNChildClass(self.feature_count, self.label_count)

    # Create data loader
    def create_data_loader(self, features, labels):
        dataset = CustomDataset(features, labels)
        return DataLoader(dataset, batch_size=self.batch_size)

    def train(self, features, labels):

        # initialize mode, optimizer, adn criterion
        self.model = NNChildClass(self.feature_count, self.label_count)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        model = self.model

        # Keep track of losses
        losses = []
        validLosses = []
        
        # create data loader 
        fTrain, fValid, lTrain, lValid = train_test_split(features, labels, test_size=self.validationSize)
        trainLoader = self.create_data_loader(fTrain, lTrain)
        validLoader = self.create_data_loader(fValid, lValid)
        
        deminisingEpochs = 0
        window = self.meanLossWindow

        # training loop
        for epoch in range(self.epochs):

            '''
            if (epoch+1) % 100 == 0:
                print(f'loss/item={losses[-1]:5f} | {losses[-2]:5f} \nEpoch {epoch} ', end=' ')
            elif epoch % 5 == 0:
                print('.', end='')
            '''

            totalTrainLoss = 0
            model.train() # train mode
            for X, Y in trainLoader:

                predicted = model(X)
                Y = Y.long()
                loss = self.criterion(predicted, Y)

                totalTrainLoss += loss.item()

                # Backward pass and optimization
                self.optimizer.zero_grad()  # Clear previous gradients
                loss.backward()             # Backpropagate gradients
                self.optimizer.step()       # Update weights
            # Save
            losses.append(totalTrainLoss)

            model.eval()
            totalValidLoss = 0
            for X, Y in validLoader:
                
                predicted = model(X)
                Y=Y.long()
                loss = self.criterion(predicted, Y)

                totalValidLoss += loss.item()
            validLosses.append(totalValidLoss)


            # Early Stopping / Window-based checks
            if epoch >= self.minEpochsBeforeStop and len(validLosses) >= 2 * window:
                # The last "window" epochs
                recent_window = validLosses[-window:]
                # The previous "window" epochs
                prev_window = validLosses[-2*window:-window]

                curr_mean = sum(recent_window) / window
                prev_mean = sum(prev_window) / window

                #  If validation loss has gotten significantly worse
                if (curr_mean - prev_mean) > self.ejectDifference:
                    # print(f'Early Stopping due to regression at epoch={epoch}/{self.epochs} ')
                    break

                #If improvement is less than a small threshold
                improvement = prev_mean - curr_mean
                if improvement < self.lossThreshold:
                    deminisingEpochs += 1
                else:
                    deminisingEpochs = 0  # reset if we see decent improvement this epoch

                # If we've stagnated for multiple windows in a row, stop
                if deminisingEpochs >= self.deminishingReturnsCount:
                    # print(f'Early Stopping due to stagnation at epoch={epoch}/{self.epochs} ')
                    break

        # print('Finished Training')
        # print(f'Final Epoch Train Loss: {totalTrainLoss:.4f}')
        # d(losses, 'Loss', True)

        # Plot
        plot_losses(losses, validLosses, "Neural Network", False);

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
        with open(fname, 'wb') as f:
            pickle.dump({
                'model_state_dict': self.model.state_dict(),
                'feature_count': self.feature_count,
                'label_count': self.label_count,
                }, f),
            print(f'Saved Data to {fname}')

    @staticmethod
    def load(fname):
        with open(fname, 'rb') as f:
            data_loaded = pickle.load(f)

        weights = data_loaded['model_state_dict']
        feature_count = data_loaded['feature_count']
        label_count = data_loaded['label_count']

        # Reconstruct the model
        model = NNChildClass(feature_count, label_count)
        model.load_state_dict(weights)

        # Create the wrapper and populate its fields
        wrapper = NeuralNetwork(feature_count, label_count)
        wrapper.model = model
        print(f"Wrapper model loaded from {fname}")
        return wrapper

# SVM model using sklearn

# This includes our SVM implementation
# The model uses sklearn's fit method: using a quadratic programming
# optimization function to optimize hinge loss

# encapsulate sklearn's svm model class
class SVM:
    # initialize class with kernel and regularization parameter c
    def __init__(self, kernel, C):
        self.model = SVC(kernel=kernel, C=C, probability=True)

    # function to train the svm model using train features and train labels
    def fit(self, train_features, train_labels):
        self.model.fit(train_features, train_labels)    # uses sklearns fit method: implements a quadratic programming optimization function to minimize hinge loss
    
    # function to make predictions on features
    def forward(self, features):
        predictions = self.model.predict(features)          # returns predicted class labels for features using sklearns predict method
        return predictions

# wrapper class for the SVM class
class SupportVectorMachine():
    # initialize class with kernel and regularization parameter c
    def __init__(self, kernel, C):
        self.model = SVM(kernel=kernel, C=C)                # set model to sklearns SVM class

    # functioin to train model using train features and labels
    def train(self, train_features, train_labels):
        self.model.fit(train_features, train_labels)        # call sklearns fit method

    # predict class labels for input features
    def predict(self, features):
        predictions = self.model.forward(features)     # get predictions for test features
        return predictions
    
    # save to pickle file
    def save(self, fname):
        with open(fname, 'wb') as f:
            pickle.dump(self, f)

    # load from pickle
    @staticmethod
    def load(fname):
        with open(fname, 'rb') as file:
            return pickle.load(file)
            
#-------- Visualization -----------------------------------------------------------------------------------------

# This function plots a training metric (loss, accuracy, etc.) over epochs for validation purposes
def plot_metrics(metric_for_epoch, metric_name, model, plot_as_log=False):

    if not SHOW_GRAPHS:
        return;

    plt.figure(figsize=(8, 6))
    epochs = np.arange(len(metric_for_epoch)) # x values

    # log metric
    if plot_as_log:
        metric_for_epoch = [math.log(x) for x in metric_for_epoch]

    # plot metric over epoch
    plt.plot(epochs, metric_for_epoch, label=f'Train {metric_name}', color='blue')

    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.title(f'{model} {metric_name} over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_losses(trainLoss, validLoss, model ,plot_as_log=False):

    if not SHOW_GRAPHS:
        return;

    plt.figure(figsize=(8, 6))
    epochs = np.arange(len(trainLoss)) # x values

    # Assertion
    assert len(trainLoss) == len(validLoss)

    # log metric
    if plot_as_log:
        trainLoss = [math.log(x) for x in trainLoss]
        validLoss = [math.log(x) for x in validLoss]

    # plot metric over epoch
    plt.plot(epochs, trainLoss, label=f'Train Loss', color='blue')
    plt.plot(epochs, validLoss, label=f'Validation Loss', color='red')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{model} Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#-------- Training ----------------------------------------------------------------------------------------------

def trainModels(models, train_features_processed, train_labels_processed, test_features_processed):
    model_train_predictions = {}  # initialize empty dictionary to store train predictions for each model
    model_test_predictions = {}  # initialize empty dictionary to store train predictions for each model

    for name, model in models.items():                                      # iterate through all models
        print(f"Training {name} model...") 

        # Train, Predict
        model.train(train_features_processed, train_labels_processed)       # train the model on train data
        train_predictions = model.predict(train_features_processed)               # make predictions on train features (for bias variance evaluation)
        test_predictions = model.predict(test_features_processed)                # make predictions on test features
        
        model_train_predictions[name] = train_predictions  # Store the train predictions for the current model
        model_test_predictions[name] = test_predictions  # Store the test predictions for the current model

    return model_train_predictions, model_test_predictions  # Return a dictionary containing predictions for all models

# function to perform k fold cross validation
def evaluate_kfold(model, features, labels, folds):

    kfolds = KFold(n_splits=folds, shuffle=True, random_state=42)                   # initialize sklearn's k-fold cross validation

    accuracy_list = []      # initialize empty list for accuracy scores per fold
    precision_list = []     # initialize empty list for precision scores per fold
    recall_list = []        # initialize empty list for recall scores per fold
    f1_list = []            # initialize empty list for f1 scores per fold

    split_indices = kfolds.split(features)      # generate train and test indicies for each new fold

    # Iterate through the indices
    for fold, (train_idx, test_idx) in enumerate(split_indices):                     # iterate through all train and test features for each fold 

        # Split data into train and test for the fold
        train_features, test_features = features[train_idx], features[test_idx]
        train_labels, test_labels = labels[train_idx], labels[test_idx]

        model.train(train_features, train_labels)                                   # train the model using train features
        predictions = model.predict(test_features)                                  # make predictions on test features

        accuracy_list.append(accuracy_score(test_labels, predictions))                              # add current fold accuracy score to list
        precision_list.append(precision_score(test_labels, predictions, average='weighted', zero_division=0))        # add current fold precision score to list
        recall_list.append(recall_score(test_labels, predictions, average='weighted', zero_division=0))              # add current fold recall score to list
        f1_list.append(f1_score(test_labels, predictions, average='weighted', zero_division=0))                      # add current fold f1 score to list

        print(f"Fold {fold + 1}/{folds} complete")

    accuracy_avg = sum(accuracy_list) / len(accuracy_list)      # calculate average for accuracy
    precision_avg = sum(precision_list) / len(precision_list)   # calculate average for precision
    recall_avg = sum(recall_list) / len(recall_list)            # calculate average for recall
    f1_avg = sum(f1_list) / len(f1_list)                        # calculate average for f1

    # print the average metrics
    print("\nAverage Metrics Across All Folds:")
    print(f"- Accuracy: {accuracy_avg:.4f}")
    print(f"- Precision: {precision_avg:.4f}")
    print(f"- Recall: {recall_avg:.4f}")
    print(f"- F1-Score: {f1_avg:.4f}")

#-------- Main --------------------------------------------------------------------------------------------------
#
#   This sections purpose is to perform the main flow of training

# function to load data, seperate features and labels, and split into training and testing sets
def load_and_split(data_path):
    data = load_data(data_path)                                                                                 # load raw csv data
    features, labels = create_feature_and_target(data)                                                          # seperate features and labels
    #features, labels = sample_technique(features, labels)                                                      # reduce class sizes to be equal
    train_features, test_features, train_labels, test_labels = split_data(features, labels, test_size=0.2)      # split into train and test sets
    return train_features, test_features, train_labels, test_labels

# function to perform kfold cross validation
def eval_kfold(models, train_features_processed, train_labels_processed):
    for name, model in models.items():                                                      # iterate through models
        print('-' * 60)
        print(f"{name}:\n")
        print(f"Performing K-Fold Cross-Validation...")                        
        evaluate_kfold(model, train_features_processed, train_labels_processed, folds=5)    # perform kfold

def savePickle(models):
    print("\n")
    # Save the models to a pickle file
    if not os.path.isdir('./pickle'):
        os.mkdir('./pickle/')

    for name, model in models.items():
        fname = './pickle/' + name.replace(' ', '').lower() + '.pkl'
        print(f'Writing {name} to Pickle File: {fname}')
        model.save(fname)

# main
def main():

    #
    # 1. Load Data & Split
    #
    print('\n' + '=' * 60 + '\n')
    print("Loading, balancing, and splitting data...\n")
    data_path = "Data/ObesityDataSet_raw.csv"                                               # raw dataset path
    train_features, test_features, train_labels, test_labels = load_and_split(data_path)    # load data, split into train and test sets

    #
    # 2. Preprocess Data
    #

    # preprocess train and test sets
    print("Pre processing data...\n")
    train_features_processed, test_features_processed, train_labels_processed, test_labels_processed = preprocess_features(train_features, test_features, train_labels, test_labels)
    
    # Define the Expected Input and Output dimensions for classification.
    # THese are technically constant.
    featureCount = train_features_processed.shape[1]
    labelCount = len(train_labels.unique())

    #
    # 3. Initialize the Models
    #

    # Initivalize Models
    svm = SupportVectorMachine(kernel='rbf', C=5)                               # initialize support vector machine model
    nn = NeuralNetwork(feature_count=featureCount, label_count=labelCount)      # initialize neural network model
    lr = LogisticRegression(featureCount,labelCount)                            # initailize logistic regression model
    
    # Save Model in a dictionary to simplify following steps
    models = {
        'Support Vector Machine': svm, 
        'Neural Network': nn, 
        'Logistic Regression': lr,
    }
    
    #
    # 4. Feature Analysis -- Including Trainig in each function
    #
    
    # Train models
    print('\n' + '=' * 60 + '\n')
    global SHOW_GRAPHS
    # SHOW_GRAPHS = True
    model_train_predictions, model_test_predictions = trainModels(models, train_features_processed, train_labels_processed, test_features_processed)

    # SAVED THE MODEL INITIALLY
    # 5. Save to Pickle
    savePickle(models)

    # In Training - Evaluate KFold using training data
    print('\n' + '=' * 60 + '\n')
    print("Beginning K-Fold Cross Validation")
    SHOW_GRAPHS = False
    eval_kfold(models, train_features_processed, train_labels_processed)                                                               # evaluate kfold
    
    
#
# Run Main Function
#
if __name__=='__main__':
    # Initialize the Random Seed for NP
    np.random.seed(42)
    main()


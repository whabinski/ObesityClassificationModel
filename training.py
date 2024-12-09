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
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from sklearn.svm import SVC

from test import evaluate_metrics, evaluate_bias_variance

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

# function to split dataset into training and testing sets
def split_data(features, labels, test_size):
    # split dataset into train and test sets
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=test_size, random_state=42)
    return train_features, test_features, train_labels, test_labels

#-------- Feature engineering  ----------------------------------------------------------------------------------
#  correlation analysis, feature augmentation, feature selection

SHOW_GRAPHS = False

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
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(f"Correlation Matrix with Threshold = {threshold}")
    if SHOW_GRAPHS: 
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
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Mutual Information", y="Feature", data=mi_df)
    plt.title(f"Mutual Information Scores for Categorical Features with Threshold = {threshold}")
    plt.xlabel("Mutual Information Score")
    plt.ylabel("Features")
    plt.tight_layout()
    if SHOW_GRAPHS: 
        plt.show()
    
    return selected_categorical_features

# function to perform correlation analysis and return feature seletion
def feature_selection(categorical_columns, numerical_columns, train_features, test_features, train_labels):
    
    selected_numerical_features = numerical_correlation_analysis(train_features[numerical_columns], train_labels, threshold=0.1)            # correlation analysis on numerical features
    
    selected_categorical_features = categorical_correlation_analysis(train_features[categorical_columns], train_labels, threshold=0.05)     # correlation analysis on numerical features
    
    return selected_categorical_features, selected_numerical_features   # return feature columns
    

#-------- Preprocessing  ----------------------------------------------------------------------------------------

# function to preprocess train and test sets
def preprocess_features(train_features, test_features, train_labels, test_labels):
    
    # process labels to numeric format using label encoder
    label_encoder = LabelEncoder()                                                                  # initialize label encoder
    train_labels_processed = label_encoder.fit_transform(train_labels)                              # fit and apply label encoder to training set
    test_labels_processed = label_encoder.transform(test_labels)                                    # apply label encoder to test set
    
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

    # process numerical columns using standard scalar
    scaler = StandardScaler()                                                                               # initialize standard scalar
    train_numerical_scaled = scaler.fit_transform(train_features[selected_numerical_columns])               # fit and apply scalar to training set
    test_numerical_scaled = scaler.transform(test_features[selected_numerical_columns])                     # apply scalar to test set

    # recombine categorical and numerical columns
    train_features_processed = np.hstack((train_categorical_encoded, train_numerical_scaled))       # combine processed categorical and numerical train set columns
    test_features_processed = np.hstack((test_categorical_encoded, test_numerical_scaled))          # combine processed categorical and numerical test set columns
    
    # Save to Numpy Files
    np.save('./Data/train_features.npy', train_features_processed);
    np.save('./Data/test_features.npy', train_labels_processed);
    np.save('./Data/train_labels.npy', test_features_processed);
    np.save('./Data/test_lables.npy', test_labels_processed);
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

# wrapper class for NNChildClass
class NeuralNetwork():
    def __init__(self, feature_count, label_count):

        # Hyper Parameters
        self.learning_rate = 0.02
        self.epochs = 1
        self.batch_size = 64
        
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
        model.train() # train mode

        losses = []
        # create data loader 
        data_loader = self.create_data_loader(features, labels)
        
        # training loop
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
            pickle.dump(self.model, f)

    # load from pickle
    def load(self, fname):
        with open(fname, 'rb') as file:
            self.model = pickle.load(file)
            
#-------- Visualization -----------------------------------------------------------------------------------------

# This function plots a training metric (loss, accuracy, etc.) over epochs for validation purposes
def plot_metrics(metric_for_epoch, metric_name, plot_as_log=False):

    plt.figure(figsize=(8, 6))
    epochs = np.arange(len(metric_for_epoch)) # x values

    # log metric
    if plot_as_log:
        metric_for_epoch = [math.log(x) for x in metric_for_epoch]

    # plot metric over epoch
    plt.plot(epochs, metric_for_epoch, label=f'Train {metric_name}', color='blue')

    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} over Epochs')
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
    train_features, test_features, train_labels, test_labels = split_data(features, labels, test_size=0.2)      # split into train and test sets
    return train_features, test_features, train_labels, test_labels

# function to perform kfold cross validation
def eval_kfold(models, train_features_processed, train_labels_processed):
    for name, model in models.items():                                                      # iterate through models
        print('-' * 60)
        print(f"{name}:\n")
        print(f"Performing K-Fold Cross-Validation...")                        
        evaluate_kfold(model, train_features_processed, train_labels_processed, folds=5)    # perform kfold

# function to evaluate regular accuracy, precisoin, recall, f1, confusion matrix metrics
def eval_metrics(models, test_predictions, test_labels_processed):
    for name, model in models.items():                                      # iterate through all models
        print('-' * 60)
        print(f"Performing metric evaluation for {name}...") 
        
        # Evaluate
        evaluate_metrics(test_labels_processed, test_predictions[name])                # calculate metrics

# function to evaluate bias and variance
def eval_bias_variance(models, train_labels_processed, test_labels_processed, train_predictions, validation_predictions):
    for name, model in models.items():                                  # iterate through models
        print('-' * 60)
        print(f"Performing Bias-Variance Analysis for {name}...")
        # evaluate bias and variance
        evaluate_bias_variance(train_labels_processed, test_labels_processed, train_predictions[name], validation_predictions[name])

def savePickle(models):
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
    print("Loading and splitting data...\n")
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
    svm = SupportVectorMachine(kernel='linear', C=1)                             # initialize support vector machine model
    nn = NeuralNetwork(feature_count=featureCount, label_count=labelCount)      # initialize neural network model
    lr = LogisticRegression(featureCount,labelCount)                           # initailize logistic regression model
    
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
    model_train_predictions, model_test_predictions = trainModels(models, train_features_processed, train_labels_processed, test_features_processed)

    # Evaluate using training data
    print('\n' + '=' * 60 + '\n')
    print("Beginning K-Fold Cross Validation")
    eval_kfold(models, train_features_processed, train_labels_processed)                                                               # evaluate kfold
    
    #
    # 5. Save to Pickle
    #

    savePickle(models)
    
    # Evalaute using Test
    print('\n' + '=' * 60 + '\n')
    print("Beginning Bias-Variance Analysis")
    eval_bias_variance(models, train_labels_processed, test_labels_processed, model_train_predictions, model_test_predictions)       # evaluate bias and variance
    
    # Evalaute using Test
    print('\n' + '=' * 60 + '\n')
    print("Beginning Metric Evaluations")
    eval_metrics(models, model_test_predictions, test_labels_processed)              # evaluate metrics

    print('\n' + '=' * 60 + '\n')

# Sample Loading each model from Pickle File.
def load_models_sample(test_features_processed, featureCount, labelCount):

    svm = SupportVectorMachine(kernel='linear', C=1)
    nn = NeuralNetwork(featureCount, labelCount)
    lr = LogisticRegression(featureCount,labelCount)

    #
    # Note: The versions of Pickle, Scikit-learn, Torch and Numpy
    # impact the ability to run this. The installed versions of each the above
    # must match (to a certain degree) as to the ones that we compiled.
    #
    # In the event you do not have the most current versions of each, you can run
    # the main file and the pickle files will be recreated, allowing you to load them in.
    # The current python version that created the pickle files was python 12.7
    #
    svm.load('./pickle/supportvectormachine.pkl')
    nn.load('./pickle/neuralnetwork.pkl')
    lr.load('./pickle/logisticregression.pkl')

    predicted1 = svm.predict(test_features_processed)
    predicted2 = nn.predict(test_features_processed)
    predicted3 = lr.predict(test_features_processed)

    print('Pickle Loaded SVM Predicted Classes:', predicted1[:10], '...')
    print('Pickle Loaded NN Predicted Classes:', predicted2[:10], '...')
    print('Pickle Loaded LR Predicted Classes:', predicted3[:10], '...')
    return svm, nn, lr


#
# Run Main Function
#
if __name__=='__main__':
    # Initialize the Random Seed for NP
    np.random.seed(42)
    main()


# Evaluation Scripts
#
# This file contains functions used for evaluating our models.
# Functions:
# - evaluate_metrics: computes accuracy, precision, recall, f1 score, and confusion matrix given test labels and predictions
# - evaluate_bias_variance: evaluates bias and variance and displays training and validation error

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Loading from Pickle
from training import LogisticRegression, NeuralNetwork, SupportVectorMachine, SVM
from sklearn.svm import SVC

# funtion to evaluate performance of models using basic metrics; accuracy, precision, recall, f1, and confusion matrix
def evaluate_metrics(test_labels, test_predictions):
    accuracy = accuracy_score(test_labels, test_predictions)                                                # calucalte accuracy using sklearns accuracy method: proportion of correctly classified samples
    precision = precision_score(test_labels, test_predictions, average='weighted', zero_division=0)         # calucalte precision using sklearns precision method: ratio of TP/TP+FP averaged over each class (weighted)
    recall = recall_score(test_labels, test_predictions, average='weighted')                                # calucalte recall using sklearns recall method: ratio of TP/TP+FN averaged over each class (weighted)
    f1 = f1_score(test_labels, test_predictions, average='weighted')                                        # calucalte f1 using sklearns f1 method: 2/ inv(precision) + inv(recall) averaged over each class (weighted)
    cm = confusion_matrix(test_labels, test_predictions)                                                    # calucalte confusion matrix using sklearns confusion matrix method

    # print all metrics
    print(f"- Accuracy: {accuracy:.4f}")
    print(f"- Precision: {precision:.4f}")
    print(f"- Recall: {recall:.4f}")
    print(f"- F1-Score: {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")

# function to evaluate bias and variance
def evaluate_bias_variance(labels_train, labels_test, train_predictions, validation_predictions):

    train_error = 1 - accuracy_score(labels_train, train_predictions)               # error on the training set (number of incorrect predictions / total samples)
    validation_error = 1 - accuracy_score(labels_test, validation_predictions)     # error on the testing set 

    print(f"- Training Error: {train_error:.4f}")             # print training errors
    print(f"- Validation Error: {validation_error:.4f}")      # print validation errors

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


def load_models():
    
    # Load in arrays
    # svm = SupportVectorMachine(kernel='linear', C=1)
    # nn = NeuralNetwork(1, 1) #will get overriden
    # lr = LogisticRegression(1, 1) #will get overriden
    
    # #
    # # Note: The versions of Pickle, Scikit-learn, Torch and Numpy
    # # impact the ability to run this. The installed versions of each the above
    # # must match (to a certain degree) as to the ones that we compiled.
    # #
    # # In the event you do not have the most current versions of each, you can run
    # # the main file and the pickle files will be recreated, allowing you to load them in.
    # # The current python version that created the pickle files was python 12.7
    # #

    svm = SupportVectorMachine.load('./pickle/supportvectormachine.pkl')
    nn = NeuralNetwork.load('./pickle/neuralnetwork.pkl')
    lr = LogisticRegression.load('./pickle/logisticregression.pkl')

    return {
        'Support Vector Machine': svm,
        'Neural Network': nn,
        'Logistic Regression': lr
    }

def main():
    
    # We have saved the pickle files in our own folder within the file structure so there is no need for you to move pickle files into file structure
    models = load_models()

    # Load in Data (Labels)
    train_labels_processed = np.load('./Data/train_labels.npy')
    test_labels_processed = np.load('./Data/test_labels.npy')

    # Load in Data (Features)
    train_features_processed = np.load('./Data/train_features.npy')
    test_features_processed = np.load('./Data/test_features.npy')

    # Perform Predictions
    model_train_predictions = {}
    model_test_predictions = {}
    for name, model in models.items():
        model_train_predictions[name] = np.array(model.predict(train_features_processed))
        model_test_predictions[name] = np.array(model.predict(test_features_processed))

    # Evalaute using Test
    print('\n' + '=' * 60 + '\n')
    print("Beginning Bias-Variance Analysis")
    eval_bias_variance(models, train_labels_processed, test_labels_processed, model_train_predictions, model_test_predictions)       # evaluate bias and variance
    
    # Evalaute using Test
    print('\n' + '=' * 60 + '\n')
    print("Beginning Metric Evaluations")
    eval_metrics(models, model_test_predictions, test_labels_processed)     # evaluate metrics

    print('\n' + '=' * 60 + '\n')

if __name__ == '__main__':
    main()
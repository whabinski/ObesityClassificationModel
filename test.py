# Evaluation Scripts
#
# This file contains functions used for evaluating our models.
# Functions:
# - evaluate_metrics: computes accuracy, precision, recall, f1 score, and confusion matrix given test labels and predictions
# - evaluate_bias_variance: evaluates bias and variance and displays training and validation error

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

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
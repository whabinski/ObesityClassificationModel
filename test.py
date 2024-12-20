# Evaluation Scripts
#
# This file contains functions used for evaluating our models.
# Functions:
# - evaluate_metrics: computes accuracy, precision, recall, f1 score, and confusion matrix given test labels and predictions
# - evaluate_bias_variance: evaluates bias and variance and displays training and validation error

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Loading from Pickle
from training import LogisticRegression, NeuralNetwork, SupportVectorMachine, SVM
from sklearn.svm import SVC

SHOW_GRAPHS = False

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

# function to comparitively plot metrics of 2 differnt models using a bar chart
def plot_compare_metrics(title, metrics_model_old, metrics_model_new, old_name="Old", new_name="New"):

    labels = list(metrics_model_old.keys())                 # metric labels
    old_values = list(metrics_model_old.values())           # old values
    new_values = list(metrics_model_new.values())           # new values
    
    x = np.arange(len(labels))              
    width = 0.35                            # width of the bars
    
    fig, ax = plt.subplots(figsize=(8, 6))                                      # create the sub plots
    ax.bar(x - width/2, old_values, width, label=old_name, color='blue')        # plot blue / old values
    ax.bar(x + width/2, new_values, width, label=new_name, color='red')         # plot red  / new values
    
    ax.set_xlabel('Metrics')            # x axis title
    ax.set_ylabel('Scores')             # y axis title
    ax.set_title(title)                 # plot title
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    plt.tight_layout()          # tight layout
    plt.show()                  # Show the plot

# function to produce heatmap for confusion matrix
def plot_confusion_matrix(true_labels, predicted_labels, class_names, model):
    cm = confusion_matrix(true_labels, predicted_labels)    # create confusion matrix
    
    plt.figure(figsize=(8, 6))                              # make plot
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="coolwarm", xticklabels=class_names, yticklabels=class_names)       # create heatmap using seaborn and confusion matrix
    
    plt.title(f"Confusion Matrix for {model}")      # title for plot
    plt.xlabel("Predicted Labels")                  # x axis title
    plt.ylabel("True Labels")                       # y axis title
    
    plt.tight_layout()              # tight layout
    plt.show()                      # show plot

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
    
    svm_old_hyperparameters = {'Accuracy': 0.9385, 'Precision': 0.9420, 'Recall': 0.9385, 'F1-Score': 0.9376, 'Training Error': 0.0355, 'Validation Error': 0.0615}
    svm_new_hyperparameters = {'Accuracy': 0.9504, 'Precision': 0.9538, 'Recall': 0.9504, 'F1-Score': 0.9498, 'Training Error': 0.0113, 'Validation Error': 0.0496}
    svm_hyperparameter_graph_title = "SVM Hyperparameters Change"
    
    nn_old_hyperparameters = {'Accuracy': 0.9267, 'Precision': 0.9378, 'Recall': 0.9267, 'F1-Score': 0.9251, 'Training Error': 0.0249, 'Validation Error': 0.0733}
    nn_new_hyperparameters = {'Accuracy': 0.9409, 'Precision': 0.9431, 'Recall': 0.9409, 'F1-Score': 0.9404, 'Training Error': 0.0278, 'Validation Error': 0.0591}
    nn_hyperparameter_graph_title = "NN Hyperparameters Change"
    
    lr_old_hyperparameters = {'Accuracy': 0.7872, 'Precision': 0.7933, 'Recall': 0.7872, 'F1-Score': 0.7741, 'Training Error': 0.1872, 'Validation Error': 0.2128}
    lr_new_hyperparameters = {'Accuracy': 0.8652, 'Precision': 0.8738, 'Recall': 0.8652, 'F1-Score': 0.8609, 'Training Error': 0.1001, 'Validation Error': 0.1348}
    lr_hyperparameter_graph_title = "LR Hyperparameters Change"
    
    global SHOW_GRAPHS
    if (SHOW_GRAPHS):
        plot_compare_metrics(svm_hyperparameter_graph_title, svm_old_hyperparameters, svm_new_hyperparameters)
        plot_compare_metrics(nn_hyperparameter_graph_title, nn_old_hyperparameters, nn_new_hyperparameters)
        plot_compare_metrics(lr_hyperparameter_graph_title, lr_old_hyperparameters, lr_new_hyperparameters)

        for name, model in models.items():
            plot_confusion_matrix(test_labels_processed,model_test_predictions[name], np.unique(test_labels_processed),name)
        
if __name__ == '__main__':
    np.random.seed(42)
    main()
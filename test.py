# Evaluation Scripts
#
# This file contains functions used for evaluating our models.
# Functions:
# - evaluate_metrics: computes accuracy, precision, recall, f1 score, and confusion matrix given test labels and predictions
# - evaluate_bias_variance: evaluates bias and variance and displays training and validation error
# - eval_metrics: wrapper of evaluate_metrics, runs it for each model
# - plot_compare_metrics: plots the comparison metrics (m2 and m3 params) for a specific model
# - plot_confusion_matrix: plots the confusion matrix for a model
# - get_misclassified_indices: gets the indeces of test points that are incorrectly classified by some model
# - get_correctly_classified_indices:   gets the indeces of test points that are correctly classified by some model
# - get_predictied_sample: return predictions made by each model for a respective sample number
# - misclassified_by_all_models: gets the index of every test point which is never correctly classified by any model.
# - correctly_classified_by_all_models: gets the index of every test point that is correctly classified by all 3 models.
# - load_models: loads the models from the .pkl files
# - main: runs the evalulation 'pipeline'

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from training import LogisticRegression, NeuralNetwork, SupportVectorMachine, SVM
from sklearn.svm import SVC

# Turn ON to regenerate graphs
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

# function to identify and return indecies of misclassified samples
def get_misclassified_indices(true_labels, predicted_labels):
    return np.where(true_labels != predicted_labels)[0]

# function to identify and return indecies of correctly classified samples
def get_correctly_classified_indices(true_labels, predicted_labels):
    return np.where(true_labels == predicted_labels)[0]

# function to return predictions made by each model for a respective sample number
def get_predictied_sample(sample_index, test_features_processed, test_labels_processed, models):
    
    sample_features = test_features_processed[sample_index].reshape(1, -1)  # get sample from the preprocessed test dataset

    true_label = test_labels_processed[sample_index]    # get true label

    sample_predictions = {}                             # initialize cictionary to store predictions

    for name, model in models.items():                  # iterate through all models
        prediction = model.predict(sample_features)     # predict the sample
        sample_predictions[name] = prediction           # store to dictionary

        predicted = f'Predicted Label: {prediction[0]}'
        true = f'True Label: {true_label}'

        print(f"Model: {name:25} {predicted:20} {true}")     # print comparison

# function to find all misclassified predictions by all models    
def misclassified_by_all_models(models, model_test_predictions, test_labels_processed):
        missclassified = None

        for name, model in models.items():
            print(f"\nMisclassified Indices for {name}:")
            misclassified_indices = get_misclassified_indices(test_labels_processed, model_test_predictions[name])
            print(f'{misclassified_indices} ({len(misclassified_indices)})')

            # Set
            misclassified_indices = set(misclassified_indices)

            #Update Initially
            if (missclassified is None):
                missclassified = misclassified_indices
            missclassified = missclassified.intersection(misclassified_indices)
        
        print('\n' + '=' * 60 + '\n')
        print(f'{len(missclassified)} Points Missclassified by all Models:\n', missclassified)
        print('\n' + '=' * 60 + '\n')

# function to find all correctly classified predictions by all models    
def correctly_classified_by_all_models(models, model_test_predictions, test_labels_processed):
        correctly_classified = None

        for name, model in models.items():
            print(f"\nMisclassified Indices for {name}:")
            correctly_classified_indices = get_correctly_classified_indices(test_labels_processed, model_test_predictions[name])
            print(f'{correctly_classified_indices} ({len(correctly_classified_indices)})')

            # Set
            correctly_classified_indices = set(correctly_classified_indices)

            #Update Initially
            if (correctly_classified is None):
                correctly_classified = correctly_classified_indices
            correctly_classified = correctly_classified.intersection(correctly_classified_indices)
        
        print('\n' + '=' * 60 + '\n')
        print(f'{len(correctly_classified)} Points Missclassified by all Models:\n', correctly_classified)
        print('\n' + '=' * 60 + '\n')

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

# Checks the distance between predicted lables and true labels, and creates an accuracy score
# (Worse predictions are further off)
def ordinal_accuracy(labels, predictedLabels):

    # Get the difference between predicted lables
    difference = np.abs(labels - predictedLabels)

    # 0-6 should provide maximum penalty (1)
    penalty = difference / 6

    # 1 is perfect, 0 is awful prediction
    scores = 1 - penalty
    return np.mean(scores)


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
    
    # From Milestone II

    # old svm hyperparameter metric scores
    svm_old_hyperparameters = {'Accuracy': 0.9385, 'Precision': 0.9420, 'Recall': 0.9385, 'F1-Score': 0.9376, 'Training Error': 0.0355, 'Validation Error': 0.0615}
    # new svm hyperparameter metric scores
    svm_new_hyperparameters = {'Accuracy': 0.9504, 'Precision': 0.9538, 'Recall': 0.9504, 'F1-Score': 0.9498, 'Training Error': 0.0113, 'Validation Error': 0.0496}
    svm_hyperparameter_graph_title = "SVM Hyperparameters Change"
    
    # old nn hyperparameter metric scores
    nn_old_hyperparameters = {'Accuracy': 0.9267, 'Precision': 0.9378, 'Recall': 0.9267, 'F1-Score': 0.9251, 'Training Error': 0.0249, 'Validation Error': 0.0733}
    # new nn hyperparameter metric scores
    nn_new_hyperparameters = {'Accuracy': 0.9409, 'Precision': 0.9431, 'Recall': 0.9409, 'F1-Score': 0.9404, 'Training Error': 0.0278, 'Validation Error': 0.0591}
    nn_hyperparameter_graph_title = "NN Hyperparameters Change"
    
    # old lr hyperparameter metric scores
    lr_old_hyperparameters = {'Accuracy': 0.7872, 'Precision': 0.7933, 'Recall': 0.7872, 'F1-Score': 0.7741, 'Training Error': 0.1872, 'Validation Error': 0.2128}
    # new lr hyperparameter metric scores
    lr_new_hyperparameters = {'Accuracy': 0.8652, 'Precision': 0.8738, 'Recall': 0.8652, 'F1-Score': 0.8609, 'Training Error': 0.1001, 'Validation Error': 0.1348}
    lr_hyperparameter_graph_title = "LR Hyperparameters Change"
    
    global SHOW_GRAPHS
    if (SHOW_GRAPHS):
        plot_compare_metrics(svm_hyperparameter_graph_title, svm_old_hyperparameters, svm_new_hyperparameters)      # compare old vs new hyperparameter scores for svm
        plot_compare_metrics(nn_hyperparameter_graph_title, nn_old_hyperparameters, nn_new_hyperparameters)         # compare old vs new hyperparameter scores for nn
        plot_compare_metrics(lr_hyperparameter_graph_title, lr_old_hyperparameters, lr_new_hyperparameters)         # compare old vs new hyperparameter scores for lr

        for name, model in models.items():
            plot_confusion_matrix(test_labels_processed,model_test_predictions[name], np.unique(test_labels_processed),name)    # plot heatmap confusion matrix for all models
        
        print('\n' + '=' * 60 + '\n')
        misclassified_by_all_models(models, model_test_predictions, test_labels_processed)
        correctly_classified_by_all_models(models, model_test_predictions, test_labels_processed)
    
    #All misclassified by all 3 models: 51, 77, 144, 146, 160, 161, 244, 250, 270, 377, 411, 420    
    #get_predictied_sample(51, test_features_processed, test_labels_processed, models)   # print predictions for each respective model on an individual sample

    # Perform Ordinal Accuracy
    for name, model in models.items():
        acc = ordinal_accuracy(test_labels_processed, model_test_predictions[name])
        print(f'Ordinal Accuracy for {name}: {acc:.4f}')
    print('\n' + '=' * 60 + '\n')


if __name__ == '__main__':
    np.random.seed(42)
    main()
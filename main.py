#    Main file
#
#   This file's purpose is to perform the main flow of the application on a high level.
#   As stated in README ...
#
#       1. We import the data found in `./Data/ObesityDataSet_raw.csv`
#       2. We preform preprocessing techniques.
#       3. Initialize Models
#       4. Perform evaluation metrics. For simiplification (as we perform K-fold Cross Validation), each metric will retrain the model.
#       5. Save each model to a pickle file, found in `./pickle/(MODEL).pkl`.
#
#   Additionally, includes a smaple code for loading models out of the ./pickle folder.
#
import os
import numpy as np

from Models.logistic_regression import LogisticRegression
from Models.neural_network import NeuralNetwork
from Models.svm_model import SupportVectorMachine

from Scripts.load_and_split_data import load_data, create_feature_and_target, split_data
from Scripts.preprocess_data import preprocess_features
from Scripts.evaluations import evaluate_metrics, evaluate_kfold, evaluate_bias_variance

# function to load data, seperate features and labels, and split into training and testing sets
def load_and_split(data_path):
    data = load_data(data_path)                                                                                 # load raw csv data
    features, labels = create_feature_and_target(data)                                                          # seperate features and labels
    train_features, test_features, train_labels, test_labels = split_data(features, labels, test_size=0.2)      # split into train and test sets
    return train_features, test_features, train_labels, test_labels

# function to evaluate regular accuracy, precisoin, recall, f1, confusion matrix metrics
def eval_normal(models, train_features_processed, train_labels_processed, test_features_processed, test_labels_processed):
    for name, model in models.items():                                      # iterate through all models
        print('-' * 60)
        print(f"Performing metric evaluation for {name}...") 

        # Train, Predict
        model.train(train_features_processed, train_labels_processed)       # train the model on train data
        predictions = model.predict(test_features_processed)                # make predictions on test features
        
        # Evaluate
        evaluate_metrics(test_labels_processed, predictions)                # calculate metrics

# function to perform kfold cross validation
def eval_kfold(models, train_features_processed, train_labels_processed):
    for name, model in models.items():                                                      # iterate through models
        print('-' * 60)
        print(f"{name}:\n")
        print(f"Performing K-Fold Cross-Validation...")                        
        evaluate_kfold(model, train_features_processed, train_labels_processed, folds=5)    # perform kfold

# function to evaluate bias and variance
def eval_bias_variance(models, train_features_processed, train_labels_processed, test_features_processed, test_labels_processed):
    for name, model in models.items():                                  # iterate through models
        print('-' * 60)
        print(f"Performing Bias-Variance Analysis for {name}...")
        # evaluate bias and variance
        evaluate_bias_variance(model, train_features_processed, train_labels_processed, test_features_processed, test_labels_processed)

# main
def main():

    #
    # 1. Load Data & Split
    #
    print('=' * 60)
    print("Loading and Preprocessing Data...")
    data_path = "Data/ObesityDataSet_raw.csv"                                               # raw dataset path
    train_features, test_features, train_labels, test_labels = load_and_split(data_path)    # load data, split into train and test sets
    print("Data successfully split into training and testing sets.")
    
    #
    # 2. Preprocess Data
    #

    # preprocess train and test sets
    train_features_processed, test_features_processed, train_labels_processed, test_labels_processed = preprocess_features(train_features, test_features, train_labels, test_labels)
    print("Data successfully processed.")

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

    # Evaluate using training data
    print('\n' + '=' * 60)
    print("Beginning K-Fold Cross Validation")
    eval_kfold(models, train_features_processed, train_labels_processed)                                                               # evaluate kfold
    
    # Evalaute using Test
    print('\n' + '=' * 60)
    print("Beginning Bias-Variance Analysis")
    eval_bias_variance(models, train_features_processed, train_labels_processed, test_features_processed, test_labels_processed)       # evaluate bias and variance
    
    # Evalaute using Test
    print('\n' + '=' * 60)
    print("Beginning Metric Evaluations")
    eval_normal(models, train_features_processed, train_labels_processed, test_features_processed, test_labels_processed)              # evaluate metrics

    print('\n' + '=' * 60)

    #
    # 5. Save to Pickle
    #

    # Save the models to a pickle file
    if not os.path.isdir('./pickle'):
        os.mkdir('./pickle/')

    for name, model in models.items():
        fname = './pickle/' + name.replace(' ', '').lower() + '.pkl'
        print(f'Writing {name} to Pickle File: {fname}')
        model.save(fname)

    load_models_sample(test_features_processed, featureCount, labelCount)

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
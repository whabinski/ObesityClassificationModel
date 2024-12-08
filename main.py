import numpy as np
from Models.logistic_regression import LogisticRegression
from Models.neural_network import NeuralNetwork
from Models.svm_model import SupportVectorMachine

from Scripts.load_and_split_data import load_data, create_feature_and_target, split_data, RANDOM_SEED
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
        print(f"\nPerforming metric evaluation for {name}...") 
        model.train(train_features_processed, train_labels_processed)       # train the model on train data
        predictions = model.predict(test_features_processed)                # make predictions on test features
        evaluate_metrics(test_labels_processed, predictions)                # calculate metrics

# function to perform kfold cross validation
def eval_kfold(models, train_features_processed, train_labels_processed):
    for name, model in models.items():                                                      # iterate through models
        print(f"\nPerforming K-Fold Cross-Validation for {name}...")                        
        evaluate_kfold(model, train_features_processed, train_labels_processed, folds=5)    # perform kfold

# function to evaluate bias and variance
def eval_bias_variance(models, train_features_processed, train_labels_processed, test_features_processed, test_labels_processed):
    for name, model in models.items():                                  # iterate through models
        print(f"\nPerforming Bias-Variance Analysis for {name}...")
        # evaluate bias and variance
        evaluate_bias_variance(model, train_features_processed, train_labels_processed, test_features_processed, test_labels_processed)

# main
def main():

    data_path = "Data/ObesityDataSet_raw.csv"                                               # raw dataset path
    train_features, test_features, train_labels, test_labels = load_and_split(data_path)    # load data, split into train and test sets
    print("Data successfully split into training and testing sets.")
    
    # preprocess traina nd test sets
    train_features_processed, test_features_processed, train_labels_processed, test_labels_processed = preprocess_features(train_features, test_features, train_labels, test_labels)
    print("Data successfully processed.")

    # Define the Expected Input and Output dimensions for classification
    featureCount = train_features_processed.shape[1]
    labelCount = len(train_labels.unique())

    # Initivalize Models
    svm = SupportVectorMachine(kernel='linear', C=1)                             # initialize support vector machine model
    nn = NeuralNetwork(feature_count=featureCount, label_count=labelCount)      # initialize neural network model
    lgrg = LogisticRegression(featureCount,labelCount)                           # initailize logistic regression model
    models = {
        'Support Vector Machine': svm, 
        'Neural Network': nn, 
        'Logistic Regression': lgrg,
    }

    # Evaluate using training data
    eval_kfold(models, train_features_processed, train_labels_processed)                                                               # evaluate kfold
    eval_bias_variance(models, train_features_processed, train_labels_processed, test_features_processed, test_labels_processed)       # evaluate bias and variance
    eval_normal(models, train_features_processed, train_labels_processed, test_features_processed, test_labels_processed)              # evaluate metrics

    for name, model in models.items():
        print(f'Writing {name} to Pickle File...')
        model.save()

if __name__=='__main__':
    np.random.seed(42)
    main()
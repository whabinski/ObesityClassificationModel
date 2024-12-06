
from Models.logistic_regression import LogisticRegression
from Models.neural_network import NeuralNetwork
from Models.svm_model import SupportVectorMachine

from Scripts.load_and_split_data import load_data, create_feature_and_target, split_data
from Scripts.preprocess_data import preprocess_features

# function to load data, seperate features and labels, and split into training and testing sets
def load_and_split(data_path):
    data = load_data(data_path)                                                                                 # load raw csv data
    features, labels = create_feature_and_target(data)                                                          # seperate features and labels
    train_features, test_features, train_labels, test_labels = split_data(features, labels, test_size=0.2)      # split into train and test sets
    return train_features, test_features, train_labels, test_labels


# main
def main():

    data_path = "Data/ObesityDataSet_raw.csv"                                               # raw dataset path
    train_features, test_features, train_labels, test_labels = load_and_split(data_path)    # load data, split into train and test sets
    print("Data successfully split into training and testing sets.")
    
    # preprocess traina nd test sets
    train_features_processed, test_features_processed, train_labels_processed, test_labels_processed = preprocess_features(train_features, test_features, train_labels, test_labels)
    print("Data successfully processed.")
    
    # Initivalize Models

    svm = SupportVectorMachine()    # initialize support vector machine model
    nn = NeuralNetwork()            # initialize neural network model
    lgrg = LogisticRegression()     # initailize logistic regression model
    models = [svm, nn, lgrg]

    #
    for model in models:
        pass

if __name__=='__main__':
    main()
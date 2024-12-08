
from Models.logistic_regression import LogisticRegression
from Models.neural_network import NeuralNetwork
from Models.svm_model import SupportVectorMachine

from Scripts.load_and_split_data import load_data, create_feature_and_target, split_data
from Scripts.preprocess_data import preprocess_features
from Scripts.evaluations import evaluate, perform_kfold, bias_variance_analysis

# function to load data, seperate features and labels, and split into training and testing sets
def load_and_split(data_path):
    data = load_data(data_path)                                                                                 # load raw csv data
    features, labels = create_feature_and_target(data)                                                          # seperate features and labels
    train_features, test_features, train_labels, test_labels = split_data(features, labels, test_size=0.2)      # split into train and test sets
    return train_features, test_features, train_labels, test_labels

def eval_normal(models, train_features_processed, train_labels_processed, test_features_processed, test_labels_processed):
    for name, model in models.items():
        print(f"-----------------------------{name}-----------------------------")
        print(f"TRAINING....................\n")
        model.train(train_features_processed, train_labels_processed)
        print(f"Predicting..................\n")
        predictions = model.predict(test_features_processed)
        print(f"Evaluating..................\n")
        evaluate(test_labels_processed, predictions)
        print(f"\n")

def eval_kfold(models, train_features_processed, train_labels_processed):
    for name, model in models.items():
        print(f"\nPerforming K-Fold Cross-Validation for {name}...")
        perform_kfold(model, train_features_processed, train_labels_processed, k=5, model_type=model)

def eval_bias_variance(models, train_features_processed, train_labels_processed, test_features_processed, test_labels_processed):
    for name, model in models.items():
        print(f"\nPerforming Bias-Variance Analysis for {name}...")
        bias_variance_analysis(model, train_features_processed, train_labels_processed, test_features_processed, test_labels_processed, model)

# main
def main():

    data_path = "Data/ObesityDataSet_raw.csv"                                               # raw dataset path
    train_features, test_features, train_labels, test_labels = load_and_split(data_path)    # load data, split into train and test sets
    print("Data successfully split into training and testing sets.")

    # preprocess traina nd test sets
    train_features_processed, test_features_processed, train_labels_processed, test_labels_processed = preprocess_features(train_features, test_features, train_labels, test_labels)
    print("Data successfully processed.")

    featureCount = train_features_processed.shape[1]
    labelCount = len(train_labels.unique())

    # Initivalize Models
    svm = SupportVectorMachine(kernel='linear', C=1)                             # initialize support vector machine model
    #nn = NeuralNetwork(feature_count=featureCount, label_count=labelCount)      # initialize neural network model
    lgrg = LogisticRegression()                                                  # initailize logistic regression model
    models = {
        'Support Vector Machine': svm, 
        #'Neural Network': nn, 
        'Logistic Regression': lgrg,
    }

    #eval_kfold(models, train_features_processed, train_labels_processed)
    #eval_bias_variance(models, train_features_processed, train_labels_processed, test_features_processed, test_labels_processed)
    eval_normal(models, train_features_processed, train_labels_processed, test_features_processed, test_labels_processed)

if __name__=='__main__':
    main()
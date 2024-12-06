
from Models.logistic_regression import LogisticRegression
from Models.neural_network import NeuralNetwork
from Models.svm_model import SupportVectorMachine

from Scripts.load_and_split import load_data, create_feature_and_target, split_data

def load_and_split(data_path):
    data = load_data(data_path)
    features, target = create_feature_and_target(data, target_col="NObeyesdad")
    features_train, features_test, target_train, target_test = split_data(features, target, test_size=0.2)
    print("Data successfully split into training and testing sets.")

def main():
    data_path = "Data/ObesityDataSet_raw.csv"
    load_and_split(data_path)
    
    #svm = SupportVectorMachine()
    #nn = NeuralNetwork()
    #lgrg = LogisticRegression()

if __name__=='__main__':
    main()
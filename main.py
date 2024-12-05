
from Models.logistic_regression import LogisticRegression
from Models.neural_network import NeuralNetwork
from Models.svm_model import SupportVectorMachine


def main():
    print('Main Function')

    svm = SupportVectorMachine()
    nn = NeuralNetwork()
    lgrg = LogisticRegression()

if __name__=='__main__':
    main()
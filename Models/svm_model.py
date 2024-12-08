import pickle
from Models.model import Model
from sklearn.svm import SVC

# encapsulate sklearn's svm model class
class SVM:
    # initialize class with kernel and regularization parameter c
    def __init__(self, kernel, C):
        self.model = SVC(kernel=kernel, C=C, probability=True)

    # function to train the svm model using train features and train labels
    def fit(self, train_features, train_labels):
        self.model.fit(train_features, train_labels)    # uses sklearns fit method: implements a quadratic programming optomization functioin to minimize hinge loss
    
    # function to make predictions on features
    def forward(self, features):
        predictions = self.model.predict(features)          # returns predicted class labels for features using sklearns predict method
        return predictions

# wrapper class for the SVM class
class SupportVectorMachine(Model):
    # initialize class with kernel and regularization parameter c
    def __init__(self, kernel, C):
        self.model = SVM(kernel=kernel, C=C)                # set model to sklearns SVM class

    # functioin to train model using train features and labels
    def train(self, train_features, train_labels):
        self.model.fit(train_features, train_labels)        # call sklearns fit method

    # predict class labels for input features
    def predict(self, features):
        predictions = self.model.forward(features)     # get predictions for test features
        return predictions
    
    # save to pickle file
    def save(self, fname):
        with open(fname, 'wb') as f:
            pickle.dump(self.model, f)

    # load from pickle
    def load(self, fname):
        with open(fname, 'rb') as file:
            self.model = pickle.load(file)

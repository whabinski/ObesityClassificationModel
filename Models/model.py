# Base Model Class
#
# This class is an interface for our different model implementations.
# The methods in this class are to be overridden by the subclasses. 
#
# Functions:
# - train: Trains model using given features and labels
# - predict: Generates predictions for given features 
# - save: Save model to pickle file
# - load: Load model from pickle file

class Model:
    def train(self, features, labels):
        pass

    def predict(self, features):
        pass

    def save(self, fname):
        pass

    def load(self, fname):
        pass

    
import pandas as pd
from sklearn.model_selection import train_test_split

# function to load raw csv data
def load_data(filepath):
    print('Loading Data ..')
    return pd.read_csv(filepath)                        # read and load the dataset      

# function to seperate features and labels
def create_feature_and_target(data):
    features = data.drop(columns=["NObeyesdad"])        # assign feature columns
    labels = data["NObeyesdad"]                         # assign label coloumn
    return features, labels

# function to split dataset into training and testing sets
def split_data(features, labels, test_size):
    # split dataset into train and test sets
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=test_size, random_state=42)
    return train_features, test_features, train_labels, test_labels

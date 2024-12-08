import pandas as pd
from Scripts.feature_engineering import add_bmi_column
from sklearn.model_selection import train_test_split

# function to load raw csv data
def load_data(filepath):
    print('Loading Data ..')
    return pd.read_csv(filepath)                        # read and load the dataset      

# engineer features to dataset
def feature_engineering(data):
    engineered_data = add_bmi_column(data)      # calculate and add bmi column to dataset
    return engineered_data

# function to seperate features and labels
def create_feature_and_target(data):
    engineered_data = feature_engineering(data)                    # add engineered features to data
    
    features = engineered_data.drop(columns=["NObeyesdad"])        # assign feature columns
    labels = engineered_data["NObeyesdad"]                         # assign label coloumn
    return features, labels

# function to split dataset into training and testing sets
def split_data(features, labels, test_size):
    # split dataset into train and test sets
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=test_size, random_state=42)
    return train_features, test_features, train_labels, test_labels

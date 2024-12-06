import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(filepath):
    return pd.read_csv(filepath)

def create_feature_and_target(data, target_col):
    features = data.drop(columns=[target_col])
    target = data[target_col]
    return features, target

def split_data(features, target, test_size):
    features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=test_size, random_state=42)
    return features_train, features_test, target_train, target_test

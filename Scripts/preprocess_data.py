# Data Preprocessing Scripts
# 
# This file contains functions for preprocessing out data.
# Functions:
# - preprocess_features

import numpy as np
import pandas as pd

from Scripts.feature_engineering import feature_selection
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler


# function to preprocess train and test sets
def preprocess_features(train_features, test_features, train_labels, test_labels):
    
    # process labels to numeric format using label encoder
    label_encoder = LabelEncoder()                                                                  # initialize label encoder
    train_labels_processed = label_encoder.fit_transform(train_labels)                              # fit and apply label encoder to training set
    test_labels_processed = label_encoder.transform(test_labels)                                    # apply label encoder to test set
    
    categorical_columns = train_features.select_dtypes(include=['object', 'category']).columns.tolist()     # dynamically define categorical columns to be processed
    numerical_columns = train_features.select_dtypes(include=['number']).columns.tolist()                   # dynamically define numerical columns to be processed

    # perform correlation analysis, and feature selection
    selected_categorical_columns, selected_numerical_columns = feature_selection(categorical_columns, numerical_columns, train_features, test_features, train_labels_processed)

    print(f"- Categorical: {len(selected_categorical_columns)} {selected_categorical_columns}")
    print(f"- Numerical: {len(selected_numerical_columns)} {selected_numerical_columns}")

    # process (nominal) categorical columns using one hot encoding
    onehot_encoder = OneHotEncoder(sparse_output=False)                                                         # initialize one hot encoding
    train_categorical_encoded = onehot_encoder.fit_transform(train_features[selected_categorical_columns])      # fit and apply encoder to training set
    test_categorical_encoded = onehot_encoder.transform(test_features[selected_categorical_columns])            # apply encoder to test set

    # process numerical columns using standard scalar
    scaler = StandardScaler()                                                                               # initialize standard scalar
    train_numerical_scaled = scaler.fit_transform(train_features[selected_numerical_columns])               # fit and apply scalar to training set
    test_numerical_scaled = scaler.transform(test_features[selected_numerical_columns])                     # apply scalar to test set

    # recombine categorical and numerical columns
    train_features_processed = np.hstack((train_categorical_encoded, train_numerical_scaled))       # combine processed categorical and numerical train set columns
    test_features_processed = np.hstack((test_categorical_encoded, test_numerical_scaled))          # combine processed categorical and numerical test set columns

    # verify preprocessing functions
    # verification(train_categorical_encoded, test_categorical_encoded, train_numerical_scaled, test_numerical_scaled, label_encoder, train_features_processed, test_features_processed, train_labels_processed, test_labels_processed)

    return train_features_processed, test_features_processed, train_labels_processed, test_labels_processed


# function to verify preprocessed outputs
# only necessary for testing purposses
def verification(train_categorical_encoded, test_categorical_encoded, train_numerical_scaled, test_numerical_scaled, label_encoder, train_features_processed, test_features_processed, train_labels_processed, test_labels_processed):
    # verify one hot encoding
    print("\nVerify one hot encoding")
    print(f"Train one hot encoded shape: {train_categorical_encoded.shape}")            # should be (x, 23) with ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS'] as categorical columns
    print(f"Test one hot encoded shape: {test_categorical_encoded.shape}")              # should be (y, 23) with ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS'] as categorical columns
    
    # verify standard scalar
    print("\nVerify standard scalar")
    print("Training set mean after scaling:", train_numerical_scaled.mean(axis=0))      # should be all really close to 0
    print("Training set std after scaling:", train_numerical_scaled.std(axis=0))        # should be all 1s
    print("Test set mean:", test_numerical_scaled.mean(axis=0))                         # should not be 0s
    print("Test set std:", test_numerical_scaled.std(axis=0))                           # should be really close to train std (1s)
    
    #verify label encoding
    print("\nVerify label encoding")
    print("Label classes:", label_encoder.classes_)                                     # should display all obesity categorical classes (7)
    print(f"Encoded training labels shape: {train_labels_processed.shape}")
    print(f"Encoded testing labels shape: {test_labels_processed.shape}")
    
    # verify final processed shapes
    print("\nVerify final processed shapes")
    print(f"Processed training features shape: {train_features_processed.shape}")       # should be (x, 31) with current categorical and numerical columns
    print(f"Processed testing features shape: {test_features_processed.shape}")         # should be (y, 31) with current categorical and numerical columns
    print(f"Encoded training labels shape: {train_labels_processed.shape}")             # should be (x,)
    print(f"Encoded testing labels shape: {test_labels_processed.shape}")               # should be (y,)
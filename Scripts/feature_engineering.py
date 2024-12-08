import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif

SHOW_GRAPHS = False

# funciton to calculate bmi and add as new column to data
def add_bmi_column(data):
    data['BMI'] = (data['Weight'] / (data['Height'] ** 2)).round(2)     # BMI = weight (kg) / height (meters) squared
    return data

# function to perform correlation analysis on numerical features
def numerical_correlation_analysis(features, labels, threshold=0.1):
    
    data = features.copy()          # copy feature set for manipulation
    data['Target'] = labels         # add target variable to all columns

    corr_matrix = data.corr()                                                           # compute correlation matrix on columns
    target_corr = corr_matrix['Target'].drop('Target').sort_values(ascending=False)     # find all correlation values with the target
    selected_features = target_corr[abs(target_corr) >= threshold].index.tolist()       # select features with correlation greater than threshold

    # visualize scores via heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(f"Correlation Matrix with Threshold = {threshold}")
    if SHOW_GRAPHS: 
        plt.show()
    
    return selected_features

# funciton to perform correlation analysis on categorical features using mutual information
def categorical_correlation_analysis(features, labels, threshold = 0.05):
    
    features_encoded = features.copy()                                      # copy feature set for manipulation
    for col in features.columns:                                            # iterate through all columns
        label_encoder = LabelEncoder()                                      # initialize label encoder for each column
        features_encoded[col] = label_encoder.fit_transform(features[col])  # convert column to numeric label

    
    mi = mutual_info_classif(features_encoded, labels)                                                  # calculate mutual information score between categorical feature and target using sklearns mutual_info_class_if method
    mi_df = pd.DataFrame({'Feature': features.columns, 'Mutual Information': mi})                       # create dataframe to store mi scores of each feature

    selected_categorical_features = mi_df[mi_df['Mutual Information'] > threshold]['Feature'].tolist()       # select features with correlation greater than threshold
    
    # visualization scores via barchart
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Mutual Information", y="Feature", data=mi_df)
    plt.title(f"Mutual Information Scores for Categorical Features with Threshold = {threshold}")
    plt.xlabel("Mutual Information Score")
    plt.ylabel("Features")
    plt.tight_layout()
    if SHOW_GRAPHS: 
        plt.show()
    
    return selected_categorical_features

# function to perform correlation analysis and return feature seletion
def feature_selection(categorical_columns, numerical_columns, train_features, test_features, train_labels):
    
    selected_numerical_features = numerical_correlation_analysis(train_features[numerical_columns], train_labels, threshold=0.1)            # correlation analysis on numerical features
    
    selected_categorical_features = categorical_correlation_analysis(train_features[categorical_columns], train_labels, threshold=0.05)     # correlation analysis on numerical features
    
    return selected_categorical_features, selected_numerical_features   # return feature columns
    
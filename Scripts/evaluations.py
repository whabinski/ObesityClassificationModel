import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

RANDOM_SEED = 1561960643

# funtion to evaluate performance of models using basic metrics; accuracy, precision, recall, f1, and confusion matrix
def evaluate_metrics(test_labels, predictions):
        accuracy = accuracy_score(test_labels, predictions)                             # calucalte accuracy using sklearns accuracy method: proportion of correctly classified samples
        precision = precision_score(test_labels, predictions, average='weighted', zero_division=0)       # calucalte precision using sklearns precision method: ratio of TP/TP+FP averaged over each class (weighted)
        recall = recall_score(test_labels, predictions, average='weighted')             # calucalte recall using sklearns recall method: ratio of TP/TP+FN averaged over each class (weighted)
        f1 = f1_score(test_labels, predictions, average='weighted')                     # calucalte f1 using sklearns f1 method: 2/ inv(precision) + inv(recall) averaged over each class (weighted)
        cm = confusion_matrix(test_labels, predictions)                                 # calucalte confusion matrix using sklearns confusion matrix method
        
        # print all metrics
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Confusion Matrix:\n{cm}")

# function to perform k fold cross validation
def evaluate_kfold(model, features, labels, folds):

    kfolds = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_SEED)                   # initialize sklearn's k-fold cross validation

    accuracy_list = []      # initialize empty list for accuracy scores per fold
    precision_list = []     # initialize empty list for precision scores per fold
    recall_list = []        # initialize empty list for recall scores per fold
    f1_list = []            # initialize empty list for f1 scores per fold

    split_indices = kfolds.split(features)      # generate train and test indicies for each new fold

    # Iterate through the indices
    for fold, (train_idx, test_idx) in enumerate(split_indices):                     # iterate through all train and test features for each fold 
        print(f"\nFold {fold + 1}/{folds}")

        # Split data into train and test for the fold
        train_features, test_features = features[train_idx], features[test_idx]
        train_labels, test_labels = labels[train_idx], labels[test_idx]

        model.train(train_features, train_labels)                                   # train the model using train features
        predictions = model.predict(test_features)                                  # make predictions on test features

        accuracy_list.append(accuracy_score(test_labels, predictions))                              # add current fold accuracy score to list
        precision_list.append(precision_score(test_labels, predictions, average='weighted', zero_division=0))        # add current fold precision score to list
        recall_list.append(recall_score(test_labels, predictions, average='weighted', zero_division=0))              # add current fold recall score to list
        f1_list.append(f1_score(test_labels, predictions, average='weighted', zero_division=0))                      # add current fold f1 score to list

    accuracy_avg = sum(accuracy_list) / len(accuracy_list)      # calculate average for accuracy
    precision_avg = sum(precision_list) / len(precision_list)   # calculate average for precision
    recall_avg = sum(recall_list) / len(recall_list)            # calculate average for recall
    f1_avg = sum(f1_list) / len(f1_list)                        # calculate average for f1

    # print the average metrics
    print("\nAverage Metrics Across All Folds:")
    print(f"Accuracy: {accuracy_avg:.4f}")
    print(f"Precision: {precision_avg:.4f}")
    print(f"Recall: {recall_avg:.4f}")
    print(f"F1-Score: {f1_avg:.4f}")

# function to evaluate bias and variance
def evaluate_bias_variance(model, features_train, labels_train, features_test, labels_test):

    model.train(features_train, labels_train)                                       # train the model on training set

    train_predictions = model.predict(features_train)                               # predict on the training set
    validation_predictions = model.predict(features_test)                           # predict on the test set

    train_error = 1 - accuracy_score(labels_train, train_predictions)               # error on the training set (number of incorrect predictions / total samples)
    validation_error = 1 - accuracy_score(labels_test, validation_predictions)     # error on the testing set 

    print(f"Training Error: {train_error:.4f}")             # print training errors
    print(f"Validation Error: {validation_error:.4f}")      # print validation errors

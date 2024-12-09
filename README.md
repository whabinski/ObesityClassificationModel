# Obesity Classification Model

Submission for McMaster University COMPSCI 4AL3 Group Project Milestone 2.

## Authors
- Eckardt, Alex - `eckardta` - [@alexeckardt](https://www.github.com/alexeckardt)
- Habinski, Wyatt - `habinskw` -[@whabinski](https://www.github.com/whabinski)
- Zhou, Eric - `zhoue16` -[@ericzz12](https://www.github.com/ericzz12)

## To Run
1. Install dependancies with `pip install -r ./requirements.txt`
2. Need python version 12.7 to properly load pickle files
3. Run `main.py`.

## Loading Pickle Files

While loading models in `test.py`, we use the same models in `training.py`. This way we can keep our static `.load()` method for simplification of the codebase.

Loading and predicting takes on the following form:
```
model = <CLASS>.load(path_to_pkl_file);
outputs = model.predict(features)
```

In the code we used a slightly more sophisticated version to allow for easy display and writing of our evaluation metrics.

## Loading Discression
The versions of Pickle, Scikit-learn, Torch and Numpy impact the ability to run this. The installed versions of each the above must match (to a certain degree) as to the ones that we compiled.
    
In the event you do not have the most current versions of each, you can run #the main file and the pickle files will be recreated, allowing you to load them in.

## Structure

;; 1. Models Folder:
;;     - svm_model.py                          support vector machine class
;;     - neural_network_model.py               neural network class
;;     - logistic_regression_model.py          logistic regression class
;;     - model.py

;; 2. Scripts Folder
;;     - evaluations.py                        evaluation / test methods
;;     - feature_engineering.py                feature engineering technique methods
;;     - load_and_slpit_data.py                load and split data into train and test methods
;;     - plots.py                              plot data via graph methods
;;     - preprocessing_data.py                 preprocess data methods

;; 3. pickle Folder                            contains saved pickle models

;; 4. Main                                     main start of program

1. Data Folder
    - `ObesityDataSet_raw.csv` - raw dataset in  CSV.
    - `train_labels.npy` - 20% Split.
    - `train_features.npy` - post pre-processed feature selection. 80% Split.
    - `test_labels.npy` - 20% Split.
    - `test_features.npy` - post pre-processed feature selection. 20% Split.

## Training Flow

1. We import the data found in `./Data/ObesityDataSet_raw.csv`

2. We preform preprocessing techniques. We then save the data to the defined files above.

3. Initialize Models

4. Train Models

5. Re-train models using  K-fold Cross Validation, and see their metrics.

6. Save each model to a pickle file, found in `./pickle/(MODEL).pkl`.

## Test Flow

1. Load Models from saved `.pkl` files.

2. Load Pre-processed data from saved `.npy` files.

3. Run their predictions

4. Evaluate Bias Variance

5. Evaluate Regular Test Metrics (Accuracy, Precision, Recall, F1 Score)

### Evaluation metrics
- K-cross fold Validation
- Bias Variance
- Regular Test Evaluation (Accuracy, Recall ...)

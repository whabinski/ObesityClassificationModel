# Obesity Classification Model

Submission for McMaster University COMPSCI 4AL3 Group Project Milestone 2.

## To Run
1. Install dependancies with `pip install -r ./requirements.txt`
2. Run `main.py`.

## Authors
- Eckardt, Alex - `eckardta` - [@alexeckardt](https://www.github.com/alexeckardt)
- Habinski, Wyatt - `habinskw` -[@whabinski](https://www.github.com/whabinski)
- Zhou, Eric - `zhoue16` -[@ericzz12](https://www.github.com/ericzz12)


## Loading Pickle Files

A class needs to be initialized. See the following code, or the function `load_models_sample()` in `main.py` for a sample code.
The inputted feature set must match the pre-processed format as created also in `main.py`.

```
def load_models_sample(test_features_processed, featureCount, labelCount):
    svm = SupportVectorMachine(kernel='linear', C=1)
    svm.load('./pickle/supportvectormachine.pkl')
    predicted = svm.predict(test_features_processed)
```


## Structure

1. Models Folder:
    - svm_model.py                          support vector machine class
    - neural_network_model.py               neural network class
    - logistic_regression_model.py          logistic regression class
    - model.py

2. Scripts
    - evaluations.py                        evaluation / test methods
    - feature_engineering.py                feature engineering technique methods
    - load_and_slpit_data.py                load and split data into train and test methods
    - plots.py                              plot data via graph methods
    - preprocessing_data.py                 preprocess data methods

3. Main                                     main start of program

## Flow

The program, found by running `./main.py`, works as follows.

1. We import the data found in `./Data/ObesityDataSet_raw.csv`

2. We preform preprocessing techniques.

3. Initialize Models

4. Perform evaluation metrics. For simiplification (as we perform K-fold Cross Validation), each metric will retrain the model.

5. Save each model to a pickle file, found in `./pickle/(MODEL).pkl`.

### Evaluation metrics
- K-cross fold Validation
- Bias Variance
- Regular Test Evaluation (Accuracy, Recall ...)

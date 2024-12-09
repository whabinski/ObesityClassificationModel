# Obesity Classification Model

Submission for McMaster University COMPSCI 4AL3 Group Project Milestone 2.

## To Run
1. Install dependancies with `pip install -r ./requirements.txt`
2. Run `main.py`.


## Authors

- Alex Eckardt - `eckardta` - [@alexeckardt](https://www.github.com/alexeckardt)
- Wyatt Habinski - `habinskw` -[@whabinski](https://www.github.com/whabinski)
- Eric Zhou - `zhoue16` -[@ericzz12](https://www.github.com/ericzz12)
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
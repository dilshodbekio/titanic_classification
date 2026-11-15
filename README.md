# titanic_classification
A simple machine learning modelt that classifies the people survived or not based on the open-source 'titanic machine learning from disaster" dataset and trained by logistic regression algoritm


# About the project below
# Titanic Survival Classification

A learning project exploring an end-to-end machine learning workflow using the Titanic dataset and Logistic Regression.

The goal is to predict whether a passenger survived (`1`) or did not survive (`0`).

## Dataset

Source: Kaggle — **Titanic: Machine Learning from Disaster**.

The labeled dataset contains 891 passengers and 12 columns. It was split into approximately 70% training, 15% validation, and 15% test data using stratified sampling.

## Workflow

- Inspected missing values and class distribution.
- Filled missing numerical values with training-set medians.
- Filled missing categorical values with the most frequent training-set category.
- Excluded PassengerId, Name, Ticket, and Cabin.
- Applied one-hot encoding to Sex and Embarked.
- Standardized numerical features.
- Trained a Logistic Regression classifier.
- Evaluated accuracy, precision, recall, F1, and the confusion matrix.
- Inspected misclassified validation examples.
- Compared against a majority-class baseline.
- Ran stratified 5-fold cross-validation on the combined training and validation data.
- Trained a final pipeline on that combined data and evaluated it on the held-out test set.

Preprocessing was fitted only on the relevant training data. During cross-validation, a scikit-learn Pipeline ensured preprocessing was learned separately within each fold.

## Results

| Evaluation | Accuracy | F1 |
|---|---:|---:|
| Majority-class baseline — validation | 61.2% | 0.0% |
| Initial Logistic Regression — validation | 85.8% | 80.8% |
| 5-fold cross-validation — mean | 80.3% | 73.2% |
| Final held-out test | 73.9% | 66.7% |

Final test precision: **64.8%**  
Final test recall: **68.6%**

Precision, recall, and F1 refer to the survived class (`1`).

The lower test accuracy shows why a single validation score should not be treated as guaranteed performance on unseen data.

## Tools

- Python
- pandas
- scikit-learn
- Google Colab

## Running the Notebook

1. Download `train.csv` from the Kaggle competition.
2. Open the notebook in Google Colab or Jupyter.
3. Place or upload `train.csv` in the notebook's working directory.
4. Run the cells from top to bottom.

## Scope

This is an educational project focused on understanding preprocessing, data leakage, classification metrics, baselines, and cross-validation. It is not a production system.

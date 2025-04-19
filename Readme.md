
Wine Quality Prediction
=======================

This project focuses on predicting the quality of wine (binary classification: good or bad)
based on its chemical properties. The dataset used includes features like acidity,
alcohol content, sulphates, and more.

The primary classifier used is **Random Forest**. To optimize its performance,
a grid search (using GridSearchCV) was performed to tune hyperparameters.
The best combination of parameters from the grid search was then used
for training and evaluation in the main code.

------------------------------------------------------------

Files Description
-----------------

Main.py
- Implements a Random Forest Classifier.
- Splits the dataset into 80% training and 20% validation.
- Evaluates performance on:
  - Training set
  - Validation set
  - Dummy test set (wine_data_test.csv)
- Also performs 10-fold stratified cross-validation.
- Prints accuracy and confusion matrices for each task.

Grid_Search.py
- Performs hyperparameter tuning using GridSearchCV from scikit-learn.
- Explores combinations of:
  - Number of estimators
  - Tree depth
  - Minimum samples per leaf/split
- Prints:
  - Best parameter combination
  - Accuracy and confusion matrix for the best model on both train and validation sets

Learning_Curve.py
- Visualizes a learning curve using the final chosen Random Forest model.
- Shows how training and validation accuracy vary with increasing training set size.

------------------------------------------------------------

Requirements
------------

- Python 3.x
- Libraries:
  - pandas
  - numpy
  - matplotlib
  - scikit-learn

Install all dependencies with:
pip install -r requirements.txt

------------------------------------------------------------

How to Run
----------

1. Main Code:
   python Main.py

2. Grid Search (optional tuning):
   python Grid_Search.py

3. Learning Curve Plot:
   python Learning_Curve.py

------------------------------------------------------------

Output
------

- Accuracy on training, validation, and dummy test set
- Confusion matrices
- Best parameters (if running grid search)
- Learning curve visualization (if running Learning_Curve.py)

------------------------------------------------------------

Additional Note
---------------

- The file `Alternate_Methods.ipynb` includes alternative classification approaches:
  - Decision Tree Classifier
  - Support Vector Machine (SVM)
- These models were implemented for comparison alongside the Random Forest used in the main submission.

------------------------------------------------------------



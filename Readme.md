
Wine Quality Prediction

This project focuses on predicting the quality of wine (binary classification: good or bad) based on its chemical properties. The dataset used includes features like acidity, alcohol content, sulphates, and more.

The primary classifier used is Random Forest. To optimize its performance, a grid search (using GridSearchCV) was performed to tune hyperparameters. The best combination of parameters from the grid search was then used for training and evaluation in the main code.

### 🔍 Unsupervised Feature Enrichment with K-Means

K-Means clustering was introduced to uncover latent patterns in the feature space. Each wine sample was assigned a cluster label, which was added as a new input feature. This enriched the dataset with unsupervised structure, offering insights into how chemical properties group together — even before labeling.

Additionally, the clusters were visualized using:
- PCA scatter plots to inspect spatial separation of clusters.
- Cluster vs. wine quality distribution to assess alignment with labels.

---

## Files Description

### Main.py
- Implements a Random Forest Classifier.
- Applies K-Means clustering to extract latent structure as a new feature.
- Splits the dataset into 80% training and 20% validation.
- Evaluates performance on:
  - Training set
  - Validation set
  - Dummy test set (wine_data_test.csv)
- Performs 10-fold stratified cross-validation.
- Generates visualizations for:
  - K-Means clustering using PCA (2D scatter plot)
  - Cluster-wise distribution of wine quality
- Prints accuracy and confusion matrices for all evaluations.

### Grid_Search.py
- Performs hyperparameter tuning using GridSearchCV from scikit-learn.
- Explores combinations of:
  - Number of estimators
  - Tree depth
  - Minimum samples per leaf/split
- Prints:
  - Best parameter combination
  - Accuracy and confusion matrix for the best model on both train and validation sets

### Learning_Curve.py
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



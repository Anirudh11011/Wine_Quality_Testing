import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


data = pd.read_csv("wine_data.csv")
X = data.drop("quality", axis=1)
y = data["quality"]


rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=5,
    min_samples_split=5,
    random_state=0
)


train_sizes, train_scores, val_scores = learning_curve(
    rf, X, y,
    cv=5,  
    scoring='accuracy',
    train_sizes=np.linspace(0.1, 1.0, 10),
    random_state=0
)


train_mean = np.mean(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_std = np.std(val_scores, axis=1)


plt.figure(figsize=(10,6))
plt.plot(train_sizes, train_mean, 'o-', label="Training Accuracy")
plt.plot(train_sizes, val_mean, 'o-', label="Validation Accuracy")
plt.title("Learning Curve (Random Forest)")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.grid()
plt.tight_layout()
plt.show()

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix


data = pd.read_csv("wine_data.csv")
X = data.drop("quality", axis=1)
y = data["quality"]


X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)


param_grid = {
    'n_estimators': [100,200,300],               
    'max_depth': [5, 10,20],                
    'min_samples_split': [5, 10],        
    'min_samples_leaf': [5, 10]          
}


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)


grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=0),
    param_grid=param_grid,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)


grid_search.fit(X_train, y_train)


print("\nBest parameters found:", grid_search.best_params_)


best_rf = grid_search.best_estimator_

y_train_pred = best_rf.predict(X_train)
y_val_pred = best_rf.predict(X_val)

train_acc = accuracy_score(y_train, y_train_pred)
val_acc = accuracy_score(y_val, y_val_pred)

print(f"\nTraining Accuracy: {train_acc:.4f}")
print("Training Confusion Matrix:\n", confusion_matrix(y_train, y_train_pred))
print(f"\nValidation Accuracy: {val_acc:.4f}")
print("Validation Confusion Matrix:\n", confusion_matrix(y_val, y_val_pred))





import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

#Task 1
data = pd.read_csv("wine_data.csv")
X = data.drop("quality", axis=1)
y = data["quality"]


X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=0
)

# I used  random forest classifier
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=5,
    min_samples_split=5,
    random_state=0
)
rf.fit(X_train, y_train)


y_train_pred = rf.predict(X_train)
train_acc = accuracy_score(y_train, y_train_pred)
train_cm = confusion_matrix(y_train, y_train_pred)


y_val_pred = rf.predict(X_val)
val_acc = accuracy_score(y_val, y_val_pred)
val_cm = confusion_matrix(y_val, y_val_pred)

print("Task 1: Train-Test Split Results")
print(f"Training Accuracy: {train_acc:.4f}")
print("Training Confusion Matrix:\n", train_cm)
print(f"Validation Accuracy: {val_acc:.4f}")
print("Validation Confusion Matrix:\n", val_cm)

#Task 2
#Tested on wine_data_test
dummy = pd.read_csv("wine_data_test.csv")
X_dummy = dummy.drop("quality", axis=1)
y_dummy = dummy["quality"]

y_dummy_pred = rf.predict(X_dummy)
dummy_acc = accuracy_score(y_dummy, y_dummy_pred)
dummy_cm = confusion_matrix(y_dummy, y_dummy_pred)

print("\nTask 2: Dummy Test Set Results")
print(f"Dummy Test Accuracy: {dummy_acc:.4f}")
print("Dummy Test Confusion Matrix:\n", dummy_cm)

# Did 10-fold stratified cross-validation fo same model with same parameters.
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
cv_accuracies = []

print("\nTask 3: 10-Fold Stratified Cross-Validation Results")
for i, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
    X_fold_train, X_fold_test = X.iloc[train_idx], X.iloc[test_idx]
    y_fold_train, y_fold_test = y.iloc[train_idx], y.iloc[test_idx]

    rf_cv = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=5,
        min_samples_split=5,
        random_state=0
    )
    rf_cv.fit(X_fold_train, y_fold_train)
    y_fold_pred = rf_cv.predict(X_fold_test)
    acc = accuracy_score(y_fold_test, y_fold_pred)
    cv_accuracies.append(acc)
    print(f"Fold {i} Accuracy: {acc:.4f}")

print(f"Average Cross-Validation Accuracy: {sum(cv_accuracies) / len(cv_accuracies):.4f}")

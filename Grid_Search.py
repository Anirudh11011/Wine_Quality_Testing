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

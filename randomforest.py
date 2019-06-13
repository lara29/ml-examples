import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from matplotlib.legend_handler import HandlerLine2D
from sklearn.metrics import classification_report

# Read the
train = pd.read_csv("titanic/train.csv")

print(train.shape)
print(train.info())

print(train.head(5))

train.drop("Cabin",axis=1,inplace=True)
train.drop("Name",axis=1,inplace=True)
train.drop("Ticket",axis=1,inplace=True)

print('mode', train['Embarked'].mode())

train['Age'] = train['Age'].fillna(train['Age'].mean())
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])

print(train.head(5))

for col in train.dtypes[train.dtypes == 'object'].index:
    for_dummy = train.pop(col)
    train = pd.concat([train, pd.get_dummies(for_dummy, prefix=col)], axis=1)

labels = train.pop('Survived')
print(train.head(5))

# Split the dataset in two equal parts
X_train, X_test, Y_train, Y_test = train_test_split(train, labels, test_size=0.25)

model = RandomForestClassifier()
model.fit(X_train, Y_train)

y_pred = model.predict(X_test)
false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, y_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)
print('roc_auc with default hyperparameters', roc_auc)

# Define hyperparameters and their ranges
n_estimators = list(range(1,32))
max_depth = list(range(1,10))
max_features = [32, 10, 1]
print()
print()
print()

# Set the parameters by cross-validation
tuned_parameters = {'n_estimators': n_estimators, 'max_depth': max_depth }
clf = GridSearchCV(model, tuned_parameters, verbose=20,  cv=3)
clf.fit(X_train, Y_train)
print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Best score found on development set:")
print()
print(clf.best_score_)
print()
print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))

print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
y_true, y_pred = Y_test, clf.predict(X_test)
print(classification_report(y_true, y_pred))

false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, y_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)
print('roc_auc with optimized hyperparameters', roc_auc)

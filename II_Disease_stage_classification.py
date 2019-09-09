# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 02:44:59 2018

@author: Amr
"""
# Code Sourse: Amr Eltelt
# AML 2018 Task2

# Load datasets
import pandas as pd
import numpy as np
X_train_raw = pd.read_csv('X_train.csv').drop(columns = ['id'])
X_test_raw = pd.read_csv('X_test.csv').drop(columns = ['id'])
Id = pd.read_csv('X_test.csv')['id']
y_train_raw = pd.read_csv('y_train.csv').drop(columns = ['id'])

# Standardize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train_raw)
X_train = scaler.transform(X_train_raw)
X_test = scaler.transform(X_test_raw)
y_train = np.ravel(y_train_raw)

# Dividing the train block into stratified train and test datasets
from sklearn.model_selection import train_test_split
X_train, X_holdout, y_train, y_holdout = train_test_split(X_train, y_train, 
                                                    test_size = 0.25, 
                                                    random_state = 1, 
                                                    stratify=y_train)

# Multiclass SVM Kernel RBF classification model selection
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

parameters = [{'gamma': [1e-4, 1e-3], 
               'C': [1, 10]}]

print("# Tuning hyper-parameters")
print()

model = GridSearchCV(SVC(kernel='rbf', class_weight='balanced',
                         decision_function_shape='ovr'), 
                    parameters, scoring='recall_macro', cv=5)
model.fit(X_train, y_train)

# Printing results
print()
print("The best parameters are %s with a score of %0.2f"
      % (model.best_params_, model.best_score_))
print()
print("Grid scores on training set:")
print()
means = model.cv_results_['mean_test_score']
stds = model.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, model.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()

# Evaluation based on holdout dataset
from sklearn.metrics import recall_score
y_true, y_pred = y_holdout, model.predict(X_holdout)
bmac = recall_score(y_true, y_pred, average='macro') 
print('True BMAC: ', bmac)

# Generate y_pred based on X_test and write into CSV file output
y_pred = pd.DataFrame(data=model.predict(X_test), columns = y_train_raw.columns)
y_file = pd.concat([Id, y_pred], axis=1, join_axes=[Id.index])
y_file.to_csv('y_pred_08.csv',index=False)


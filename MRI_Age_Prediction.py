# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 02:44:59 2018

@author: Amr
"""
# Code Sourse: Amr Eltelt
# AML 2018 Task0

# Load Libraries
import csv
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.feature_selection import SelectKBest, VarianceThreshold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import Imputer, scale, StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split

#import matplotlib.pyplot as plt
#from sklearn.preprocessing import scale, PolynomialFeatures
#from pandas.plotting import scatter_matrix
#from sklearn import model_selection
#from sklearn.metrics import classification_report
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import accuracy_score
#from sklearn.linear_model import LogisticRegression
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.naive_bayes import GaussianNB
#from sklearn.svm import SVC

# Load datasets
Xr_raw = pd.read_csv('X_train.csv').drop(columns = ['id'])
Xs_raw = pd.read_csv('X_test.csv').drop(columns = ['id'])
Id = pd.read_csv('X_test.csv')['id']
y_train = pd.read_csv('y_train.csv').drop(columns = ['id'])

# Impute mean values in empty cells
imp = Imputer(missing_values = 'NaN', strategy='median', axis=0)
Xr_imp = pd.DataFrame(data=imp.fit_transform(Xr_raw), columns=Xr_raw.columns)
Xs_imp = pd.DataFrame(data=imp.fit_transform(Xs_raw), columns=Xs_raw.columns)

# Outlier detection 
def outliers_iqr(x):
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    floor = q1 - (iqr * 1.5)
    ceiling = q3 + (iqr * 1.5)
    
    outlier_indices = list(x.index[(x < floor)|(x > ceiling)])
    outlier_values = list(x[outlier_indices])
    
    return outlier_indices, outlier_values

#Xr_lierInd, Xr_lierVal = outliers_iqr(Xr_imp['x1'])
#Xs_lierInd, Xs_lierVal = outliers_iqr(Xs_imp['x1'])

# test PCA
#pca = PCA(n_components=20)
#Xr_pca = pd.DataFrame(pca.fit_transform(Xr_imp))
#Xs_pca = pd.DataFrame(pca.fit_transform(Xs_imp))
#print (X_pca.head(5))

# feature selection Stage 1: Removing features with low variance
sel1 = VarianceThreshold(threshold=(.8 * (1 - .8)))
sel1_features = sel1.fit(Xr_imp)
indices_sel1 = sel1_features.get_support(indices=True)
colnames_sel1 = [Xr_imp.columns[i] for i in indices_sel1]

X_train = Xr_imp[colnames_sel1]
X_test = Xs_imp[colnames_sel1]

# Use feature selection
select = SelectKBest(k=670)
selected_features = select.fit(X_train, y_train)
indices_selected = selected_features.get_support(indices=True)
colnames_selected = [X_train.columns[i] for i in indices_selected]

X_train = X_train[colnames_selected]
X_test = X_test[colnames_selected]

# Data Standardization
X_train = pd.DataFrame(data=RobustScaler().fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(data=RobustScaler().fit_transform(X_test), columns=X_test.columns)
# split X_train data into train and test sets
#X_train, X_test, y_train, y_test = train_test_split(X_Std, y, train_size = 0.70, random_state = 1)


# Model
model = RidgeCV(alphas=[0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 
                0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1], cv=20) # alphas=[1e-3, 1e-2, 1e-1, 1]

model.fit(X_train, y_train)
y_pred = pd.DataFrame(data=model.predict(X_test), columns = y_train.columns)
print('R^2 Score is: ', model.score(X_train, y_train))

y_file = pd.concat([Id, y_pred], axis=1, join_axes=[Id.index])

y_file.to_csv('Task1_aeltelt_03.csv',index=False)


# =============================================================================
# lower_bound = .25
# upper_bound = .75
# quant_df = X_train_imp.quantile([lower_bound, upper_bound]) # auxiliary dataframe, it consist of quantiles computed for each column
# 
# # select outliers, i.e. values lie outside corresponding [lower_bound, upper_bound] intervals
# filtering_rule_2 = X_train_imp.apply(lambda x: (x < quant_df.loc[lower_bound, x.name]) |  (x > quant_df.loc[upper_bound, x.name]), axis=0)
# 
# =============================================================================

# Data Standardization
#X_train_scale = StandardScaler().fit_transform(X_train_imp[:,1:])

# =============================================================================
# nptrain = np.float64(train_data)
# nptest = np.float64(test_data)
# 
# Id_train, X_train, Y_train = nptrain[:,0], nptrain[:,2:12], nptrain[:,1]
# Id_test, X_test = nptest[:,0], nptest[:,1:11]
# 
# # Train the model on train set
# regr = linear_model.LinearRegression().fit(X_train,Y_train)
# print('Coefficients: \n', regr.coef_)
# 
# # Make predictions on the test set
# Y_pred = regr.predict(X_test)
# 
# # Evaluate RMSE of predicted against true value
# Y = np.mean(X_test,axis=1)
# RMSE = mean_squared_error(Y, Y_pred)**0.5
# print('Root mean squared error: ', RMSE)
# 
# final_data = np.vstack((Id_test,Y_pred)).T
# 
# with open('result_AmrEltelt.csv', 'w',newline='') as f:
#     w = csv.writer(f)
#     w.writerow(['Id', 'y'])
#     w.writerows(final_data)
# 
# #plt.scatter(X,Y, color = 'blue')
# #plt.plot(Y.iloc[1:200])
# #plt.show()
# =============================================================================

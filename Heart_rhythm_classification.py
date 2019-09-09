# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 02:44:59 2018

@author: Amr
"""
# Code Sourse: Amr Eltelt
# AML 2018 Task3

# Import Libraries
import pandas as pd
import numpy as np
from biosppy.signals import ecg
from sklearn.metrics import f1_score
from sklearn.preprocessing import scale
import datetime

# Defining Function for ECG Feature Extrcation
def ExtractECGFeature(df_X, i_TrimFirst, i_SamplingRate):
    str_Columns = ['R_mean', 'R_std', 'RR_mean', 'RR_std', 'Q_mean', 
                   'Q_std', 'S_mean', 'S_std', 'QRS_mean', 'T_mean', 
                   'T_std', 'T_stamp', 'hr_mean', 'hr_std']
    
    i_ShortRange = range(0,180,6)
    str_Columns.extend(['TMean_' + str(i) for i in i_ShortRange])
    str_Columns.extend(['TSd_' + str(i) for i in i_ShortRange])
    
    df_Features = pd.DataFrame(columns=str_Columns)
    for i_Index in range(df_X.shape[0]):
        # Preprocessing Signal
        Index = i_Index
        signal = df_X.iloc[Index, i_TrimFirst:]
        signal = signal.reset_index(drop=True)
        signal = signal.dropna()
        signal = np.array(tuple(signal))
        signal = np.array(tuple(scale(signal)))                  # <---- Check
        
        # Biosppy analysis
        out = ecg.ecg(signal=signal, sampling_rate=i_SamplingRate, show=False)
        o_ts = out['ts']
        o_filtered = out['filtered']
        o_rpeaks = out['rpeaks']
        o_templates_ts = out['templates_ts']
        o_templates = out['templates']
        o_heart_rate_ts = out['heart_rate_ts']
        o_heart_rate = out['heart_rate']
        
        # which signal to use?
        #ref_signal = o_filtered # Filtered signal selected      # <---- Check
        ref_signal = signal    # Orginal signal selected
        
        # Features Extraction
        # Beat
        beat_mean = o_templates.mean(0)
        beat_std = o_templates.std(0)
        beat_left = np.argwhere(o_templates_ts<0)
        beat_right = np.argwhere(o_templates_ts>0)
        beat_R = len(beat_left)
        beat_end = len(beat_mean)
        
        # R
        R_values = ref_signal[o_rpeaks]
        R_stamps = o_ts[o_rpeaks]
        R_mean = np.mean(R_values)
        R_std = np.std(R_values)
        
        # RR
        RR_values = np.diff(R_stamps)
        RR_mean = RR_values.mean()
        RR_std = RR_values.std()
        
        # Q
        Q_mean = np.min(beat_mean[beat_left])
        Q_index = np.argwhere(beat_mean[beat_left]==Q_mean)[0,0]
        Q_std = beat_std[Q_index]
        Q_stamp = o_templates_ts[Q_index]
        
        # S
        S_mean = np.min(beat_mean[beat_right])
        S_index = beat_R + np.argwhere(beat_mean[beat_right]==S_mean)[0,0]
        S_std = beat_std[S_index]
        S_stamp = o_templates_ts[S_index]
        
        # QRS
        QRS_mean = S_stamp - Q_stamp
        
        # T
        T_mean = np.max(beat_mean[S_index:beat_end])
        T_index = S_index + np.argwhere(beat_mean[S_index:beat_end] == T_mean)[0,0]
        T_std = beat_std[T_index]
        T_stamp = o_templates_ts[T_index]
        
        # Heart Rate
        hr_mean = o_heart_rate.mean()
        hr_std = o_heart_rate.std()
        
        # Short Ranges
        d_ShortTMean = [beat_mean[i] for i in i_ShortRange]
        d_ShortTSd = [beat_std[i] for i in i_ShortRange]
        
        
        df_Features.loc[i_Index] = [R_mean, R_std, RR_mean, RR_std, Q_mean, 
                                   Q_std, S_mean, S_std, QRS_mean, T_mean, 
                                   T_std, T_stamp, hr_mean, hr_std] + d_ShortTMean + \
                                   d_ShortTSd
    
    return df_Features

print()

# Load datasets
print("# Loading data")
X_train_raw = pd.read_csv('X_train.csv').drop(columns = ['id'])
X_test_raw = pd.read_csv('X_test.csv').drop(columns = ['id'])
Id = pd.read_csv('X_test.csv')['id']
y_train_raw = pd.read_csv('y_train.csv').drop(columns = ['id'])
print("# Done")
print()

# Extract features from ECG waveforms
print("# Extracting features")
TrimFirst = 4*300 #5*300                                        # <---- Check
X_train_ft = ExtractECGFeature(X_train_raw, TrimFirst, 300)
X_test_ft = ExtractECGFeature(X_test_raw, TrimFirst, 300)
print("# Done")
print()
#%%

# Impute median values in empty cells
from sklearn.preprocessing import Imputer
#from sklearn.impute import SimpleImputer
imp = Imputer(missing_values = 'NaN', strategy='median', axis=0).fit(X_train_ft)
X_train = imp.transform(X_train_ft)
X_test = imp.transform(X_test_ft)

# Standardize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
y_train = np.ravel(y_train_raw)


# Apply Multiclass SVM Kernel RBF classification model selection
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

parameters = [{'gamma': [0.02], 
                'C': [5]}]                            # <---- Check

print("# Tuning hyper-parameters")
model = GridSearchCV(SVC(kernel='rbf', class_weight='balanced',
                         decision_function_shape='ovr'), 
                         parameters, scoring='f1_micro', cv=5)
model.fit(X_train, y_train)
print("# Done")
print()

# =============================================================================
# parameters = [{'gamma': [0.001, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4], 
#                'C': [1, 3, 4, 5, 6, 10, 20, 50, 100]}]
# =============================================================================

# Printing results
print("Grid scores on training set:")
means = model.cv_results_['mean_test_score']
stds = model.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, model.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()
print("The best parameters are %s with a score of %0.3f"
      % (model.best_params_, model.best_score_))
print()


# Generate y_pred based on X_test and write into CSV file output
y_pred = pd.DataFrame(data=model.predict(X_test), columns = y_train_raw.columns)
y_file = pd.concat([Id, y_pred], axis=1, join_axes=[Id.index])
now = datetime.datetime.now()
y_file.to_csv('y_pred_08_'+ now.strftime("%Y%m%d%H%M") + ".csv",index=False)


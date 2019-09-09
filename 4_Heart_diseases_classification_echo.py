from get_data import get_videos_from_folder,get_target_from_csv
import os
import numpy as np
import pandas as pd
from utils import save_solution
from sklearn.metrics import roc_auc_score
import datetime

def xscale(x):
    x = np.array(x)
    mn, mx = x.min(), x.max()
    x_scaled = (x - mn) / (mx - mn)
    x_scaled = x_scaled * (2 - 1) + 1
    return x_scaled

def find_nearest(value, array, lower_limit, upper_limit):
    array = np.asarray(array)
    low = lower_limit
    up = upper_limit
    diff = array - value
    near = up
    flag = 1
    for n in diff:
        if n > low and n < near:
            near = n
            flag = 0
            result = value + near
    if flag:
        idx = (np.abs(array - value)).argmin()
        result = array[idx]
    return result, flag

def ExtractECOFeatures(df_X):
    str_Columns = ['EE_time_mean', 'EC_time_mean', 'EC_ratio_mean'] 
    #samples = 10
    #str_Columns.extend(['ECsample_' + str(i) for i in range(samples)])
    df_Features = pd.DataFrame(columns=str_Columns)
    
    for i_Index in range(df_X.shape[0]):
        vedio = df_X[i_Index]
        # Extract Blood & Muscle variation in main chamber (Left ventricle)
        # and get the cardiac periodic expansive/contractive (diastole/systole) motion
        s_blood_frame = []
        s_muscle_frame = []
        for frame in vedio:
            blood_frame = 0
            muscle_frame = 0
            for row in range(20,60):
                for col in range(40,80):
                    if frame[row,col] < 10:
                        blood_frame = blood_frame + 1
                    elif frame[row,col] > 100:
                        muscle_frame = muscle_frame + 1
            s_blood_frame.append(blood_frame)
            s_muscle_frame.append(muscle_frame)
        # Blood-Muscle ratio
        s_blood_xscale = xscale(s_blood_frame)
        s_muscle_xscale = xscale(s_muscle_frame)
        s_ratio = np.divide(s_blood_xscale,s_muscle_xscale)
        
        # Extract local extrema which are the max and min peaks 
        # of the cardiac periodic coresponding to expansive/contractive motion
        
        # max peaks
        x = s_ratio
        max_ind = np.argsort(-x)[:len(x)] #x.argsort()[-1:-len(x):-1] #
        max_best = []
        max_best.append(max_ind[0])
        for i in range(len(max_ind)):
            flag = 1
            for n in max_best:
                if abs(n - max_ind[i]) < 15:
                    flag = 0
            if flag:
                max_best.append(max_ind[i])
                if len(max_best) == 3:
                   break
        max_ind = np.array(max_best)
        # min peaks
        min_ind = np.argsort(x)[:len(x)] #x.argsort()[1:len(x):1] #
        min_best = []
        min_best.append(min_ind[0])
        for i in range(len(min_ind)):
            flag = 1
            for n in min_best:
                if abs(n - min_ind[i]) < 15:
                    flag = 0
            if flag:
                min_best.append(min_ind[i])
                if len(min_best) == 3:
                   break
        min_ind = np.array(min_best)
        # Align each max peak with the nearest min peak
        e = np.zeros((len(max_ind),3), dtype =int)
        for i in range(len(max_ind)):
            e[i,0]= max_ind[i]
            e[i,1], e[i,2] = find_nearest(max_ind[i], min_ind, 5, 40)
        e = np.sort(e, axis = 0)
        peaks1 = e
        e = pd.DataFrame(e, columns=['E','C','flag'])
        if np.sum(peaks1[:,2]) != len(peaks1):
            e.drop(e[e['flag'] == 1].index, inplace=True)
        peaks0 = np.array(e)
        
        # Extract key infromation
        # Expansion-Expansion intervals
        EE_time = np.diff(peaks1[:,0], axis=0)
        EE_time_mean = EE_time.mean()
        EE_time_std = EE_time.std()
        # Expansion-Contraction intervals
        EC_time = np.diff(peaks0[:,:], axis=1)[:,0]
        EC_time_mean = EC_time.mean()
        EC_time_std = EC_time.std()
        # Expansion-Contraction deviation
        EC_ratio = s_ratio[peaks0[:,0]] - s_ratio[peaks0[:,1]]
        EC_ratio_mean = EC_ratio.mean()
        EC_ratio_std = EC_ratio.std()
    # =============================================================================
    #         # Get samples of the beat between exp. and cont.
    #         Expansion_frame = peaks0[np.int(len(peaks0)/2),0]
    #         Contraction_frame = peaks0[np.int(len(peaks0)/2),1]
    #         single_EC = s_ratio[Expansion_frame : Contraction_frame+1]
    #         single_EC_samples = np.zeros(samples)
    #         for i in range(samples):
    #             step = np.int(np.floor(len(single_EC)/samples))
    #             single_EC_samples[i] = single_EC[i*step]
    #         #Beat_samples = [single_EC[i] for i in range(0,len(single_EC),np.int(len(single_EC)/10))]
    # =============================================================================
        
        df_Features.loc[i_Index] = [EE_time_mean, EC_time_mean, EC_ratio_mean] #+ single_EC_samples
    

    return df_Features

print()

# Load datasets
print("# Loading data") 
dir_path = "D:/Amr/OneDrive/_MyActivities/180922 ETH DAS Data Science/Advanced Machine Learning/Projects/Task4/"
train_folder = os.path.join(dir_path,"train/")
test_folder = os.path.join(dir_path,"test/")

# =============================================================================
# train_target = os.path.join(dir_path,'train_target.csv')
# my_solution_file = os.path.join(dir_path,'solution.csv')
# train_folder = "D:/Amr/OneDrive/_MyActivities/180922 ETH DAS Data Science/Advanced Machine Learning/Projects/Task4/train/"
# test_folder = "D:/Amr/OneDrive/_MyActivities/180922 ETH DAS Data Science/Advanced Machine Learning/Projects/Task4/test/"
# train_target = "D:/Amr/OneDrive/_MyActivities/180922 ETH DAS Data Science/Advanced Machine Learning/Projects/Task4/train_target.csv"
# #my_solution_file = os.path.join(dir_path,'solution.csv')
# =============================================================================

X_train_raw = get_videos_from_folder(train_folder)
X_test_raw = get_videos_from_folder(test_folder)
Id = pd.read_csv('solution.csv')['id']
y_train_raw = pd.read_csv('train_target.csv').drop(columns = ['id'])

print("# Done")
print()

# Extract features from ECG waveforms
print("# Extracting features")
X_train_ft = ExtractECOFeatures(X_train_raw)
X_test_ft = ExtractECOFeatures(X_test_raw)
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

parameters = [{'gamma': [0.001, 0.01], 
                'C': [1]}]

print("# Tuning hyper-parameters")
model = GridSearchCV(SVC(kernel='rbf', class_weight='balanced',
                         decision_function_shape='ovr'), 
                         parameters, scoring='roc_auc', cv=5)
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
y_file.to_csv('solution_02_'+ now.strftime("%Y%m%d%H%M") + ".csv",index=False)





## Disease Stage Classification Projects:
Advanced machine learning projects in the medical domian

### Age prediction based on brain MRI data 
Age regression from original brain MRI features. The following workflow is applied:

1- Imputation: Handling missing by imputing median value of given feature

2- Outlier detection: IQR method was used for outlier detection. It is a robust measure of dispersion method for labeling outliers

3- Data Standardization: Since Ridge regression is planned to be used, scaling is important. Typically this is done by removing the mean and scaling to unit variance. However, outliers can often influence the sample mean / variance in a negative way. In such cases, the median and the interquartile range often give better results

4- Feature selection: 
Since the data is in high dimension especially compared to the data rows, then a feature selection method is required to select the most important features, thus avoiding overfitting and also slow computing. Univariate feature selection from Sklearn was used

5- Model selection: 
Ridge regression is used. The ridge coefficients minimize a penalized residual sum of squares, lambda is a complexity parameter that controls the amount of shrinkage: the larger the value of, the greater the amount of shrinkage and thus the coefficients become more robust to collinearity.


### Disease classification from medical image
The following workflow have been applied in python and using Scikit Learn library:

1-	Classification model selection: 
Multiclass SVM, type C-Support Vector Classification (C-SVM) is selected with Radial Basis Function (RBF) kernel. RBF: exp(-gamma |x-x'|^2).

2-	Dealing with the imbalanced data: 
The chosen solution is to adjust class weights to be inversely proportional to class frequencies. In Sklearn library ‘balanced’ mode is selected which automatically calculates the weights adjustment

3-	Features Standardization (No feature selection): 
Since SVM RBF kernel is used, its essential to standardize (mean=0, Std=1). Thus, avoid the domination of large values (variance) on the objective function.

4-	Hyper parameters selection: 
Two paramters must be considered: (K) best features, (C) Penalty parameter, and (gamma) parameter which defines how much influence a single training example has.

C= 1 and gamma= 0.001 were selected. These values are optimal intermediate values, thus avoiding overfitting with high C or gamma and avoid underfitting with low C or gamma.
The Kfold (5) scores = [0.70324074, 0.70324074, 0.68287037, 0.68981481, 0.6662037]
Resulting to a mean score of 0.689 and variance (+/-0.028)


### Heart rhythm classification from electrocardiogram
The following workflow is applied using python with Biosppy and Scikit Learn libraries:

#### Feature extraction:
For each signal in the data set Feature extraction is done in the following sequence:
1-	Eliminate the first 4 seconds of the signal (4*300 data points)

2-	Scale the signal

3-	Use the Biosppy library ECG function to generate templates (Beats), which is the waveform around each R peak with a time window from 0.2s before the peak till 0.4s after the peak

4-	Do segmentation of the mean Beat and extract features out of the wave form, as follows:
1)	R Mean, 2)	R Std., 3)	RR Interval Mean, 4)	RR Interval Std., 5)	Q Mean, 6)	Q Std., 7)	S Mean, 8)	S Std., 9)	QRS Interval Mean, 10)	T Mean, 11)	T Std., 12)	RT Interval Mean, 13)	Heart Rate Mean, 14)	Heart Rate Std.

5-	Take a sample every 6th data point of Beat (every 0.02 sec) and get Mean and Std. which adds 2x180/6 = 60 additional features

#### Classification:
6-	Impute median values in any empty (i.e NaN) cells

7-	All features are standardized using Sklearn StandardScaler

8-	The model selected for classification is the multiclass C type SVM with Radial Basis Function (RBF) kernel. RBF: exp(-gamma |x-x'|^2)

9-	Hyper parameters selection for Penalty parameter (C) and parameter (gamma) using cross validation K=5 Folds and weight adjustments ‘Stratified cross validation’, to evaluate the best parameters based on the F-measure (F1)

10-	F1 Score = 0.745, Variance= +/-0.013 (that’s +/- 2sd). For parameters: {'C': 5, 'gamma': 0.02}


### Heart diseases diagnosis from echocardiography

####  1st solution:
1-For each frame I counted the number of Blood pixels (<10) and muscle pixels (>100) within the heart main chamber (Left ventricle) using a window range of rows 20-60 rows and columns 40-80

2-Calculated the ratio of blood/muscle per frame. The plot shows peaks corresponds to cardiac periodic expansive/contractive motion as in videos

3-Extract the peaks and calculate 3 features: EE_interval, EC interval, EC ratio difference

4-Applied for classification the multiclass C type SVM with RBF kernel. Hyper parameters using CV K=5 with roc_auc_score. Score 0.744 (+/-0.131) for C: 1, gamma: 0.001. Public is 0.627

####  2nd solution:
Neural network




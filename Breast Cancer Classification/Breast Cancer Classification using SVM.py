#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 20:56:27 2019

@author: shonjacob
"""

#Breast Cancer Classification using Support Vector Machines

# Support Vector Machine uses the concept of separating the 2 classes or more classes of output
# by maximizing the separating hyperplace between the support vectors(the edge cases of the classes)



'''
Predicting if the cancer diagnosis is benign or malignant based on several observations/features
30 features are used, examples:

  - radius (mean of distances from center to points on the perimeter)
  - texture (standard deviation of gray-scale values)
  - perimeter
  - area
  - smoothness (local variation in radius lengths)
  - compactness (perimeter^2 / area - 1.0)
  - concavity (severity of concave portions of the contour)
  - concave points (number of concave portions of the contour)
  - symmetry 
  - fractal dimension ("coastline approximation" - 1)
Datasets are linearly separable using all 30 input features

Number of Instances: 569
Class Distribution: 212 Malignant, 357 Benign
Target class:
   - Malignant
   - Benign
   
   https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns # Statistical Data Visualization

#importing the dataset
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
cancer.keys()
print(cancer['feature_names'])
print(cancer['data'].shape)
#np.c_ is used to add a column to the shape of the dataframe
df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'], ['target']))

# To view the top most rows of the dataframe
df_cancer.head()

#To show the comparison of the different target classes
sns.pairplot(df_cancer, hue = 'target', vars = ['mean radius', 'mean texture', 'mean area', 'mean perimeter', 'mean smoothness'] )
# Gives the count of targets
sns.countplot(df_cancer['target'], label = "Count") 
#Scatteer plot for a variable with target comparison
sns.scatterplot(x = 'mean area', y = 'mean smoothness', hue = 'target', data = df_cancer)
# Let's check the correlation between the variables 
# Strong correlation between the mean radius and mean perimeter, mean area and mean primeter
plt.figure(figsize=(20,10)) 
sns.heatmap(df_cancer.corr(), annot=True) 

# Let's drop the target label coloumns
X = df_cancer.drop(['target'],axis=1)

y = df_cancer['target']


#TRAINING THE MODEL
# splitting into training and test set


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=5)



#creating the  model
#SVM parameters are C and gamma
#C parameter is used to adjust the level of penalty for wrong classification. Too much and the model is overfitted. Less for smooth curve
#gamma is how much does a training set influence. less gamma ->far reach(use for generalized soln.)
#high gamma - > points close to the hyperplane has higher weights and the focus is here


from sklearn.svm import SVC 
from sklearn.metrics import classification_report, confusion_matrix

svc_model = SVC()
svc_model.fit(X_train, y_train)

#type 1 error is better than type 2 error
#evaluating the model
y_predict = svc_model.predict(X_test)
cm = confusion_matrix(y_test, y_predict)

sns.heatmap(cm, annot=True)

print(classification_report(y_test, y_predict))

#improving the model
#feature scaling (X-Xmin)/(Xmax-Xmin)
min_train = X_train.min()
range_train = (X_train - min_train).max()
X_train_scaled = (X_train - min_train)/range_train
sns.scatterplot(x = X_train['mean area'], y = X_train['mean smoothness'], hue = y_train)
sns.scatterplot(x = X_train_scaled['mean area'], y = X_train_scaled['mean smoothness'], hue = y_train)

min_test = X_test.min()
range_test = (X_test - min_test).max()
X_test_scaled = (X_test - min_test)/range_test

from sklearn.svm import SVC 
from sklearn.metrics import classification_report, confusion_matrix

svc_model = SVC()
svc_model.fit(X_train_scaled, y_train)

y_predict = svc_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_predict)
cm

sns.heatmap(cm,annot=True,fmt="d")

print(classification_report(y_test,y_predict))


#grid search will search for the best parameters, rbf = radial basis function
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']} 

from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=4)

grid.fit(X_train_scaled,y_train)

grid.best_params_

grid.best_estimator_

grid_predictions = grid.predict(X_test_scaled)

cm = confusion_matrix(y_test, grid_predictions)

sns.heatmap(cm, annot=True)

print(classification_report(y_test,grid_predictions))




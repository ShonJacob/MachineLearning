#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 06:07:40 2019

@author: shonjacob
"""

#Diabetic Hospital Readmission
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
# load the csv file
df = pd.read_csv('diabetic_data.csv')
#If we look at the IDs_mapping.csv we can see that 11,13,14,19,20,21 are related to death or hospice. We should remove these samples from the predictive model.
df = df.loc[~df.discharge_disposition_id.isin([11,13,14,19,20,21])]
df['OUTPUT_LABEL'] = (df.readmitted == '<30').astype('int')
def calc_prevalence(y_actual):
    return (sum(y_actual)/len(y_actual))
print('Prevalence:%.3f'%calc_prevalence(df['OUTPUT_LABEL'].values))
# for each column
'''
for c in list(df.columns):
    
    # get a list of unique values
    n = df[c].unique()
    
    # if number of unique values is less than 30, print the values. Otherwise print the number of unique values
    if len(n)<30:
        print(c)
        print(n)
    else:
        print(c + ': ' +str(len(n)) + ' unique values')
'''
'''
From analysis of the columns, we can see there are a mix of categorical (non-numeric) and numerical data. A few things to point out,

- encounter_id and patient_nbr: these are just identifiers and not useful variables
- age and weight: are categorical in this data set
- admission_type_id,discharge_disposition_id,admission_source_id: are numerical here, but are IDs (see IDs_mapping). They should be considered categorical. 
- examide and citoglipton only have 1 value, so we will not use these variables
- diag1, diag2, diag3 - are categorical and have a lot of values. We will not use these as part of this project, but you could group these ICD codes to reduce the dimension. We will use number_diagnoses to capture some of this information. 
- medical_speciality - has many categorical variables, so we should consider this when making features. 
'''

# replace ? with nan
df = df.replace('?',np.nan)

cols_num = ['time_in_hospital','num_lab_procedures', 'num_procedures', 'num_medications',
       'number_outpatient', 'number_emergency', 'number_inpatient','number_diagnoses']

df[cols_num].isnull().sum()
cols_cat = ['race', 'gender', 
       'max_glu_serum', 'A1Cresult',
       'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
       'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
       'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
       'tolazamide', 'insulin',
       'glyburide-metformin', 'glipizide-metformin',
       'glimepiride-pioglitazone', 'metformin-rosiglitazone',
       'metformin-pioglitazone', 'change', 'diabetesMed','payer_code']
df[cols_cat].isnull().sum()
df['race'] = df['race'].fillna('UNK')
df['payer_code'] = df['payer_code'].fillna('UNK')
df['medical_specialty'] = df['medical_specialty'].fillna('UNK')

print('Number medical specialty:', df.medical_specialty.nunique())
df.groupby('medical_specialty').size().sort_values(ascending = False)

'''
We can see that most of them are unknown and that the count drops off pretty quickly. 
We don't want to add 73 new variables since some of them only have a few samples.
 As an alternative, we can create a new variable that only has 11 options (the top 10 specialities and then an other category).
 Obviously, there are other options for bucketing, but this is one of the easiest methods.
'''

top_10 = ['UNK','InternalMedicine','Emergency/Trauma',\
          'Family/GeneralPractice', 'Cardiology','Surgery-General' ,\
          'Nephrology','Orthopedics',\
          'Orthopedics-Reconstructive','Radiologist']

# make a new column with duplicated data
df['med_spec'] = df['medical_specialty'].copy()

# replace all specialties not in top 10 with 'Other' category
df.loc[~df.med_spec.isin(top_10),'med_spec'] = 'Other'
df.groupby('med_spec').size()

cols_cat_num = ['admission_type_id', 'discharge_disposition_id', 'admission_source_id']

df[cols_cat_num] = df[cols_cat_num].astype('str')
df_cat = pd.get_dummies(df[cols_cat + cols_cat_num + ['med_spec']],drop_first = True)
df = pd.concat([df,df_cat], axis = 1)
cols_all_cat = list(df_cat.columns)

age_id = {'[0-10)':0, 
          '[10-20)':10, 
          '[20-30)':20, 
          '[30-40)':30, 
          '[40-50)':40, 
          '[50-60)':50,
          '[60-70)':60, 
          '[70-80)':70, 
          '[80-90)':80, 
          '[90-100)':90}
df['age_group'] = df.age.replace(age_id)

df.weight.notnull().sum()
df['has_weight'] = df.weight.notnull().astype('int')
cols_extra = ['age_group','has_weight']

print('Total number of features:', len(cols_num + cols_all_cat + cols_extra))
print('Numerical Features:',len(cols_num))
print('Categorical Features:',len(cols_all_cat))
print('Extra features:',len(cols_extra))

col2use = cols_num + cols_all_cat + cols_extra
df_data = df[col2use + ['OUTPUT_LABEL']]


# shuffle the samples
df_data = df_data.sample(n = len(df_data), random_state = 42)
df_data = df_data.reset_index(drop = True)

# Save 30% of the data as validation and test data 
df_valid_test=df_data.sample(frac=0.30,random_state=42)
print('Split size: %.3f'%(len(df_valid_test)/len(df_data)))

#splitting into validaiton and test set
df_test = df_valid_test.sample(frac = 0.5, random_state = 42)
df_valid = df_valid_test.drop(df_test.index)

# use the rest of the data as training data
df_train_all=df_data.drop(df_valid_test.index)


print('Test prevalence(n = %d):%.3f'%(len(df_test),calc_prevalence(df_test.OUTPUT_LABEL.values)))
print('Valid prevalence(n = %d):%.3f'%(len(df_valid),calc_prevalence(df_valid.OUTPUT_LABEL.values)))
print('Train all prevalence(n = %d):%.3f'%(len(df_train_all), calc_prevalence(df_train_all.OUTPUT_LABEL.values)))


print('all samples (n = %d)'%len(df_data))
assert len(df_data) == (len(df_test)+len(df_valid)+len(df_train_all)),'math didnt work'

'''
At this point, you might say, drop the training data into a predictive model and see the outcome. However, if we do this, it is possible that we will get back a model that is 89% accurate. Great! Good job! But wait, we never catch any of the readmissions (recall= 0%). How can this happen?

What is happening is that we have an imbalanced dataset where there are much more negatives than positives, so the model might just assigns all samples as negative.

Typically, it is better to balance the data in some way to give the positives more weight. There are 3 strategies that are typically utilized:

- sub-sample the more dominant class: use a random subset of the negatives
- over-sample the imbalanced class: use the same positive samples multiple times
- create synthetic positive data

Usually, you will want to use the latter two methods if you only have a handful of positive cases. Since we have a few thousand positive cases, let's use the sub-sample approach. Here, we will create a balanced training data set that has 50% positive and 50% negative. You can also play with this ratio to see if you can get an improvement.
'''

# split the training data into positive and negative
rows_pos = df_train_all.OUTPUT_LABEL == 1
df_train_pos = df_train_all.loc[rows_pos]
df_train_neg = df_train_all.loc[~rows_pos]

# merge the balanced data
#take all the positive rows and a sample of negative rows equal to length of positive
df_train = pd.concat([df_train_pos, df_train_neg.sample(n = len(df_train_pos), random_state = 42)],axis = 0)
#testdata = df_train_neg.sample(n = len(df_train_pos), random_state = 42)
# shuffle the order of training samples 
df_train = df_train.sample(n = len(df_train), random_state = 42).reset_index(drop = True)

print('Train balanced prevalence(n = %d):%.3f'%(len(df_train), calc_prevalence(df_train.OUTPUT_LABEL.values)))


X_train = df_train[col2use].values
X_train_all = df_train_all[col2use].values
X_valid = df_valid[col2use].values
X_test = df_test[col2use].values

y_train = df_train['OUTPUT_LABEL'].values
y_valid = df_valid['OUTPUT_LABEL'].values
y_test = df_test['OUTPUT_LABEL'].values

print('Training All shapes:',X_train_all.shape)
print('Training shapes:',X_train.shape, y_train.shape)
print('Validation shapes:',X_valid.shape, y_valid.shape)
print('Test shapes:',X_test.shape, y_test.shape)

from sklearn.preprocessing import StandardScaler

scaler  = StandardScaler()
scaler.fit(X_train_all)


import pickle
scalerfile = 'scaler.sav'
pickle.dump(scaler, open(scalerfile, 'wb'))


# load it back
scaler = pickle.load(open(scalerfile, 'rb'))


X_train_tf = scaler.transform(X_train)
X_valid_tf = scaler.transform(X_valid)
X_test_tf = scaler.transform(X_test)

from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
def calc_specificity(y_actual, y_pred, thresh):
    # calculates specificity
    return sum((y_pred < thresh) & (y_actual == 0)) /sum(y_actual ==0)

def print_report(y_actual, y_pred, thresh):
    
    auc = roc_auc_score(y_actual, y_pred)
    accuracy = accuracy_score(y_actual, (y_pred > thresh))
    recall = recall_score(y_actual, (y_pred > thresh))
    precision = precision_score(y_actual, (y_pred > thresh))
    specificity = calc_specificity(y_actual, y_pred, thresh)
    print('AUC:%.3f'%auc)
    print('accuracy:%.3f'%accuracy)
    print('recall:%.3f'%recall)
    print('precision:%.3f'%precision)
    print('specificity:%.3f'%specificity)
    print('prevalence:%.3f'%calc_prevalence(y_actual))
    print(' ')
    return auc, accuracy, recall, precision, specificity

thresh = 0.5

# k-nearest neighbors
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors = 100)
knn.fit(X_train_tf, y_train)

y_train_preds = knn.predict_proba(X_train_tf)[:,1]
y_valid_preds = knn.predict_proba(X_valid_tf)[:,1]
y_test_preds = knn.predict_proba(X_test_tf)[:,1]

print('KNN')
print('Training:')
knn_train_auc, knn_train_accuracy, knn_train_recall, \
    knn_train_precision, knn_train_specificity = print_report(y_train,y_train_preds, thresh)
print('Validation:')
knn_valid_auc, knn_valid_accuracy, knn_valid_recall, \
    knn_valid_precision, knn_valid_specificity = print_report(y_valid,y_valid_preds, thresh)
print('Testing:')
knn_test_auc, knn_test_accuracy, knn_test_recall, \
    knn_test_precision, knn_test_specificity = print_report(y_test,y_test_preds, thresh)
    
#Random forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(max_depth = 6, random_state = 42)

# Train the model on training data
rf.fit(X_train_tf, y_train);

y_train_predsrf = rf.predict_proba(X_train_tf)[:,1]
y_valid_predsrf = rf.predict_proba(X_valid_tf)[:,1]
y_test_predsrf = rf.predict_proba(X_test_tf)[:,1]

print('Random Forest')
print('Training:')
rf_train_auc, rf_train_accuracy, rf_train_recall, \
    rf_train_precision, rf_train_specificity = print_report(y_train,y_train_predsrf, thresh)
print('Validation:')
rf_valid_auc, rf_valid_accuracy, rf_valid_recall, \
    rf_valid_precision, rf_valid_specificity = print_report(y_valid,y_valid_predsrf, thresh)
print('Testing:')
rf_test_auc, rf_test_accuracy, rf_test_recall, \
    rf_test_precision, rf_test_specificity = print_report(y_test,y_test_predsrf, thresh)

#Naive Bayes
from sklearn.naive_bayes import GaussianNB
NBclf = GaussianNB()
NBclf.fit(X_train_tf, y_train)

y_train_predsNB = NBclf.predict_proba(X_train_tf)[:,1]
y_valid_predsNB = NBclf.predict_proba(X_valid_tf)[:,1]
y_test_predsNB = NBclf.predict_proba(X_test_tf)[:,1]

print('Naive Bayes')
print('Training:')
nb_train_auc, nb_train_accuracy, nb_train_recall, \
    nb_train_precision, nb_train_specificity = print_report(y_train,y_train_predsNB, thresh)
print('Validation:')
nb_valid_auc, nb_valid_accuracy, nb_valid_recall, \
    nb_valid_precision, nb_valid_specificity = print_report(y_valid,y_valid_predsNB, thresh)
print('Testing:')
nb_test_auc, nb_test_accuracy, nb_test_recall, \
    nb_test_precision, nb_test_specificity = print_report(y_test,y_test_predsNB, thresh)


# SVC
from sklearn.svm import SVC
clf = SVC(gamma='auto')
clf.fit(X_train_tf, y_train) 
SVCclf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=True, random_state=None, shrinking=True,
    tol=0.001, verbose=False)    
SVCclf.fit(X_train_tf, y_train)

y_train_predsSVC = SVCclf.predict_proba(X_train_tf)[:,1]
y_valid_predsSVC = SVCclf.predict_proba(X_valid_tf)[:,1]
y_test_predsSVC = SVCclf.predict_proba(X_test_tf)[:,1]

print('Support Vector Classification')
print('Training:')
svc_train_auc, svc_train_accuracy, svc_train_recall, \
    svc_train_precision, svc_train_specificity = print_report(y_train,y_train_predsSVC, thresh)
print('Validation:')
svc_valid_auc, svc_valid_accuracy, svc_valid_recall, \
    svc_valid_precision, svc_valid_specificity = print_report(y_valid,y_valid_predsSVC, thresh)
print('Testing:')
svc_test_auc, svc_test_accuracy, svc_test_recall, \
    svc_test_precision, svc_test_specificity = print_report(y_test,y_test_predsSVC, thresh)



# logistic regression
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(random_state = 42)
lr.fit(X_train_tf, y_train)

y_train_predsLR = lr.predict_proba(X_train_tf)[:,1]
y_valid_predsLR = lr.predict_proba(X_valid_tf)[:,1]
y_test_predsLR = lr.predict_proba(X_test_tf)[:,1]

print('Logistic Regression')
print('Training:')
lr_train_auc, lr_train_accuracy, lr_train_recall, \
    lr_train_precision, lr_train_specificity = print_report(y_train,y_train_predsLR, thresh)
print('Validation:')
lr_valid_auc, lr_valid_accuracy, lr_valid_recall, \
    lr_valid_precision, lr_valid_specificity = print_report(y_valid,y_valid_predsLR, thresh)
print('Testing:')
lr_test_auc, lr_test_accuracy, lr_test_recall, \
    lr_test_precision, lr_test_specificity = print_report(y_test,y_test_predsLR, thresh)


#Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier
sgdc=SGDClassifier(loss = 'log',alpha = 0.1,random_state = 42)
sgdc.fit(X_train_tf, y_train)

y_train_predsSGD = sgdc.predict_proba(X_train_tf)[:,1]
y_valid_predsSGD = sgdc.predict_proba(X_valid_tf)[:,1]
y_test_predsSGD = sgdc.predict_proba(X_test_tf)[:,1]

print('Stochastic Gradient Descent')
print('Training:')
sgdc_train_auc, sgdc_train_accuracy, sgdc_train_recall, sgdc_train_precision, sgdc_train_specificity =print_report(y_train,y_train_predsSGD, thresh)
print('Validation:')
sgdc_valid_auc, sgdc_valid_accuracy, sgdc_valid_recall, sgdc_valid_precision, sgdc_valid_specificity = print_report(y_valid,y_valid_predsSGD, thresh)
print('Testing:')
sgdc_test_auc, sgdc_test_accuracy, sgdc_test_recall, \
    sgdc_test_precision, sgdc_test_specificity = print_report(y_test,y_test_predsSGD, thresh)



# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth = 10, random_state = 42)
tree.fit(X_train_tf, y_train)

y_train_predsDT = tree.predict_proba(X_train_tf)[:,1]
y_valid_predsDT = tree.predict_proba(X_valid_tf)[:,1]
y_test_predsDT = tree.predict_proba(X_test_tf)[:,1]

print('Decision Tree')
print('Training:')
tree_train_auc, tree_train_accuracy, tree_train_recall, tree_train_precision, tree_train_specificity =print_report(y_train,y_train_predsDT, thresh)
print('Validation:')
tree_valid_auc, tree_valid_accuracy, tree_valid_recall, tree_valid_precision, tree_valid_specificity = print_report(y_valid,y_valid_predsDT, thresh)
print('Testing:')
tree_test_auc, tree_test_accuracy, tree_test_recall, \
    tree_test_precision, tree_test_specificity = print_report(y_test,y_test_predsDT, thresh)



#Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
gbc =GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
     max_depth=3, random_state=42)
gbc.fit(X_train_tf, y_train)

y_train_predsGBC = gbc.predict_proba(X_train_tf)[:,1]
y_valid_predsGBC = gbc.predict_proba(X_valid_tf)[:,1]
y_test_predsGBC = gbc.predict_proba(X_test_tf)[:,1]


print('Gradient Boosting Classifier')
print('Training:')
gbc_train_auc, gbc_train_accuracy, gbc_train_recall, gbc_train_precision, gbc_train_specificity = print_report(y_train,y_train_predsGBC, thresh)
print('Validation:')
gbc_valid_auc, gbc_valid_accuracy, gbc_valid_recall, gbc_valid_precision, gbc_valid_specificity = print_report(y_valid,y_valid_predsGBC, thresh)
print('Testing:')
gbc_test_auc, gbc_test_accuracy, gbc_test_recall, \
    gbc_test_precision, gbc_test_specificity = print_report(y_test,y_test_predsGBC, thresh)


#PART 2 - CREATING THE ANN
import keras
from keras.models import  Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold

classifier = Sequential()
#rectifier activation function for hidden layer and sigmoid activation fn. for output layer
#try with softmax to see output for output layer
#later on we need to do crossvalidation to determine the number of nodes in hidden layer through experimentation

#adding the input layer and first hidden layer
classifier.add(Dense(output_dim = 10,init = 'random_uniform',activation = 'tanh', input_dim = 143))
#LEARNING RATE
#adding the second hidden layer
#classifier.add(Dense(output_dim = 92,init = 'uniform',activation = 'relu', input_dim = 92))

#adding the output layer, softmax activation fn. for more than 2 categories output
classifier.add(Dense(output_dim = 1,init = 'random_uniform',activation = 'sigmoid', input_dim = 10))

#compiling the ANN, optimizer is stochastic gradient descend(garrett mclaws function) logarithmic error ,binary_crossentropy for binary output,categorical_crossentropy
#metrics is how to improve performance
#adaptive moment estimation
#Cross-Entropy. Cross-entropy loss, or log loss, measures the performance of a classification model whose output is a probability value between 0 and 1. 
SGDOptimizer = keras.optimizers.SGD(lr=0.2, momentum=0.3, decay=0.0, nesterov=False)
classifier.compile(optimizer = SGDOptimizer, loss = 'mean_squared_error', metrics = ['accuracy'])

#Fitting the ANN to the Training set
#batchsize is number of observations after which we update weights
#find an optimal method to find batchsize and number of epochs
classifier.fit(X_train_tf, y_train, batch_size = None, nb_epoch = 300)

# Predicting the Test set results
y_train_predsANN = classifier.predict_proba(X_train_tf)[:,0]
y_valid_predsANN = classifier.predict_proba(X_valid_tf)[:,0]
y_test_predsANN = classifier.predict_proba(X_test_tf)[:,0]

print('Artificial Neural Network')
print('Training:')
ann_train_auc, ann_train_accuracy, ann_train_recall, ann_train_precision, ann_train_specificity = print_report(y_train,y_train_predsANN, thresh)
print('Validation:')
ann_valid_auc, ann_valid_accuracy, ann_valid_recall, ann_valid_precision, ann_valid_specificity = print_report(y_valid,y_valid_predsANN, thresh)
print('Testing:')
ann_test_auc, ann_test_accuracy, ann_test_recall, \
    ann_test_precision, ann_test_specificity = print_report(y_test,y_test_predsANN, thresh)



# ANN with cross validation

def make_ANN():
    import keras
    from keras.models import  Sequential
    from keras.layers import Dense
    #from keras.wrappers.scikit_learn import KerasClassifier
    #from sklearn.model_selection import StratifiedKFold
    
    classifier = Sequential()
    #rectifier activation function for hidden layer and sigmoid activation fn. for output layer
    #try with softmax to see output for output layer
    #later on we need to do crossvalidation to determine the number of nodes in hidden layer through experimentation
    
    #adding the input layer and first hidden layer
    classifier.add(Dense(output_dim = 10,init = 'random_uniform',activation = 'tanh', input_dim = 143))
    #LEARNING RATE
    #adding the second hidden layer
    #classifier.add(Dense(output_dim = 10,init = 'uniform',activation = 'relu', input_dim = 10))
    
    #adding the output layer, softmax activation fn. for more than 2 categories output
    classifier.add(Dense(output_dim = 1,init = 'random_uniform',activation = 'tanh', input_dim = 10))
    #compiling the ANN, optimizer is stochastic gradient descend(garrett mclaws function) logarithmic error ,lbinary_crossentropy for binary output,categorical_crossentropy
    #metrics is how to improve performance
    #adaptive moment estimation
    #Cross-Entropy. Cross-entropy loss, or log loss, measures the performance of a classification model whose output is a probability value between 0 and 1. 
    #AdamOptimizer = keras.optimizers.Adam(lr=0.2, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    #Stochastic Gradient Descent
    SGDOptimizer = keras.optimizers.SGD(lr=0.2, momentum=0.3, decay=0.0, nesterov=False)
    classifier.compile(optimizer = SGDOptimizer, loss = 'mean_squared_error', metrics = ['accuracy'])
    return classifier
#Fitting the ANN to the Training set
#batchsize is number of observations after which we update weights
#find an optimal method to find batchsize and number of epochs
#classifier.fit(X_train, y_train, batch_size = 500, nb_epoch = 300)

#CROSS VALIDATION
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
neural_network = KerasClassifier(build_fn=make_ANN, 
                                 epochs=300, 
                                 batch_size=None,#None, 
                                 verbose=1)
#verbose: 0 for no logging to stdout, 1 for progress bar logging, 2 for one log line per epoch."
X_cross = pd.concat([pd.DataFrame(X_train_tf), pd.DataFrame(X_valid_tf), pd.DataFrame(X_test_tf)], axis = 0)
y_cross = pd.concat([pd.DataFrame(y_train), pd.DataFrame(y_valid), pd.DataFrame(y_test)], axis = 0)
X_cross = X_cross.sample(frac=1)
y_cross = y_cross.sample(frac=1)
accuracies = cross_val_score(estimator = neural_network,
                             X = X_cross,
                             y = y_cross,
                             cv = 10,
                             n_jobs = None)
mean = accuracies.mean()
variance = accuracies.var()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 09:51:56 2019

@author: shonjacob
"""

import numpy as np
import pandas as pd
#from matplotlib import pyplot as plt
import random as rd
from sklearn import cross_validation
#from sklearn import preprocessing
#from sklearn.neural_network import MLPClassifier
# Loading the data, shuffling and preprocessing it

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
df = df.replace('Unknown/Invalid', np.NaN)
df.dropna(inplace=True)
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
#df['has_weight'] = df.weight.notnull().astype('int')
cols_extra = ['age_group']
#cols_extra = ['age_group','has_weight']

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
df_test = df_valid_test.sample(frac = 1, random_state = 42)
#df_valid = df_valid_test.drop(df_test.index)

# use the rest of the data as training data
df_train_all=df_data.drop(df_valid_test.index)


print('Test prevalence(n = %d):%.3f'%(len(df_test),calc_prevalence(df_test.OUTPUT_LABEL.values)))
#print('Valid prevalence(n = %d):%.3f'%(len(df_valid),calc_prevalence(df_valid.OUTPUT_LABEL.values)))
print('Train all prevalence(n = %d):%.3f'%(len(df_train_all), calc_prevalence(df_train_all.OUTPUT_LABEL.values)))


#print('all samples (n = %d)'%len(df_data))
#assert len(df_data) == (len(df_test)+len(df_valid)+len(df_train_all)),'math didnt work'

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
#X_valid = df_valid[col2use].values
X_test = df_test[col2use].values

y_train = df_train['OUTPUT_LABEL'].values
#y_valid = df_valid['OUTPUT_LABEL'].values
y_test = df_test['OUTPUT_LABEL'].values

print('Training All shapes:',X_train_all.shape)
print('Training shapes:',X_train.shape, y_train.shape)
#print('Validation shapes:',X_valid.shape, y_valid.shape)
print('Test shapes:',X_test.shape, y_test.shape)


#Scaling
from sklearn.preprocessing import StandardScaler

scaler  = StandardScaler()
scaler.fit(X_train_all)


import pickle
scalerfile = 'scaler.sav'
pickle.dump(scaler, open(scalerfile, 'wb'))


# load it back
scaler = pickle.load(open(scalerfile, 'rb'))


X_train_tf = scaler.transform(X_train)
#X_valid_tf = scaler.transform(X_valid)
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

X = np.concatenate((X_train_tf, X_test_tf), axis =0)
Y = np.concatenate((y_train, y_test), axis = 0)




#GENETIC ALGORITHM



Cnt1 = len(X)
# 10, 4, 0, 1, 0, 1, 1, 0, 0

### The solver has no crossover because mutation is enough, since it only has two values
### VARIABLES ###
### VARIABLES ###
p_c_con = 1 # Probability of crossover
p_c_comb = 0.3 # Probability of crossover for integers
p_m_con = 0.4 # Probability of mutation
p_m_comb = 0.4 # Probability of mutation for integers
p_m_solver = 0.3 # Probability of mutation for the solver
K = 3 # For Tournament selection
pop = 5 # Population per generation
gen = 5 # Number of generations
ii2 = 2 # Number of K

### VARIABLES ###
### VARIABLES ###


### Combinatorial ###
UB_X1 = 25 # X1, Number of Neurons
LB_X1 = 1
UB_X2 = 2 # X2, Number of Hidden Layers
LB_X2 = 1



### Continuous ###
# Where the first 15 represent X3 and the second 15 represent X4
XY0 = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1,
                1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1]) # Initial solution

Init_Sol = XY0.copy()

n_list = np.empty((0,len(XY0)+2))
n_list_ST = np.empty((0,len(XY0)+2))
Sol_Here = np.empty((0,len(XY0)+2))
Sol_Here_ST = np.empty((0,1))

Solver_Type = ['sgd']

#generating random population
for i in range(pop): # Shuffles the elements in the vector n times and stores them
    ST = rd.choice(Solver_Type)
    X1 = rd.randrange(3,25,1)#Step size
    X2 = rd.randrange(1,2,1)
    rd.shuffle(XY0)
    Sol_Here = np.append((X1,X2),XY0)
    n_list_ST = np.append(n_list_ST,ST)
    n_list = np.vstack((n_list,Sol_Here))


# Calculating fitness value

# X3 = Learning Rate
a_X3 = 0.01 # Lower bound of X
b_X3 = 0.7 # Upper bound of X
l_X3 = (len(XY0)/2) # Length of Chrom. X

# X4 = Momentum
a_X4 = 0.01 # Lower bound of Y
b_X4 = 0.7 # Upper bound of Y
l_X4 = (len(XY0)/2) # Length of Chrom. Y


Precision_X = (b_X3 - a_X3)/((2**l_X3)-1)

Precision_Y = (b_X4 - a_X4)/((2**l_X4)-1)


z = 0
t = 1
X0_num_Sum = 0

for i in range(len(XY0)//2):
    X0_num = XY0[-t]*(2**z)
    X0_num_Sum += X0_num
    t = t+1
    z = z+1


p = 0
u = 1 + (len(XY0)//2)
Y0_num_Sum = 0

for j in range(len(XY0)//2):
    Y0_num = XY0[-u]*(2**p)
    Y0_num_Sum += Y0_num
    u = u+1
    p = p+1


Decoded_X3 = round((X0_num_Sum * Precision_X) + a_X3, 2)
Decoded_X4 = round((Y0_num_Sum * Precision_Y) + a_X4, 2)


print()
print("Decoded_X3:",Decoded_X3)
print("Decoded_X4:",Decoded_X4)


For_Plotting_the_Best = np.empty((0,len(Sol_Here)+1))

One_Final_Guy = np.empty((0,len(Sol_Here)+2))
One_Final_Guy_Final = []

# to keep track of all the populations,the mutated children,
Min_for_all_Generations_for_Mut_1 = np.empty((0,len(Sol_Here)+1))
Min_for_all_Generations_for_Mut_2 = np.empty((0,len(Sol_Here)+1))
#the +2 is to keep track of which generation
Min_for_all_Generations_for_Mut_1_1 = np.empty((0,len(Sol_Here)+2))
Min_for_all_Generations_for_Mut_2_2 = np.empty((0,len(Sol_Here)+2))

Min_for_all_Generations_for_Mut_1_1_1 = np.empty((0,len(Sol_Here)+2))
Min_for_all_Generations_for_Mut_2_2_2 = np.empty((0,len(Sol_Here)+2))


Generation = 1 


for i in range(gen):
    
    
    New_Population = np.empty((0,len(Sol_Here))) # Saving the new generation
    
    All_in_Generation_X_1 = np.empty((0,len(Sol_Here)+1))
    All_in_Generation_X_2 = np.empty((0,len(Sol_Here)+1))
    
    Min_in_Generation_X_1 = []
    Min_in_Generation_X_2 = []
    
    
    Save_Best_in_Generation_X = np.empty((0,len(Sol_Here)+1))
    #for elitisim, substitute worst with best
    Final_Best_in_Generation_X = []
    Worst_Best_in_Generation_X = []
    
    
    print()
    print("--> GENERATION: #",Generation)
    
    Family = 1

    for j in range(int(pop/2)): # range(int(pop/2))
            
        print()
        print("--> FAMILY: #",Family)
              
            
        # Tournament Selection
        # Tournament Selection
        # Tournament Selection
        
        Parents = np.empty((0,len(Sol_Here)))
        
        for i in range(2):
            
            Battle_Troops = []
            
            Warrior_1_index = np.random.randint(0,len(n_list)) #3
            Warrior_2_index = np.random.randint(0,len(n_list)) #5
            Warrior_3_index = np.random.randint(0,len(n_list))
            
            while Warrior_1_index == Warrior_2_index:
                Warrior_1_index = np.random.randint(0,len(n_list))
            while Warrior_2_index == Warrior_3_index:
                    Warrior_3_index = np.random.randint(0,len(n_list))
            while Warrior_1_index == Warrior_3_index:
                    Warrior_3_index = np.random.randint(0,len(n_list))
            
            Warrior_1 = n_list[Warrior_1_index,:]
            Warrior_2 = n_list[Warrior_2_index,:]
            Warrior_3 = n_list[Warrior_3_index,:]
            
            
            Battle_Troops = [Warrior_1,Warrior_2,Warrior_3]
            
            
            # For Warrior #1
            
            W1_Comb_1 = Warrior_1[0]
            W1_Comb_1 = int(W1_Comb_1)
            W1_Comb_2 = Warrior_1[1]
            W1_Comb_2 = int(W1_Comb_2)
            
            W1_Con = Warrior_1[2:]
            
            X0_num_Sum_W1 = 0
            Y0_num_Sum_W1 = 0
            
            z = 0
            t = 1
            OF_So_Far_W1 = 0
            
            for i in range(len(XY0)//2):
                X0_num_W1 = W1_Con[-t]*(2**z)
                X0_num_Sum_W1 += X0_num_W1
                t = t+1
                z = z+1
                
            p = 0
            u = 1 + (len(XY0)//2)
            
            for j in range(len(XY0)//2):
                Y0_num_W1 = W1_Con[-u]*(2**p)
                Y0_num_Sum_W1 += Y0_num_W1
                u = u+1
                p = p+1
                
        
            Decoded_X3_W1 = round((X0_num_Sum_W1 * Precision_X) + a_X3, 2)
            Decoded_X4_W1 = round((Y0_num_Sum_W1 * Precision_Y) + a_X4, 2)
            '''
            print()
            print("X0_num_W1:",X0_num_W1)
            print("Y0_num_W1:",Y0_num_W1)
            print("X0_num_Sum_W1:",X0_num_Sum_W1)
            print("Y0_num_Sum_W1:",Y0_num_Sum_W1)
            
            print("Decoded_X_W1:", Decoded_X_W1)
            print("Decoded_Y_W1:",Decoded_Y_W1)
            '''  
            
            Emp_3 = 0

            kf = cross_validation.KFold(Cnt1, n_folds=ii2)
            
            for train_index, test_index in kf:
                X_train, X_test = X[train_index], X[test_index]
                Y_train, Y_test = Y[train_index], Y[test_index]
                
                Hid_Lay = ()
    
                # Objective Function
                
                #8,8,8
                
                for i in range(W1_Comb_2):
                    Hid_Lay = Hid_Lay + (W1_Comb_1,)
                  
                #model1 = MLPClassifier(activation='tanh',hidden_layer_sizes=Hid_Lay,
                 #                      learning_rate_init=Decoded_X3_W1,momentum=Decoded_X4_W1)
                import keras
                SGD = keras.optimizers.SGD(lr=Decoded_X3_W1, momentum=Decoded_X4_W1, decay=0.0, nesterov=False)
                #model1 = MLPClassifier(activation='tanh',hidden_layer_sizes=Hid_Lay,
                                   #learning_rate_init=Decoded_X3_MC_1,momentum=Decoded_X4_MC_1, solver = 'sgd')
                def make_ANN(optimizer='sgd', activation = 'tanh', init = 'uniform', lr = Decoded_X3_W1, momentum = Decoded_X4_W1, neurons=W1_Comb_1):
                    from keras.models import  Sequential
                    from keras.layers import Dense
                    from keras.optimizers import SGD
 
                    classifier = Sequential()
    
                    classifier.add(Dense(output_dim = neurons,init = init,activation = activation, input_dim = 141))
   
                    classifier.add(Dense(output_dim = 1,init = init,activation = activation, input_dim = neurons))
    

                    optimizer = SGD(lr = Decoded_X3_W1, momentum = Decoded_X4_W1) 
                    classifier.compile(optimizer = optimizer, loss = 'mean_squared_error', metrics = ['accuracy'])
                    return classifier

                from keras.wrappers.scikit_learn import KerasClassifier
                model1 = KerasClassifier(build_fn=make_ANN, verbose = 2)
                
                model1.fit(X_train, Y_train)
                PL1=model1.predict(X_test)
                AC1=model1.score(X_test,Y_test)
            
                OF_So_Far_3 = 1-(model1.score(X_test,Y_test))
                
                
                Emp_3 += OF_So_Far_3
            
            OF_So_Far_W1 = Emp_3/ii2
            
            '''
            print()
            print("OF_So_Far_W1:",(1-OF_So_Far_W1))
            '''
            Prize_Warrior_1 = OF_So_Far_W1
            
            
            # For Warrior #2
            
            W2_Comb_1 = Warrior_2[0]
            W2_Comb_1 = int(W2_Comb_1)
            W2_Comb_2 = Warrior_2[1]
            W2_Comb_2 = int(W2_Comb_2)
            
            W2_Con = Warrior_2[2:]
            
            X0_num_Sum_W2 = 0
            Y0_num_Sum_W2 = 0
            
            z = 0
            t = 1
            OF_So_Far_W2 = 0
        
            for i in range(len(XY0)//2):
                X0_num_W2 = W2_Con[-t]*(2**z)
                X0_num_Sum_W2 += X0_num_W2
                t = t+1
                z = z+1
                
            p = 0
            u = 1 + (len(XY0)//2)
            
            for j in range(len(XY0)//2):
                Y0_num_W2 = W2_Con[-u]*(2**p)
                Y0_num_Sum_W2 += Y0_num_W2
                u = u+1
                p = p+1
                
        
            Decoded_X3_W2 = round((X0_num_Sum_W2 * Precision_X) + a_X3, 2)
            Decoded_X4_W2 = round((Y0_num_Sum_W2 * Precision_Y) + a_X4, 2)
            '''
            print()
            print("X0_num_W2:",X0_num_W2)
            print("Y0_num_W2:",Y0_num_W2)
            print("X0_num_Sum_W2:",X0_num_Sum_W2)
            print("Y0_num_Sum_W2:",Y0_num_Sum_W2)
            
            print("Decoded_X_W2:", Decoded_X_W2)
            print("Decoded_Y_W2:",Decoded_Y_W2)
            '''  
            
            Emp_4 = 0

            kf = cross_validation.KFold(Cnt1, n_folds=ii2)
            
            for train_index, test_index in kf:
                X_train, X_test = X[train_index], X[test_index]
                Y_train, Y_test = Y[train_index], Y[test_index]
                
                Hid_Lay = ()
    
                # Objective Function
                
                for i in range(W2_Comb_2):
                    Hid_Lay = Hid_Lay + (W2_Comb_1,)
                  
                #model1 = MLPClassifier(activation='tanh',hidden_layer_sizes=Hid_Lay,
                                       #learning_rate_init=Decoded_X3_W2,momentum=Decoded_X4_W2)
               # import keras
                SGD = keras.optimizers.SGD(lr=Decoded_X3_W2, momentum=Decoded_X4_W2, decay=0.0, nesterov=False)
                #model1 = MLPClassifier(activation='tanh',hidden_layer_sizes=Hid_Lay,
                                   #learning_rate_init=Decoded_X3_MC_1,momentum=Decoded_X4_MC_1, solver = 'sgd')
                def make_ANN(optimizer='sgd', activation = 'tanh', init = 'uniform', lr = Decoded_X3_W2, momentum = Decoded_X4_W2, neurons=W2_Comb_1):
                    from keras.models import  Sequential
                    from keras.layers import Dense
                    from keras.optimizers import SGD
 
                    classifier = Sequential()
    
                    classifier.add(Dense(output_dim = neurons,init = init,activation = activation, input_dim = 141))
   
                    classifier.add(Dense(output_dim = 1,init = init,activation = activation, input_dim = neurons))
    

                    optimizer = SGD(lr = Decoded_X3_W2, momentum = Decoded_X4_W2) 
                    classifier.compile(optimizer = optimizer, loss = 'mean_squared_error', metrics = ['accuracy'])
                    return classifier

            #    from keras.wrappers.scikit_learn import KerasClassifier
                model1 = KerasClassifier(build_fn=make_ANN, verbose = 2)
                model1.fit(X_train, Y_train)
                PL1=model1.predict(X_test)
                AC1=model1.score(X_test,Y_test)
            
                OF_So_Far_4 = 1-(model1.score(X_test,Y_test))
                
                
                Emp_4 += OF_So_Far_4
            
            OF_So_Far_W2 = Emp_4/ii2
            
            '''
            print()
            print("OF_So_Far_W2:",(1-OF_So_Far_W2))
            '''
            Prize_Warrior_2 = OF_So_Far_W2
            
            
            # For Warrior #3
            
            W3_Comb_1 = Warrior_3[0]
            W3_Comb_1 = int(W3_Comb_1)
            W3_Comb_2 = Warrior_3[1]
            W3_Comb_2 = int(W3_Comb_2)
            
            W3_Con = Warrior_3[2:]
            
            X0_num_Sum_W3 = 0
            Y0_num_Sum_W3 = 0
            
            z = 0
            t = 1
            OF_So_Far_W3 = 0
        
            for i in range(len(XY0)//2):
                X0_num_W3 = W3_Con[-t]*(2**z)
                X0_num_Sum_W3 += X0_num_W3
                t = t+1
                z = z+1
                
            p = 0
            u = 1 + (len(XY0)//2)
            
            for j in range(len(XY0)//2):
                Y0_num_W3 = W3_Con[-u]*(2**p)
                Y0_num_Sum_W3 += Y0_num_W3
                u = u+1
                p = p+1
                
        
            Decoded_X3_W3 = round((X0_num_Sum_W3 * Precision_X) + a_X3, 2)
            Decoded_X4_W3 = round((Y0_num_Sum_W3 * Precision_Y) + a_X4, 2)
            '''
            print()
            print("X0_num_W3:",X0_num_W3)
            print("Y0_num_W3:",Y0_num_W3)
            print("X0_num_Sum_W3:",X0_num_Sum_W3)
            print("Y0_num_Sum_W3:",Y0_num_Sum_W3)
            
            print("Decoded_X_W3:", Decoded_X_W3)
            print("Decoded_Y_W3:",Decoded_Y_W3)
            '''  
            
            Emp_5 = 0

            kf = cross_validation.KFold(Cnt1, n_folds=ii2)
            
            for train_index, test_index in kf:
                X_train, X_test = X[train_index], X[test_index]
                Y_train, Y_test = Y[train_index], Y[test_index]
                
                Hid_Lay = ()
    
                # Objective Function
                
                for i in range(W3_Comb_2):
                    Hid_Lay = Hid_Lay + (W3_Comb_1,)
                  
                #model1 = MLPClassifier(activation='tanh',hidden_layer_sizes=Hid_Lay,
                                       #learning_rate_init=Decoded_X3_W3,momentum=Decoded_X4_W3)
                SGD = keras.optimizers.SGD(lr=Decoded_X3_W3, momentum=Decoded_X4_W3, decay=0.0, nesterov=False)
                #model1 = MLPClassifier(activation='tanh',hidden_layer_sizes=Hid_Lay,
                                   #learning_rate_init=Decoded_X3_MC_1,momentum=Decoded_X4_MC_1, solver = 'sgd')
                def make_ANN(optimizer='sgd', activation = 'tanh', init = 'uniform', lr = Decoded_X3_W3, momentum = Decoded_X4_W3, neurons=W3_Comb_1):
                    from keras.models import  Sequential
                    from keras.layers import Dense
                    from keras.optimizers import SGD
 
                    classifier = Sequential()
    
                    classifier.add(Dense(output_dim = neurons,init = init,activation = activation, input_dim = 141))
   
                    classifier.add(Dense(output_dim = 1,init = init,activation = activation, input_dim = neurons))
    

                    optimizer = SGD(lr = Decoded_X3_W3, momentum = Decoded_X4_W3) 
                    classifier.compile(optimizer = optimizer, loss = 'mean_squared_error', metrics = ['accuracy'])
                    return classifier

            #    from keras.wrappers.scikit_learn import KerasClassifier
                model1 = KerasClassifier(build_fn=make_ANN, verbose = 2)
                model1.fit(X_train, Y_train)
                PL1=model1.predict(X_test)
                AC1=model1.score(X_test,Y_test)
            
                OF_So_Far_5 = 1-(model1.score(X_test,Y_test))
                
                
                Emp_5 += OF_So_Far_5
            
            OF_So_Far_W3 = Emp_5/ii2
            
            '''
            print()
            print("OF_So_Far_W3:",(1-OF_So_Far_W3))
            '''
            Prize_Warrior_3 = OF_So_Far_W3 
            
            '''
            print()
            print("OF_So_Far_W3:",(1-OF_So_Far_W3))
            '''
            Prize_Warrior_3 = OF_So_Far_W3
            
            
            if Prize_Warrior_1 == min(Prize_Warrior_1,Prize_Warrior_2,Prize_Warrior_3):
                Winner = Warrior_1
                Winner_str = "Warrior_1"
                Prize = Prize_Warrior_1
            elif Prize_Warrior_2 == min(Prize_Warrior_1,Prize_Warrior_2,Prize_Warrior_3):
                Winner = Warrior_2
                Winner_str = "Warrior_2"
                Prize = Prize_Warrior_2
            else:
                Winner = Warrior_3
                Winner_str = "Warrior_3"
                Prize = Prize_Warrior_3
            '''
            print()
            print("Prize_Warrior_1:",Prize_Warrior_1)
            print("Prize_Warrior_2:",Prize_Warrior_2)
            print("Prize_Warrior_3:",Prize_Warrior_3)
            print("Winner is:",Winner_str,"at:",Prize)
            '''
            Parents = np.vstack((Parents,Winner))
        '''
        print()
        print("Parents:",Parents)
        '''
        
        Parent_1 = Parents[0]
        Parent_2 = Parents[1]
        
        
        # Crossover
        # Crossover
        # Crossover
        
        
        Child_1 = np.empty((0,len(Sol_Here)))
        Child_2 = np.empty((0,len(Sol_Here)))
        
        
        # Crossover the Integers
        # Combinatorial
        
        # For X1
        # For X1
        Ran_CO_1 = np.random.rand()
        if Ran_CO_1 < p_c_comb:
            # For X1
            Int_X1_1 = Parent_2[0]
            Int_X1_2 = Parent_1[0]
        else:
            # For X1
            Int_X1_1 = Parent_1[0]
            Int_X1_2 = Parent_2[0]
        
        # For X2
        # For X2
        Ran_CO_1 = np.random.rand()
        if Ran_CO_1 < p_c_comb:
            # For X2
            Int_X2_1 = Parent_2[1]
            Int_X2_2 = Parent_1[1]
        else:
            # For X2
            Int_X2_1 = Parent_1[1]
            Int_X2_2 = Parent_2[1]
        
        
        # Continuous
        # Where to crossover
        # Two-point crossover
        
        Ran_CO_1 = np.random.rand()
        
        if Ran_CO_1 < p_c_con:
        
            Cr_1 = np.random.randint(2,len(Sol_Here))
            Cr_2 = np.random.randint(2,len(Sol_Here))
                
            while Cr_1 == Cr_2:
                Cr_2 = np.random.randint(2,len(Sol_Here))
            
            if Cr_1 < Cr_2:
                
                Cr_2 = Cr_2 + 1
                
                Copy_1 = Parent_1[2:]
                Mid_Seg_1 = Parent_1[Cr_1:Cr_2]
                
                Copy_2 = Parent_2[2:]
                Mid_Seg_2 = Parent_2[Cr_1:Cr_2]
                
                First_Seg_1 = Parent_1[2:Cr_1]
                Second_Seg_1 = Parent_1[Cr_2:]
                
                First_Seg_2 = Parent_2[2:Cr_1]
                Second_Seg_2 = Parent_2[Cr_2:]
                
                Child_1 = np.concatenate((First_Seg_1,Mid_Seg_2,Second_Seg_1))
                Child_2 = np.concatenate((First_Seg_2,Mid_Seg_1,Second_Seg_2))
                
                Child_1 = np.insert(Child_1,0,(Int_X1_1,Int_X2_1))###
                Child_2 = np.insert(Child_2,0,(Int_X1_2,Int_X2_2))
            else:
                
                Cr_1 = Cr_1 + 1
                
                Copy_1 = Parent_1[2:]
                Mid_Seg_1 = Parent_1[Cr_2:Cr_1]
                
                Copy_2 = Parent_2[2:]
                Mid_Seg_2 = Parent_2[Cr_2:Cr_1]
                
                First_Seg_1 = Parent_1[2:Cr_2]
                Second_Seg_1 = Parent_1[Cr_1:]
                
                First_Seg_2 = Parent_2[2:Cr_2]
                Second_Seg_2 = Parent_2[Cr_1:]
                
                Child_1 = np.concatenate((First_Seg_1,Mid_Seg_2,Second_Seg_1))
                Child_2 = np.concatenate((First_Seg_2,Mid_Seg_1,Second_Seg_2))
                Child_1 = np.insert(Child_1,0,(Int_X1_1,Int_X2_1))###
                Child_2 = np.insert(Child_2,0,(Int_X1_2,Int_X2_2))
        else:
            
            Child_1 = Parent_1[2:]
            Child_2 = Parent_2[2:]
            '''
            print()
            print("Child #1 here2:",Child_1)
            print("Child #2 here2:",Child_2)
            '''
            Child_1 = np.insert(Child_1,0,(Int_X1_1,Int_X2_1))###
            Child_2 = np.insert(Child_2,0,(Int_X1_2,Int_X2_2))
            
            
        '''    
        print()
        print("Child #1:",Child_1)
        print("Child #2:",Child_2)
        '''
        '''
        print("Cr_1:",Cr_1)
        print("Cr_2:",Cr_2)
        print("Parent #1:",Parent_1)
        print("Parent #2:",Parent_2)
        print("Child #1:",Child_1)
        print("Child #2:",Child_2)
        '''
        
        
        # Mutation Child #1
        # Mutation Child #1
        # Mutation Child #1
        
        Mutated_Child_1 = []
        
        
        # Combinatorial
        
        # For X1
        # For X1
        Ran_M1_1 = np.random.rand()
        if Ran_M1_1 < p_m_comb:
            Ran_M1_2 = np.random.rand()
            if Ran_M1_2 >= 0.5:
                if Child_1[0] == UB_X1:
                    C_X1_M1 = Child_1[0]
                elif Child_1[0] == LB_X1:
                    C_X1_M1 = Child_1[0]
                else:
                    C_X1_M1 = Child_1[0] + 2
            else:
                if Child_1[0] == UB_X1:
                    C_X1_M1 = Child_1[0]
                elif Child_1[0] == LB_X1:
                    C_X1_M1 = Child_1[0]
                else:
                    C_X1_M1 = Child_1[0] - 2
        else:
            C_X1_M1 = Child_1[0]
        
        # For X2
        # For X2
        Ran_M1_3 = np.random.rand()
        if Ran_M1_3 < p_m_comb:
            Ran_M1_4 = np.random.rand()
            if Ran_M1_4 >= 0.5:
                if Child_1[1] == UB_X2:
                    C_X2_M1 = Child_1[1]
                elif Child_1[1] == LB_X2:
                    C_X2_M1 = Child_1[1]
                else:
                    C_X2_M1 = Child_1[1] + 1
            else:
                if Child_1[1] == UB_X2:
                    C_X2_M1 = Child_1[1]
                elif Child_1[1] == LB_X2:
                    C_X2_M1 = Child_1[1]
                else:
                    C_X2_M1 = Child_1[1] - 1
        else:
            C_X2_M1 = Child_1[1]
           
        
        # Continuous
        
        t = 0
        
        Child_1n = Child_1[2:]
        
        for i in Child_1n:
        
            Ran_Mut_1 = np.random.rand() # Probablity to Mutate
            
            if Ran_Mut_1 < p_m_con: # If probablity to mutate is less than p_m, then mutate
                
                if Child_1n[t] == 0:
                    Child_1n[t] = 1
                else:
                    Child_1n[t] = 0
                t = t+1
            
                Mutated_Child_1n = Child_1n
                
            else:
                Mutated_Child_1n = Child_1n
        
        Mutated_Child_1 = np.insert(Mutated_Child_1n,0,(C_X1_M1,C_X2_M1))
        
        '''
        print()
        print("Mutated_Child #1:",Mutated_Child_1)
        '''
        
        # Mutation Child #2
        # Mutation Child #2
        # Mutation Child #2
        
        Mutated_Child_2 = []
        
        
        # Combinatorial
        
        # For X1
        # For X1
        Ran_M2_1 = np.random.rand()
        if Ran_M2_1 < p_m_comb:
            Ran_M2_2 = np.random.rand()
            if Ran_M2_2 >= 0.5:
                if Child_2[0] == UB_X1:
                    C_X1_M2 = Child_1[0]
                elif Child_2[0] == LB_X1:
                    C_X1_M2 = Child_2[0]
                else:
                    C_X1_M2 = Child_2[0] + 2
            else:
                if Child_2[0] == UB_X1:
                    C_X1_M2 = Child_2[0]
                elif Child_1[0] == LB_X1:
                    C_X1_M2 = Child_2[0]
                else:
                    C_X1_M2 = Child_1[0] - 2
        else:
            C_X1_M2 = Child_2[0]
        
        # For X2
        # For X2
        Ran_M2_3 = np.random.rand()
        if Ran_M2_3 < p_m_comb:
            Ran_M2_4 = np.random.rand()
            if Ran_M2_4 >= 0.5:
                if Child_2[1] == UB_X2:
                    C_X2_M2 = Child_2[1]
                elif Child_2[1] == LB_X2:
                    C_X2_M2 = Child_2[1]
                else:
                    C_X2_M2 = Child_2[1] + 1
            else:
                if Child_2[1] == UB_X2:
                    C_X2_M2 = Child_2[1]
                elif Child_2[1] == LB_X2:
                    C_X2_M2 = Child_2[1]
                else:
                    C_X2_M2 = Child_2[1] - 1
        else:
            C_X2_M2 = Child_2[1]
           
        
        # Continuous
        
        t = 0
        
        Child_2n = Child_2[2:]
        
        for i in Child_2n:
        
            Ran_Mut_2 = np.random.rand() # Probablity to Mutate
            
            if Ran_Mut_2 < p_m_con: # If probablity to mutate is less than p_m, then mutate
                
                if Child_2n[t] == 0:
                    Child_2n[t] = 1
                else:
                    Child_2n[t] = 0
                t = t+1
            
                Mutated_Child_2n = Child_2n
                
            else:
                Mutated_Child_2n = Child_2n
        
        Mutated_Child_2 = np.insert(Mutated_Child_2n,0,(C_X1_M2,C_X2_M2))
        
        '''
        print()
        print("Mutated_Child #2:",Mutated_Child_2)
        '''
        
        # Calculate fitness values of mutated children
        
        fit_val_muted_children = np.empty((0,2))
        
        
        # For mutated child #1
        
        MC_1_Comb_1 = Mutated_Child_1[0]
        MC_1_Comb_1 = int(MC_1_Comb_1)
        MC_1_Comb_2 = Mutated_Child_1[1]
        MC_1_Comb_2 = int(MC_1_Comb_2)
        
        MC_1_Con = Mutated_Child_1[2:]
        
        X0_num_Sum_MC_1 = 0
        Y0_num_Sum_MC_1 = 0
        
        z = 0
        t = 1
        OF_So_Far_MC_1 = 0
    
        for i in range(len(XY0)//2):
            X0_num_MC_1 = MC_1_Con[-t]*(2**z)
            X0_num_Sum_MC_1 += X0_num_MC_1
            t = t+1
            z = z+1
            
        p = 0
        u = 1 + (len(XY0)//2)
        
        for j in range(len(XY0)//2):
            Y0_num_MC_1 = MC_1_Con[-u]*(2**p)
            Y0_num_Sum_MC_1 += Y0_num_MC_1
            u = u+1
            p = p+1
            
        Decoded_X3_MC_1 = round((X0_num_Sum_MC_1 * Precision_X) + a_X3, 2)
        Decoded_X4_MC_1 = round((Y0_num_Sum_MC_1 * Precision_Y) + a_X4, 2)
        
        
        Emp_6 = 0

        kf = cross_validation.KFold(Cnt1, n_folds=ii2)
        
        for train_index, test_index in kf:
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]
            
            Hid_Lay = ()

            # Objective Function
            
            for i in range(MC_1_Comb_2):
                Hid_Lay = Hid_Lay + (MC_1_Comb_1,)
            
            #import keras
            SGD = keras.optimizers.SGD(lr=Decoded_X3_MC_1, momentum=Decoded_X4_MC_1, decay=0.0, nesterov=False)
            #model1 = MLPClassifier(activation='tanh',hidden_layer_sizes=Hid_Lay,
                                   #learning_rate_init=Decoded_X3_MC_1,momentum=Decoded_X4_MC_1, solver = 'sgd')
            def make_ANN(optimizer='sgd', activation = 'tanh', init = 'uniform', lr = Decoded_X3_MC_1, momentum = Decoded_X4_MC_1, neurons=MC_1_Comb_1):
                from keras.models import  Sequential
                from keras.layers import Dense
                from keras.optimizers import SGD
 
                classifier = Sequential()
    
                classifier.add(Dense(output_dim = neurons,init = init,activation = activation, input_dim = 141))
   
                classifier.add(Dense(output_dim = 1,init = init,activation = activation, input_dim = neurons))
    

                optimizer = SGD(lr = Decoded_X3_MC_1, momentum = Decoded_X4_MC_1) 
                classifier.compile(optimizer = optimizer, loss = 'mean_squared_error', metrics = ['accuracy'])
                return classifier

           # from keras.wrappers.scikit_learn import KerasClassifier
            model1 = KerasClassifier(build_fn=make_ANN, verbose = 2)


            model1.fit(X_train, Y_train)
            PL1=model1.predict(X_test)
            AC1=model1.score(X_test,Y_test)
        
            OF_So_Far_6 = 1-(model1.score(X_test,Y_test))
            
            Emp_6 += OF_So_Far_6
        
        OF_So_Far_MC_1 = Emp_6/ii2
        
        
        # For mutated child #2
        
        MC_2_Comb_1 = Mutated_Child_2[0]
        MC_2_Comb_1 = int(MC_2_Comb_1)
        MC_2_Comb_2 = Mutated_Child_2[1]
        MC_2_Comb_2 = int(MC_2_Comb_2)
        
        MC_2_Con = Mutated_Child_2[2:]
        
        X0_num_Sum_MC_2 = 0
        Y0_num_Sum_MC_2 = 0
        
        z = 0
        t = 1
        OF_So_Far_MC_2 = 0
    
        for i in range(len(XY0)//2):
            X0_num_MC_2 = MC_2_Con[-t]*(2**z)
            X0_num_Sum_MC_2 += X0_num_MC_2
            t = t+1
            z = z+1
            
        p = 0
        u = 1 + (len(XY0)//2)
        
        for j in range(len(XY0)//2):
            Y0_num_MC_2 = MC_2_Con[-u]*(2**p)
            Y0_num_Sum_MC_2 += Y0_num_MC_2
            u = u+1
            p = p+1
            
        Decoded_X3_MC_2 = round((X0_num_Sum_MC_2 * Precision_X) + a_X3, 2)
        Decoded_X4_MC_2 = round((Y0_num_Sum_MC_2 * Precision_Y) + a_X4, 2)
        
        
        Emp_7 = 0

        kf = cross_validation.KFold(Cnt1, n_folds=ii2)
        
        for train_index, test_index in kf:
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]
            
            Hid_Lay = ()

            # Objective Function
            
            for i in range(MC_2_Comb_2):
                Hid_Lay = Hid_Lay + (MC_2_Comb_1,)
              
            #model1 = MLPClassifier(activation='tanh',hidden_layer_sizes=Hid_Lay,
                                   #learning_rate_init=Decoded_X3_MC_2,momentum=Decoded_X4_MC_2)
            #import keras
            SGD = keras.optimizers.SGD(lr=Decoded_X3_MC_2, momentum=Decoded_X4_MC_2, decay=0.0, nesterov=False)
            def make_ANN(optimizer='sgd', activation = 'tanh', init = 'uniform', lr = Decoded_X3_MC_2, momentum = Decoded_X4_MC_2, neurons=MC_2_Comb_1):
                from keras.models import  Sequential
                from keras.layers import Dense
                from keras.optimizers import SGD
 
                classifier = Sequential()
    
                classifier.add(Dense(output_dim = neurons,init = init,activation = activation, input_dim = 141))
   
                classifier.add(Dense(output_dim = 1,init = init,activation = activation, input_dim = neurons))
    

                optimizer = SGD(lr = Decoded_X3_MC_2, momentum = Decoded_X4_MC_2) 
                classifier.compile(optimizer = optimizer, loss = 'mean_squared_error', metrics = ['accuracy'])
                return classifier

            #from keras.wrappers.scikit_learn import KerasClassifier
            model1 = KerasClassifier(build_fn=make_ANN, verbose = 2)
            model1.fit(X_train, Y_train)
            PL1=model1.predict(X_test)
            AC1=model1.score(X_test,Y_test)
        
            OF_So_Far_7 = 1-(model1.score(X_test,Y_test))
            
            Emp_7 += OF_So_Far_7
        
        OF_So_Far_MC_2 = Emp_7/ii2
        
        '''
        print()
        print("FV at Mutated Child #1 at Gen #",Generation,":", OF_So_Far_MC_1)
        print("FV at Mutated Child #2 at Gen #",Generation,":", OF_So_Far_MC_2)
        '''   
              
        All_in_Generation_X_1_1_temp = Mutated_Child_1[np.newaxis]
        All_in_Generation_X_1_1 = np.column_stack((OF_So_Far_MC_1, All_in_Generation_X_1_1_temp))
        
        
        All_in_Generation_X_2_1_temp = Mutated_Child_2[np.newaxis]
        All_in_Generation_X_2_1 = np.column_stack((OF_So_Far_MC_2, All_in_Generation_X_2_1_temp))
        
        
        All_in_Generation_X_1 = np.vstack((All_in_Generation_X_1,All_in_Generation_X_1_1))
        All_in_Generation_X_2 = np.vstack((All_in_Generation_X_2,All_in_Generation_X_2_1))
        
        
        Save_Best_in_Generation_X = np.vstack((All_in_Generation_X_1,All_in_Generation_X_2))
        
        
        New_Population = np.vstack((New_Population,Mutated_Child_1,Mutated_Child_2))
        
        t = 0
        R_1 = []
        for i in All_in_Generation_X_1:
            
            if (All_in_Generation_X_1[t,:1]) <= min(All_in_Generation_X_1[:,:1]):
                R_1 = All_in_Generation_X_1[t,:]
            t = t+1
            
        
        Min_in_Generation_X_1 = R_1[np.newaxis]
        '''
        print()
        print("Min_in_Generation_X_1:",Min_in_Generation_X_1)
        '''
        t = 0
        R_2 = []
        for i in All_in_Generation_X_2:
            
            if (All_in_Generation_X_2[t,:1]) <= min(All_in_Generation_X_2[:,:1]):
                R_2 = All_in_Generation_X_2[t,:]
            t = t+1
                
        
        Min_in_Generation_X_2 = R_2[np.newaxis]
        '''
        print()
        print("Min_in_Generation_X_2:",Min_in_Generation_X_2)
        '''
        
        Family = Family+1
    
    '''
    print()
    print("New_Population Before:",New_Population)
    '''
    t = 0
    R_Final = []
    for i in Save_Best_in_Generation_X:
        
        if (Save_Best_in_Generation_X[t,:1]) <= min(Save_Best_in_Generation_X[:,:1]):
            R_Final = Save_Best_in_Generation_X[t,:]
        t = t+1
            
    
    Final_Best_in_Generation_X = R_Final[np.newaxis]
    '''
    print()
    print("Final_Best_in_Generation_X:",Final_Best_in_Generation_X)
    '''
    t = 0
    R_22_Final = []
    for i in Save_Best_in_Generation_X:
        
        if (Save_Best_in_Generation_X[t,:1]) >= max(Save_Best_in_Generation_X[:,:1]):
            R_22_Final = Save_Best_in_Generation_X[t,:]
        t = t+1
            
    
    Worst_Best_in_Generation_X = R_22_Final[np.newaxis]
    '''
    print()
    print("Worst_Best_in_Generation_X:",Worst_Best_in_Generation_X)
    '''
    
    # Elitism, the best in the generation lives
    # Elitism, the best in the generation lives
    # Elitism, the best in the generation lives
    
    Darwin_Guy = Final_Best_in_Generation_X[:]
    Not_So_Darwin_Guy = Worst_Best_in_Generation_X[:]
    
    
    Darwin_Guy = Darwin_Guy[0:,1:].tolist()
    Not_So_Darwin_Guy = Not_So_Darwin_Guy[0:,1:].tolist()
    
    '''
    print()
    print("Darwin_Guy:",Darwin_Guy)
    print("Not_So_Darwin_Guy:",Not_So_Darwin_Guy)
    
    print()
    print("Before:",New_Population)
    print()
    '''
    Best_1 = np.where((New_Population == Darwin_Guy).all(axis=1))
    Worst_1 = np.where((New_Population == Not_So_Darwin_Guy).all(axis=1))
    '''
    print()
    print("Best_1:",Best_1)
    print("Worst_1:",Worst_1)
    '''
    New_Population[Worst_1] = Darwin_Guy
    '''
    print("New_Population After:",New_Population)
    
    print()
    print("After:",New_Population)
    '''
    n_list = New_Population
    
    '''
    print()
    print("The New Population Are:\n",New_Population)
    '''
    
    Min_for_all_Generations_for_Mut_1 = np.vstack((Min_for_all_Generations_for_Mut_1,Min_in_Generation_X_1))
    Min_for_all_Generations_for_Mut_2 = np.vstack((Min_for_all_Generations_for_Mut_2,Min_in_Generation_X_2))
    
    
    Min_for_all_Generations_for_Mut_1_1 = np.insert(Min_in_Generation_X_1, 0, Generation)
    Min_for_all_Generations_for_Mut_2_2 = np.insert(Min_in_Generation_X_2, 0, Generation)
    '''
    Min_for_all_Generations_for_Mut_1_1_ST = np.insert(Min_in_Generation_X_1_ST, 0, Generation)
    Min_for_all_Generations_for_Mut_2_2_ST = np.insert(Min_in_Generation_X_2_ST, 0, Generation)
    '''
    Min_for_all_Generations_for_Mut_1_1_1 = np.vstack((Min_for_all_Generations_for_Mut_1_1_1,Min_for_all_Generations_for_Mut_1_1))
    Min_for_all_Generations_for_Mut_2_2_2 = np.vstack((Min_for_all_Generations_for_Mut_2_2_2,Min_for_all_Generations_for_Mut_2_2))
    
    
    Generation = Generation+1
    
    



One_Final_Guy = np.vstack((Min_for_all_Generations_for_Mut_1_1_1,Min_for_all_Generations_for_Mut_2_2_2))


t = 0
Final_Here = []
for i in One_Final_Guy:
    
    if (One_Final_Guy[t,1]) <= min(One_Final_Guy[:,1]):
        Final_Here = One_Final_Guy[t,:]
    t = t+1
        

One_Final_Guy_Final = Final_Here[np.newaxis]


XY0_Encoded_After = Final_Here[4:]

# DECODING
# DECODING
# DECODING

z = 0
t = 1
X0_num_Sum_Encoded_After = 0

for i in range(len(XY0)//2):
    X0_num_Encoded_After = XY0_Encoded_After[-t]*(2**z)
    X0_num_Sum_Encoded_After += X0_num_Encoded_After
    t = t+1
    z = z+1


p = 0
u = 1 + (len(XY0)//2)
Y0_num_Sum_Encoded_After = 0

for j in range(len(XY0)//2):
    Y0_num_Encoded_After = XY0_Encoded_After[-u]*(2**p)
    Y0_num_Sum_Encoded_After += Y0_num_Encoded_After
    u = u+1
    p = p+1


Decoded_X_After = round((X0_num_Sum_Encoded_After * Precision_X) + a_X3, 2)
Decoded_Y_After = round((Y0_num_Sum_Encoded_After * Precision_Y) + a_X4, 2)

print()
print()
print("The High Accuracy is:",(1-One_Final_Guy_Final[:,1]))
print("Number of Neurons:",Final_Here[2])
print("Number of Hidden Layers:",Final_Here[3])
print("Learning Rate:",Decoded_X_After)
print("Momentum:",Decoded_Y_After)

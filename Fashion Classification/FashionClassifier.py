#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 20:24:31 2019

@author: shonjacob
"""
# Created along the similar lines of the machine learning tutorial https://www.udemy.com/machine-learning-practical/learn/v4/ , by Krill and Dr. Ryan Ahmed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Read the csv file and the separator is a comma
# use input/filename
fashion_train_df = pd.read_csv('fashion-mnist_train.csv', sep = ',')
fashion_test_df = pd.read_csv('fashion-mnist_test.csv', sep = ',')

#checking the dataset
fashion_train_df.head()
fashion_train_df.tail()
fashion_train_df.shape

#convert dataframe to array
training = np.array(fashion_train_df, dtype = 'float32')
testing = np.array(fashion_test_df, dtype = 'float32')


#show just 1 image, reshape is only there for nparray and not for dataframe
# i is a random number between 1  and 60000, so it shows 1 image in whole dataset
import random
i = random.randint(1, 60000)
# we start from index 1 to end, the 0th column is label
plt.imshow(training[i, 1:].reshape(28, 28))

label = training[i,0]
label

'''
The 10 labels are defined as:
    0-> tshirt/top
    1-> trouser
    2-> pullover
    3-> dress
    4-> coat
    5-> sandal
    6-> shirt
    7-> sneaker
    8-> bag
    9-> ankle boot
'''


# Viewing images in a grid format
# Define dimensions of the plot grid
W_grid = 15
l_grid = 15
#subplot returns objects of figure and axes
# we can use the axes object to plot specific features at various locations
fig, axes = plt.subplots(l_grid, W_grid, figsize = (17,17))
axes = axes.ravel() # flatten the 15x15 matrix into 225 array

m_training =len(training) # get the lenght of the training dataset

#select a random number from 0 to m_training 
for i in np.arange(0, W_grid * l_grid): #create evenly spaced variables in the range , returns 0 to 254 indexes
    #select a random number
    index = np.random.randint(0, m_training)
    #read and display an image with the selected index
    axes[i].imshow(training[index, 1:].reshape(28,28))
    axes[i].set_title(training[index, 0], fontsize = 8)
    axes[i].axis('off')

plt.subplots_adjust(hspace = 0.4)    

#TRAINING THE MODEL
# For image data we need to preserve the spacial dependencies bethween the pixels of the images
# So before we feed the image to the neural netwrok we ened to perform convolution
# The convolutional layer uses feature detectors/kernels and pooling layers uses pooling filters  and takes max pooling and then the data is flattened
# Feature kernel are used to create feature maps     
X_train = training[:,1:]/255
y_train = training[:, 0]

X_test = testing[:,1:]/255
y_test = testing[:, 0]

from sklearn.model_selection import train_test_split
X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size = 0.2, random_state = 12345)
# random state is used to get the same result , it doesnt matter what the state number is

#Put the dataset to proper shape
X_train = X_train.reshape(X_train.shape[0], *(28, 28, 1))
X_test = X_test.reshape(X_test.shape[0], *(28, 28, 1))
X_validate = X_validate.reshape(X_validate.shape[0], *(28, 28, 1))
# the 1 in the reshape represents the binary greyscale


# Creating the CNN through Keras
import keras
from keras.models import Sequential
from keras.layers import  Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

cnn_model = Sequential()
cnn_model.add(Conv2D(32, 3, 3, input_shape = (28, 28, 1), activation = 'relu'))
# 32 kernel layers of 3*3 each
cnn_model.add(MaxPooling2D(pool_size = (3,3)))
cnn_model.add(Flatten())
#hidden layer
cnn_model.add(Dense(output_dim = 32, activation = 'relu'))
#output 10 classes layer
cnn_model.add(Dense(output_dim = 10, activation = 'sigmoid'))
cnn_model.compile(loss = 'sparse_categorical_crossentropy', optimizer = Adam(lr = 0.001), metrics = ['accuracy'])
epochs = 50
cnn_model.fit(X_train, y_train, batch_size = 200, nb_epoch = epochs, verbose = 1, validation_data = (X_validate, y_validate))



evaluation = cnn_model.evaluate(X_test, y_test)
print("Test Accuracy : {:,.3f}".format(evaluation[1]))

predicted_classes = cnn_model.predict_classes(X_test)

L = 5
M = 5
fig, axes = plt.subplots(L, M, figsize = (12, 12))
axes = axes.ravel()

for i in np.arange(0, L*M):
    axes[i].imshow(X_test[i].reshape(28, 28))
    axes[i].set_title("Prediction Class = {:0.1f}\n True Class = {:0.1f}".format(predicted_classes[i], y_test[i]))
    axes[i].axis('off')

plt.subplots_adjust(wspace= 0.5)
    
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predicted_classes)
plt.figure(figsize = (14, 10))
sns.heatmap(cm, annot=True)


from sklearn.metrics import classification_report

num_classes = 10
target_names = ["Classes {}".format(i) for i in range(num_classes)]
print(classification_report(y_test, predicted_classes, target_names = target_names))

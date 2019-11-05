#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import urllib.request as request
from sklearn.model_selection import (KFold, train_test_split)
from tensorflow.python.training import gradient_descent
import tensorflow as tf
import os
import numpy as np
from operator import itemgetter
import pickle

tf.compat.v1.disable_eager_execution()


if not os.path.exists('./Skin_NonSkin.txt'):
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00229/Skin_NonSkin.txt'
    request.urlretrieve(url,'./Skin_NonSkin.txt')


#Read using pandas

df = pd.read_csv('Skin_NonSkin.txt', sep='\t',names =['B','G','R','skin'])
#df = data printed as columns
print(df.head())

#Check Missing values

# NO MISSING VALUES
df.isna().sum()


#Standardize dataset
#removing 'skin' from dataset. just prints R G B columns
feature = df[df.columns[~df.columns.isin(['skin'])]] #Except Label

#Converting to 0 and 1 (this col has values 1 and 2) returns 0 if false,
#1 if true, then multiplies by 1
label = (df[['skin']] == 1)*1

#Pixel values range from 0-255 converting between 0-1
feature = feature / 255.


#Explore your data
#Please try to understand the nature of data

# Lets see how many 0s and 1s 194198 - 0, 50859 - 1
(label == 0).skin.sum(), (label == 1).skin.sum()

#SPLIT DATA INTO 5 CROSS - VALIDATION
#returns an array rbg (0-1)
x = feature.values

#returns an array of labeled values 0 or 1
y = label.values

# We will keep fix test and take 5 cross validation set
# so we will have five different data set

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.20, random_state=1)

#Lets see the size of xtrain, xtest
print(len(xtrain),len(xtest))

# 5 Fold Split
# First merge xtrain and ytrain so that we can easily divide into 5 chunks

data = np.concatenate([xtrain,ytrain],axis = 1)
# Observe the shape of array
xtrain.shape,ytrain.shape,data.shape

# Divide our data to 5 chunks
chunks = np.split(data,5)

# print("Chunks", chunks)
# print("endchunks")

#creates a dict for 5 folds. each fold has a train, val and test field
datadict = {'fold1':{'train':{'x':None,'y':None},'val':{'x':None,'y':None},'test':{'x':xtest,'y':ytest}},
            'fold2':{'train':{'x':None,'y':None},'val':{'x':None,'y':None},'test':{'x':xtest,'y':ytest}},
            'fold3':{'train':{'x':None,'y':None},'val':{'x':None,'y':None},'test':{'x':xtest,'y':ytest}},
            'fold4':{'train':{'x':None,'y':None},'val':{'x':None,'y':None},'test':{'x':xtest,'y':ytest}},
            'fold5':{'train':{'x':None,'y':None},'val':{'x':None,'y':None},'test':{'x':xtest,'y':ytest}},}



for i in range(5):
    datadict['fold'+str(i+1)]['val']['x'] = chunks[i][:,0:3]
    # print(datadict['fold'+str(i+1)]['val']['x'])
    datadict['fold'+str(i+1)]['val']['y'] = chunks[i][:,3:4]

    idx = list(set(range(5))-set([i]))
    X = np.concatenate(itemgetter(*idx)(chunks),0)
    datadict['fold'+str(i+1)]['train']['x'] = X[:,0:3]
    datadict['fold'+str(i+1)]['train']['y'] = X[:,3:4]



def writepickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def readpickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

writepickle(datadict,'data.pkl')


#Now You Can Read This Pickle File And Use In Your Project
data = readpickle('data.pkl')

# print(data)
#How to access data
#Example : Access fold1 data
fold1 = data['fold1']
fold1_train = fold1['train']
fold1_val = fold1['val']
fold1_test = fold1['test']


# fold2 = data['fold2']
# fold2_train = fold2['train']
# fold2_val = fold2['val']
# fold2_test = fold2['test']

#xtrain stores just the first fold of training values
xtrain, ytrain = fold1_train['x'],fold1_train['y']
xval, yval = fold1_val['x'], fold1_val['y']
xtest, ytest = fold1_test['x'],fold1_test['y']


xval.shape, yval.shape
xtest.shape,ytest.shape
print(xtrain.shape)

#hyperparameters
learning_rate = 0.01
epochs = 50
batch_size = 100
batches = int(xtrain.shape[0] / batch_size)

#one_hot encoding for train and test data (y values)
with tf.compat.v1.Session() as sesh:
    ytrain = sesh.run(tf.one_hot(ytrain, 2))
    ytest = sesh.run(tf.one_hot(ytest, 2))


print("YEET ",ytrain[:5])

# Y = o(X * W + B)
X = tf.compat.v1.placeholder(tf.float32, [None, 3])
Y = tf.compat.v1.placeholder(tf.float32, [None, 2])

Weights = tf.Variable(np.random.randn(3, 2).astype(np.float32))

Bias = tf.Variable(np.random.randn(2).astype(np.float32))

prediction = tf.nn.softmax(tf.add(tf.matmul(X, Weights), Bias))
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.math.log(prediction), axis=1))
optimizer = gradient_descent.GradientDescentOptimizer(learning_rate).minimize(cost)


#now we train:
with tf.compat.v1.Session() as sesh:
    sesh.run(tf.compat.v1.global_variables_initializer())

    for epoch in range(epochs):
        for i in range(batches):
            offset = i * epoch
            x = xtrain[offset: offset + batch_size]
            y = ytrain[offset: offset + batch_size]
            sesh.run(optimizer, feed_dict={X: x, Y: y})
            c = sesh.run(cost, feed_dict={X: x, Y: y})

        if not epoch % 2:
            print(f'epoch:{epoch} cost={c:.4f}')




#Now use above dataset to complete following work

# # Assignment 2
#     You can use any libraires you want, but choose python as your platform
#
#     1. Implement Logistic Regression on this 5 fold data
#     2. Report Test Accuracy, Val Accuracy on each fold
#        Follow following format
#        ________________________
#
#             |  ACCURACY
#        FOLD | VAL | TEST
#        ________________________
#             |     |
#        1    |  ?? |  ??
#        2    |  ?? |  ??
#        3    |  ?? |  ??
#        4    |  ?? |  ??
#        5    |  ?? |  ??
#        ________________________
#        AVG  |  ?? |  ??
#
#     3. Report Visualization
#
#     NOTE :  You must submit two things
#             First : A pdf report with following explanation
#                     - What tools you used and why?
#                     - Metrics as explained in (2)
#                     - Visualization/Graph
#                     - Conclude your experiment
#                     - Add a github repo as report
#
#             Second : A github repo
#

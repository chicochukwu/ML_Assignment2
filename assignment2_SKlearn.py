import pandas as pd
import urllib.request as request
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (KFold, train_test_split)
from sklearn.linear_model import LogisticRegression
from operator import itemgetter
from sklearn import metrics


if not os.path.exists('./Skin_NonSkin.txt'):
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00229/Skin_NonSkin.txt'
    request.urlretrieve(url,'./Skin_NonSkin.txt')

df = pd.read_csv('Skin_NonSkin.txt', sep='\t',names =['B','G','R','skin'])

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

# 5 Fold Split
# First merge xtrain and ytrain so that we can easily divide into 5 chunks
data = np.concatenate([xtrain,ytrain],axis = 1)

# Observe the shape of array
xtrain.shape,ytrain.shape,data.shape

# Divide our data to 5 chunks
chunks = np.split(data,5)


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

#How to access data
#Example : Access fold1 data
fold1 = data['fold1']
fold1_train = fold1['train']
fold1_val = fold1['val']
fold1_test = fold1['test']

fold2 = data['fold2']
fold2_train = fold2['train']
fold2_val = fold2['val']
fold2_test = fold2['test']

fold3 = data['fold3']
fold3_train = fold3['train']
fold3_val = fold3['val']
fold3_test = fold3['test']

fold4 = data['fold4']
fold4_train = fold4['train']
fold4_val = fold4['val']
fold4_test = fold4['test']

fold5 = data['fold5']
fold5_train = fold5['train']
fold5_val = fold5['val']
fold5_test = fold5['test']

logReg = LogisticRegression(random_state = 0)

def train_test():
    for i in range(1, 6):

        if(i == 1):
            #load val and test data
            xval, yval = fold1_val['x'], fold1_val['y']
            xtest, ytest = fold1_test['x'], fold1_test['y']

            #load train data
            xtrain, ytrain = fold2_train['x'] + fold3_train['x'] + fold4_train['x'] + fold5_train['x'], fold2_train['y'] + \
                             fold3_train['y'] + fold4_train['y'] + fold5_train['y']

            #change ytain's type to an integer
            ytrain = ytrain.astype('int')
            logReg.fit(xtrain, ytrain.ravel())

            #change test and val types
            xtest = xtest.astype('float')
            ytest = ytest.astype('float')
            xval = xval.astype('float')
            yval = yval.astype('float')

            #run prediction on test data
            pred = logReg.predict(xtest)
            cm = metrics.confusion_matrix(ytest, pred)

            #store each folds accuracy for test and val
            fold1_test_acc = logReg.score(xtest, ytest) * 100.
            fold1_val_acc = logReg.score(xval, yval) * 100.

        elif(i == 2):
            xval, yval = fold2_val['x'], fold2_val['y']
            xtest, ytest = fold2_test['x'], fold2_test['y']

            xtrain, ytrain = fold1_train['x'] + fold3_train['x'] + fold4_train['x'] + fold5_train['x'], fold1_train['y'] + \
                             fold3_train['y'] + fold4_train['y'] + fold5_train['y']

            ytrain = ytrain.astype('int')
            logReg.fit(xtrain, ytrain.ravel())

            xtest = xtest.astype('float')
            ytest = ytest.astype('float')
            xval = xval.astype('float')
            yval = yval.astype('float')

            fold2_test_acc = logReg.score(xtest, ytest) * 100.
            fold2_val_acc = logReg.score(xval, yval) * 100.

        elif(i == 3):
            xval, yval = fold3_val['x'], fold3_val['y']
            xtest, ytest = fold3_test['x'], fold3_test['y']

            xtrain, ytrain = fold1_train['x'] + fold2_train['x'] + fold4_train['x'] + fold5_train['x'], fold1_train['y'] + \
                             fold2_train['y'] + fold4_train['y'] + fold5_train['y']

            ytrain = ytrain.astype('int')
            logReg.fit(xtrain, ytrain.ravel())

            xtest = xtest.astype('float')
            ytest = ytest.astype('float')
            xval = xval.astype('float')
            yval = yval.astype('float')

            logReg.predict(xtest)

            fold3_test_acc = logReg.score(xtest, ytest) * 100.
            fold3_val_acc = logReg.score(xval, yval) * 100.

        elif(i == 4):
            xval, yval = fold4_val['x'], fold4_val['y']
            xtest, ytest = fold4_test['x'], fold4_test['y']

            xtrain, ytrain = fold1_train['x'] + fold2_train['x'] + fold3_train['x'] + fold5_train['x'], fold1_train['y'] + \
                             fold2_train['y'] + fold3_train['y'] + fold5_train['y']

            ytrain = ytrain.astype('int')
            logReg.fit(xtrain, ytrain.ravel())

            xtest = xtest.astype('float')
            ytest = ytest.astype('float')
            xval = xval.astype('float')
            yval = yval.astype('float')

            logReg.predict(xtest)

            fold4_test_acc = logReg.score(xtest, ytest) * 100.
            fold4_val_acc = logReg.score(xval, yval) * 100.

        elif(i == 5):
            xval, yval = fold5_val['x'], fold5_val['y']
            xtest, ytest = fold5_test['x'], fold5_test['y']

            xtrain, ytrain = fold1_train['x'] + fold2_train['x'] + fold3_train['x'] + fold4_train['x'], fold1_train['y'] + \
                             fold2_train['y'] + fold3_train['y'] + fold4_train['y']

            ytrain = ytrain.astype('int')
            logReg.fit(xtrain, ytrain.ravel())

            xtest = xtest.astype('float')
            ytest = ytest.astype('float')
            xval = xval.astype('float')
            yval = yval.astype('float')

            logReg.predict(xtest)

            fold5_test_acc = logReg.score(xtest, ytest) * 100.
            fold5_val_acc = logReg.score(xval, yval) * 100.

    return fold1_test_acc, fold1_val_acc, fold2_test_acc, fold2_val_acc, fold3_test_acc,\
           fold3_val_acc, fold4_test_acc, fold4_val_acc, fold5_test_acc, fold5_val_acc


f1t, f1v, f2t, f2v, f3t, f3v, f4t, f4v, f5t, f5v  = train_test()

def format_output(f1t, f1v, f2t, f2v, f3t, f3v, f4t, f4v, f5t, f5v):
    #first determine averages
    val_avg = (f1v + f2v + f3v + f4v + f5v) / 5

    test_avg = (f1t + f2t + f3t + f4t + f5t) / 5



    print(" --------------------------- \n\n", "     |  ACCURACY\n",\
         "FOLD | VAL  | TEST\n\n", "--------------------------- \n", \
         "     |      |")
    print("1     |{f1v:.3f}|{f1t:.3f}".format(f1v=f1v, f1t=f1t))
    print("2     |{f2v:.3f}|{f2t:.3f}".format(f2v=f2v, f2t=f2t))
    print("3     |{f3v:.3f}|{f3t:.3f}".format(f3v=f3v, f3t=f3t))
    print("4     |{f4v:.3f}|{f4t:.3f}".format(f4v=f4v, f4t=f4t))
    print("5     |{f5v:.3f}|{f5t:.3f}".format(f5v=f5v, f5t=f5t))
    print("\n ---------------------------")
    print("AVG   |{v:.3f}|{t:.3f}".format(v=val_avg, t=test_avg))


format_output(f1t, f1v, f2t, f2v, f3t, f3v, f4t, f4v, f5t, f5v)

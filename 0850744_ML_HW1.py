#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from numpy.linalg import inv
import math
import pandas 
import random 
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement, combinations, product


# In[2]:


dataframe1 = pandas.read_csv("data_X.csv")
dataframe2 = pandas.read_csv("data_T.csv")

X_all = np.array(dataframe1.values, dtype = np.float)
Y_all = np.array(dataframe2.values, dtype = np.float)

c = list(zip(X_all, Y_all))
random.shuffle(c)
X_all, Y_all = zip(*c)

X_all = np.array(X_all)
Y_all = np.array(Y_all)

X_all.reshape(500,8)
Y_all.reshape(500,2)

X_test = np.array(X_all[400:500,1:8], dtype = np.float)
Y_test = np.array(Y_all[400:500,1:2], dtype = np.float)
X_test.reshape(100,7)
Y_test.reshape(100,1)

X_train = np.array(X_all[:400,1:8], dtype = np.float)
Y_train = np.array(Y_all[:400,1:2], dtype = np.float)
X_train.reshape(400,7)
Y_train.reshape(400,1)
print()


# In[3]:


def RMSE(x, w, t):
    y = np.dot(x, w)
    return math.sqrt(np.power(y-t, 2).sum()/len(x))

def train(x, y, lambda0 = 0): 
    w = np.linalg.pinv(x.T.dot(x) + lambda0 * np.identity(len(x.T.dot(x)))).dot(x.T).dot(y)
    return w

def regression_makingPhi(x, M):
    init1 = np.full((len(x), 1), 1)      
    for M1 in range(M):
        for c in list(product(range(len(x[0])), repeat=M1+1)):
            p = np.expand_dims(np.prod(x[:, c], axis=1), axis=1)
            init1 = np.concatenate((init1, p), axis=1)
    return init1
    
def gaussion_makingPhi(x, meanX, varX):
    phi = []     
    for featureIndex in range(len(meanX)):
        thisColume = x[:,featureIndex]
        oneExpW = []
        for x0Index in range(len(x)):                
            oneExpW.append(math.exp(-1 *  ((thisColume[x0Index] - meanX[featureIndex]) ** 2)/ (2 * (varX[featureIndex] ** 2))))
        phi.append(oneExpW)
    phi = np.array(phi).T     
    return phi


# In[4]:


print("[INFO] for 1.(a) & 2.(c)")
print("=================================================")
lambdas = [0.01,0.1,1,10,100]
numberOfFold = 5
X_num = len(X_all)
oneFoldNum = int(X_num / numberOfFold)
for foldIndex in range(numberOfFold):
    X_test = np.array(X_all[foldIndex*oneFoldNum:(foldIndex+1)*oneFoldNum,1:8], dtype = np.float)
    Y_test = np.array(Y_all[foldIndex*oneFoldNum:(foldIndex+1)*oneFoldNum,1:2], dtype = np.float)
    X_test.reshape(oneFoldNum,7)
    Y_test.reshape(oneFoldNum,1)
    if(foldIndex == 0):
        X_train = np.array(X_all[:X_num-oneFoldNum,1:8], dtype = np.float)
        Y_train = np.array(Y_all[:X_num-oneFoldNum,1:2], dtype = np.float)
    elif(foldIndex == numberOfFold-1 ):
        X_train = np.array(X_all[oneFoldNum:X_num,1:8], dtype = np.float)
        Y_train = np.array(Y_all[oneFoldNum:X_num,1:2], dtype = np.float)
    else:
        X_train = np.array(X_all[0:foldIndex*oneFoldNum,1:8], dtype = np.float)
        X_train = np.append(X_train,np.array(X_all[(foldIndex+1)*oneFoldNum:X_num,1:8], dtype = np.float) , axis = 0)
        Y_train = np.array(Y_all[0:foldIndex*oneFoldNum,1:2], dtype = np.float)
        Y_train = np.append(Y_train,np.array(Y_all[(foldIndex+1)*oneFoldNum:X_num,1:2], dtype = np.float) , axis = 0)        
    X_train.reshape(X_num - oneFoldNum,7)
    Y_train.reshape(X_num - oneFoldNum,1)
    
    trainRmse = []
    testRmse = []
    M_max = 3
    

    for m in range(1, M_max+1): 
        phi = regression_makingPhi(X_train,m)    
        w = train(phi, Y_train)
        RMSE0 = RMSE(phi, w, Y_train)
        trainRmse.append(RMSE0)


        phii = regression_makingPhi(X_test,m)    
        RMSE0 = RMSE(phii, w, Y_test)
        testRmse.append(RMSE0)
        
    print("fold:",foldIndex)
    for i in range(1, M_max+1):
        print("M=%d, training rmse: %f, testing rmse: %f" % (i, trainRmse[i-1], testRmse[i-1]))
    print()
print("=================================================")


# In[5]:


print("[INFO] for 3.(a)")
print("=================================================")
for lambdaa in lambdas:
    for foldIndex in range(numberOfFold):
        X_test = np.array(X_all[foldIndex*oneFoldNum:(foldIndex+1)*oneFoldNum,1:8], dtype = np.float)
        Y_test = np.array(Y_all[foldIndex*oneFoldNum:(foldIndex+1)*oneFoldNum,1:2], dtype = np.float)
        X_test.reshape(oneFoldNum,7)
        Y_test.reshape(oneFoldNum,1)
        if(foldIndex == 0):
            X_train = np.array(X_all[:X_num-oneFoldNum,1:8], dtype = np.float)
            Y_train = np.array(Y_all[:X_num-oneFoldNum,1:2], dtype = np.float)
        elif(foldIndex == numberOfFold-1 ):
            X_train = np.array(X_all[oneFoldNum:X_num,1:8], dtype = np.float)
            Y_train = np.array(Y_all[oneFoldNum:X_num,1:2], dtype = np.float)
        else:
            X_train = np.array(X_all[0:foldIndex*oneFoldNum,1:8], dtype = np.float)
            X_train = np.append(X_train,np.array(X_all[(foldIndex+1)*oneFoldNum:X_num,1:8], dtype = np.float) , axis = 0)
            Y_train = np.array(Y_all[0:foldIndex*oneFoldNum,1:2], dtype = np.float)
            Y_train = np.append(Y_train,np.array(Y_all[(foldIndex+1)*oneFoldNum:X_num,1:2], dtype = np.float) , axis = 0)        
        X_train.reshape(X_num - oneFoldNum,7)
        Y_train.reshape(X_num - oneFoldNum,1)

        trainRmse = []
        testRmse = []
        M_max = 3
       

        for m in range(1, M_max+1):  
            phi = regression_makingPhi(X_train,m)    
            w = train(phi, Y_train,lambdaa)
            RMSE0 = RMSE(phi, w, Y_train)
            trainRmse.append(RMSE0)


            phii = regression_makingPhi(X_test,m)    
            RMSE0 = RMSE(phii, w, Y_test)
            testRmse.append(RMSE0)
        print("fold:",foldIndex," , lamda:",lambdaa)
        for i in range(1, M_max+1):
            print("M=%d, training rmse: %f, testing rmse: %f" % (i, trainRmse[i-1], testRmse[i-1]))
        print()
print("=================================================")


# In[6]:


print("[INFO] for 1.(b)")
print("=================================================")
for c in list(combinations(range(0, 7), 6)): 
    phi = regression_makingPhi(X_train[:, list(c)],1)
    w = train(phi, Y_train)
    RMSE0 = RMSE(phi, w, Y_train)

    print("usingIndexes: ", list(c), "\nRMSE: ", RMSE0)
    print()
print("=================================================")


# In[7]:


print("[INFO] for 2.(b)")
print("=================================================")
XtrainMean = []
XtrainVar = []
for feature in range(X_train.shape[1]):
    temm = 0
    for studentIndex in range(len(X_train)):
        temm += X_train[studentIndex][feature]
    mean0 = temm / len(X_train)
    XtrainMean.append(mean0)

    temm = 0
    for studentIndex in range(len(X_train)):
        temm += (X_train[studentIndex][feature] - mean0) ** 2
    var0 = math.sqrt(temm / len(X_train))
    XtrainVar.append(var0)
XtrainMean = np.array(XtrainMean)
XtrainVar = np.array(XtrainVar)

phi = gaussion_makingPhi(X_train, XtrainMean, XtrainVar)
w = train(phi, Y_train)
RMSE0 = RMSE(phi, w, Y_train)

phii = gaussion_makingPhi(X_test, XtrainMean, XtrainVar)
RMSE1 = RMSE(phii, w, Y_test)

print("rmse_train_ML: ",RMSE0)
print("rmse_test_ML: ",RMSE1)
print("=================================================")


# In[8]:


print("[INFO] for 3.(b)")
print("=================================================")
for lambdaa in lambdas:
    
    phi = gaussion_makingPhi(X_train, XtrainMean, XtrainVar)
    w = train(phi, Y_train,lambdaa)
    RMSE0 = RMSE(phi, w, Y_train)

    phii = gaussion_makingPhi(X_test, XtrainMean, XtrainVar)
    RMSE1 = RMSE(phii, w, Y_test)

    print("lambda = ", lambdaa," , rmse_train_MAP:",RMSE0)
    print("lambda = ", lambdaa," , rmse_test_MAP:",RMSE1)
    print()
print("=================================================")


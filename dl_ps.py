# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 04:04:20 2019

@author: Shaurya Gupta
"""
#everything required is imported
import numpy as np # linear algebra
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# import warnings
import warnings
# filter warnings
warnings.filterwarnings('ignore')
import pandas


#loading data
dataframe = pandas.read_csv("pulsar_stars.csv", header=None)
dataset = dataframe.values
X = dataset[1:-1,0:8].astype(float)
Y = dataset[1:-1,8].astype(float)
Y = Y.reshape(X.shape[0],1)

#print(X.shape)
#print(Y.shape) 

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=26)
ntrain = X_train.shape[0]
ntest = X_test.shape[0]

x_train = X_train.T
x_test = X_test.T
y_train = Y_train.T
y_test = Y_test.T

#print(x_train.shape)
#print(x_test.shape)
#print(y_train.shape)
#print(y_test.shape)

def sigmoid(x):
    yout = 1/(1+np.exp(-x))
    return yout

def init(x_train, y_train):
    para = {"w1": np.random.randn(6,x_train.shape[0]) * 0.1,
            "b1": np.zeros((6,1)),
            "w2": np.random.randn(y_train.shape[0],6) * 0.1,
            "b2": np.zeros((y_train.shape[0],1))}
    return para

def forward(X,para):
    Z1 = np.dot(para["w1"],X) +para["b1"]
    A1 = np.tanh(Z1)
    Z2 = np.dot(para["w2"],A1) + para["b2"]
    A2 = sigmoid(Z2)

    res = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    return A2,res
    
def learn(A2,res,X,Y,para):

    
    
    #logp = np.multiply(np.log(A2),Y)
    #cost = -np.sum(logp)/Y.shape[1]
    
    loss = -Y*np.log(A2)-(1-Y)*np.log(1-A2)
    cost = (np.sum(loss))/X.shape[1] 
    
    dZ2 = res["A2"]-Y
    dW2 = np.dot(dZ2,res["A1"].T)/X.shape[1]
    db2 = np.sum(dZ2,axis =1,keepdims=True)/X.shape[1]
    dZ1 = np.dot(para["w2"].T,dZ2)*(1 - np.power(res["A1"], 2))
    dW1 = np.dot(dZ1,X.T)/X.shape[1]
    db1 = np.sum(dZ1,axis =1,keepdims=True)/X.shape[1]
    grads = {"dw1": dW1,
             "db1": db1,
             "dw2": dW2,
             "db2": db2}
    return cost,grads

def update(para, grads, alpha = 0.01):
    para = {"w1": para["w1"]-alpha*grads["dw1"],
                  "b1": para["b1"]-alpha*grads["db1"],
                  "w2": para["w2"]-alpha*grads["dw2"],
                  "b2": para["b2"]-alpha*grads["db2"]}
    
    return para

def predict(para,x_test):
    
    A2, res = forward(x_test,para)
    Y_pred = np.zeros((1,x_test.shape[1]))
    for i in range(A2.shape[1]):
        if A2[0,i]<= 0.5:
            Y_pred[0,i] = 0
        else:
            Y_pred[0,i] = 1

    return Y_pred

def NNcompile(x_train,y_train,x_test,y_test,epochs):
    costl = []
    indexl = []
    #initialize function called 
    para = init(x_train, y_train)

    for i in range(0, epochs):
         # forward function 
        A2,res = forward(x_train,para)
        cost,grads = learn(A2,res,x_train, y_train,para)
        
        para = update(para,grads)
        if i % 5 == 0:
            costl.append(cost)
            indexl.append(i)
            print ("Cost after",i,":",cost)
    plt.plot(indexl,costl)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.show()
    
    # parameters learned, now predict on both sets
    y_pred_test = predict(para,x_test)
    y_pred_train = predict(para,x_train)

    # print accuracy
    print("train accuracy: {} %".format(100-np.mean(np.abs(y_pred_train-y_train))*100))
    print("test accuracy: {} %".format(100-np.mean(np.abs(y_pred_test-y_test))*100))
    return para

para = NNcompile(x_train,y_train,x_test,y_test,epochs=500)
#print(para)




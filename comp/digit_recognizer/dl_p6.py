#everything required is imported
import numpy as np # linear algebra
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

import warnings
# filter warnings
warnings.filterwarnings('ignore')
import pandas


dataframe1 = pandas.read_csv("train.csv", header=None)
dataset1 = dataframe1.values
dataframe2 = pandas.read_csv("test.csv", header=None)
dataset2 = dataframe2.values
X = dataset1[1:42001,1:785].astype(float)
Y = dataset1[1:42001,0].astype(float)
Y = Y.reshape(X.shape[0],1)
X_test = dataset2[1:42001,0:784].astype(float)
x_test = X_test.T

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

print(X.shape)
print(Y.shape) 
print(dummy_y.shape)


x_train = X.T
y_train = dummy_y.T

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
 

def sigmoid(x):
    yout = 1/(1+np.exp(-x))
    return yout

def ReLU(x):
    return x * (x > 0)

def init(x_train, y_train):
    para = {"w1": np.random.randn(25,x_train.shape[0]) * 0.1,
            "b1": np.zeros((25,1)),
            "w2": np.random.randn(y_train.shape[0],25) * 0.1,
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

    
    
    logp = np.multiply(np.log(A2),Y)
    cost = -np.sum(logp)/Y.shape[1]
    
    #loss = -Y*np.log(A2)-(1-Y)*np.log(1-A2)
    #cost = (np.sum(loss))/X.shape[1] 
    
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
        m = A2[0,i]
        k=0
        for j in range(1,10):
            if (m<A2[j,i]):
                m = A2[j,i]
                k = j
        Y_pred[0,i] = k

    return Y_pred

def NNcompile(x_train,y_train,x_test,epochs):
    costl = []
    indexl = []
    #initialize function called 
    para = init(x_train, y_train)

    for i in range(0, epochs):
         # forward function 
        A2,res = forward(x_train,para)
        cost,grads = learn(A2,res,x_train, y_train,para)
        
        para = update(para,grads)
        if i % 50 == 0:
            costl.append(cost)
            indexl.append(i)
            print ("Cost after",i,":",cost)
    #plt.plot(indexl,costl)
    #plt.xlabel("Iteration")
    #plt.ylabel("Cost")
    #plt.show()
    
    # parameters learned, now predict on both sets
    y_pred_test = predict(para,x_test)
    y_pred_train = predict(para,x_train)
    
    # print accuracy
    #print("train accuracy: {} %".format(100-np.mean(np.abs(y_pred_train-y_train))*100))
    #print("test accuracy: {} %".format(100-np.mean(np.abs(y_pred_test-y_test))*100))
    return y_pred_test,y_pred_train

y_pred_test,y_pred_train = NNcompile(x_train,y_train,x_test,epochs=500)
y_pred_test = y_pred_test.astype(int)
#print(y_pred_train.shape)
#print(y_pred_test.shape)




y_x = np.array(range(1,28001)).reshape(1,28000).T
y_y = y_pred_test.T
y_ans = np.concatenate((y_x,y_y),axis=1).astype(int) #(28000,2)

#y_x = np.array(range(1,11)).reshape(1,10).T
#y_y = np.zeros((1,10)).astype(int).reshape(1,10).T
#y_ans = np.concatenate((y_x,y_y),axis=1).astype(int)

#print(y_x)
#print(y_y)
#print(y_ans)
#print(y_ans.shape)
y_l = y_ans.tolist()

#l= [('ImageId','Label')],y_l]
#print(y_l)

import csv

myData = [['ImageId','Label']]
myFile = open('sub_1.csv', 'w')  
with myFile:   
   writer = csv.writer(myFile) 
   writer.writerows(myData)
   for item in y_l:
       writer.writerows([item])

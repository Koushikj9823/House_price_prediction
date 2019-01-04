# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 22:05:00 2019

@author: Koushik
"""


# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 20:20:30 2018

@author: Koushik
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

data=pd.read_csv('ex1data2.txt',header= None)

X = data.iloc[:,0:2] # read first two columns into X
y = data.iloc[:,2] # read the third column into y
m = len(y) # no. of training samples
data.head()
X = (X - np.mean(X))/np.std(X)
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

plt.scatter(X[:,1], y)
plt.scatter(X[:,1:2], y)
plt.ylabel('Price')
plt.xlabel('House features')
plt.show()
#plt.scatter(X, y)
#plt.xlabel('Population of City in 10,000s')
#plt.ylabel('Profit in $10,000s')
#plt.show()
a=np.shape(y)
#X = X[:,np.newaxis]
y = y[:,np.newaxis]
theta = np.zeros((3,1))
alpha = 0.01
ones = np.ones((m,1)) 
X = np.hstack((ones, X))


def costfunc(X_train,y_train,theta):#
        #temp = np.dot(X,theta)-y
       # temp= (1/2*m)*pow(temp,2)
       temp = np.dot(X_train, theta) - y_train
       return np.sum(np.power(temp, 2)) / (2*m)
J = costfunc(X_train, y_train, theta)
print(J)
        
def grad(X,y,alpha,theta):
    for i in range(1500):
        temp=np.dot(X,theta)-y
        temp=np.dot(X.T,temp)
        theta = theta - (alpha/m)*temp
    return theta
theta = grad(X_train, y_train, alpha,theta)
print(theta)   
        
J = costfunc(X_train, y_train, theta)
print(J)        
f=theta[1]
#p=[1,1600,3]
#p = (p - np.mean(p))/np.std(p)
res=np.dot(X_test,theta)
print(res)
plt.scatter(X[:,1],y)
plt.scatter(X[:,1:2], y)
plt.xlabel('House_features')
plt.ylabel('House_price')
plt.plot(X[:,1:2], np.dot(X, theta))
plt.show()





# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 20:20:30 2018

@author: Koushik
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

data=pd.read_csv('ex1data2.txt',header= None) #read the dataset

X = data.iloc[:,0:2] # read the independent columns into X
y = data.iloc[:,2] # read the dependent column into y
m = len(y) # no. of training samples

X = (X - np.mean(X))/np.std(X) #Feature scaling

from sklearn.cross_validation import train_test_split #split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


y = y[:,np.newaxis]
theta = np.zeros((3,1))
alpha = 0.01
ones = np.ones((m,1)) 
X = np.hstack((ones, X))
iterations = 1500
cost = np.zeros((len(iterations),1))

def costfunc(X_train,y_train,theta): #Determine the Cost Function
       temp = np.dot(X_train, theta) - y_train
       return np.sum(np.power(temp, 2)) / (2*m)

J = costfunc(X_train, y_train, theta) #Determine the Cost
print(J)
        
def grad(X,y,alpha,theta): #Gradient Descent to find global minima
    for i in range(iterations):
        temp=np.dot(X,theta)-y
        temp=np.dot(X.T,temp)
        theta = theta - (alpha/m)*temp
        cost[i]= costfunc(X, y, theta)
    return theta

theta = grad(X_train, y_train, alpha,theta) #Calculate theta(Parameters)
print(theta)   
    

#Plot Cost vs iterations graph 
iters = list()
for i in range(len(cost)):
    iters.append(i+1)
plt.figure("Cost v/s Iterations")
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.plot(iters, cost, 'b')





#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import sys, os
import struct

from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

fp_image = open('machine_learning/train-images.idx3-ubyte', 'rb')
fp_label = open('machine_learning/train-labels.idx1-ubyte', 'rb')


dataSet = []
labelSet = []
#img = np.zeros((28,28))

s = fp_image.read(16)
l = fp_label.read(8)

dataCount = 0
while True :
    s = fp_image.read(784)
    l = fp_label.read(1)
    
    if not s:
        break
    if not l:
        break
    
    index = int(l[0])
    labelSet.append(index)
    
    arr = struct.unpack(len(s)*'B', s)
    dataSet.append(arr)
    dataCount = dataCount + 1
    
print("Data Count :",dataCount)

linear_accuracy, nonlinear_accuracy = [], []

# Linear SVM
for i in range (0, 5) :
    train_Image, test_Image, train_Label, test_Label = train_test_split(dataSet, labelSet, test_size=0.3, random_state=0)
    model = LinearSVC()
    model.fit(train_Image, train_Label)
    linear_accuracy.append( model.score(test_Image, test_Label ))
    
# Non-Linear SVM
for i in range (0, 5) :
    train_Image, test_Image, train_Label, test_Label = train_test_split(dataSet, labelSet, test_size=0.3, random_state=0)
    model = SVC(kernel='rbf')
    model.fit(train_Image, train_Label)
    nonlinear_accuracy.append( model.score(test_Image, test_Label ))
    
linear = np.round( np.mean(linear_accuracy) * 100 , 2 )
nonlinear = np.round ( np.mean(nonlinear_accuracy) * 100 , 2 )

print("Linear Accuracy :", linear_accuracy)
print("Linear Accuracy Average :",linear)

print("Non-Linear Accuracy :", nonlinear_accuracy)
print("Non-Linear Accuracy Average :", nonlinear)


# In[ ]:





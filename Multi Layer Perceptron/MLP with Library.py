#!/usr/bin/env python
# coding: utf-8

# In[25]:


import numpy as np
import sys, os
import struct

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

fp_image = open('machine_learning/train-images.idx3-ubyte', 'rb')
fp_label = open('machine_learning/train-labels.idx1-ubyte', 'rb')

dataSet = []
labelSet = []

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

dataSet = np.array(dataSet)
train_Image, test_Image, train_Label, test_Label = train_test_split(dataSet, labelSet, test_size=0.3, random_state=0)

#기본
mlp = MLPClassifier(hidden_layer_sizes=(100), max_iter=200)
mlp.fit(train_image_scaled, train_Label)

'''
#은닉층 조절 (1)
mlp = MLPClassifier(hidden_layer_sizes=(200), max_iter=200)
mlp.fit(train_image_scaled, train_Label)
'''

'''
#은닉층 조절 (2)
mlp = MLPClassifier(hidden_layer_sizes=(500,100), max_iter=200)
mlp.fit(train_image_scaled, train_Label)
'''

'''
#전처리
train_image_scaled = train_Image / 255
test_image_scaled = test_Image / 255
mlp = MLPClassifier(hidden_layer_sizes=(100), max_iter=200)
mlp.fit(train_image_scaled, train_Label)
'''

'''
#훈련 반복 회수 조절
mlp = MLPClassifier(hidden_layer_sizes=(100), max_iter=500)
mlp.fit(train_image_scaled, train_Label)
'''

accuracy = []
accuracy.append(mlp.score(test_image_scaled, test_Label))
acc = np.round( np.mean(accuracy) * 100 , 2 )

print("Accuracy :", acc)


# In[ ]:





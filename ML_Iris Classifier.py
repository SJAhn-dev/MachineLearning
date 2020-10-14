#!/usr/bin/env python
# coding: utf-8

# In[67]:


import numpy as np
from matplotlib import pyplot as plt

train_data = np.loadtxt("machine_learning/iris_train.csv", delimiter=',', dtype=np.float32)
test_data = np.loadtxt("machine_learning/iris_test.csv", delimiter=',', dtype=np.float32)

group_1, group_2, group_3 = [], [], []

for i in train_data:
    if(i[4]==1):
        group_1.append(i)
    elif(i[4]==2):
        group_2.append(i)
    elif(i[4]==3):
        group_3.append(i)

group_1 = np.array(group_1)[:, :4]
group_2 = np.array(group_2)[:, :4]
group_3 = np.array(group_3)[:, :4]

group_1 = group_1.mean(axis=0)
group_2 = group_2.mean(axis=0)
group_3 = group_3.mean(axis=0)

correct = 0
for i in test_data:
    vec_1 = np.linalg.norm(i[:4]-group_1)
    vec_2 = np.linalg.norm(i[:4]-group_2)
    vec_3 = np.linalg.norm(i[:4]-group_3)
    
    minVal = min(vec_1, vec_2, vec_3)
    
    if(minVal == vec_1 and i[4] == 1):
        correct += 1
    elif(minVal == vec_2 and i[4] == 2):
        correct += 1
    elif(minVal == vec_3 and i[4] == 3):
        correct += 1

print(correct/len(test_data))


# In[ ]:





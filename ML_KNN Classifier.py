#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np
from matplotlib import pyplot as plt

iris_data = np.loadtxt("machine_learning/iris.csv", delimiter=',', dtype=np.float32)

group_1, group_2, group_3, test_group, train_group = [], [], [], [], []

group_1 = np.array(iris_data)[0:50, : ]
group_2 = np.array(iris_data)[50:100, : ]
group_3 = np.array(iris_data)[100:150, : ]

average = 0
for i in range (0, 5):
    correct = 0
    test_group = np.concatenate((group_1[i*10:i*10+10, : ], group_2[i*10:i*10+10, : ], group_3[i*10:i*10+10, : ]),axis = 0)
    if i==0:
        train_group = np.concatenate((group_1[10:50, : ], group_2[10:50, : ], group_3[10:50, : ]), axis = 0)
        
    elif i==4:
        train_group = np.concatenate((group_1[0:40, : ], group_2[0:40, : ], group_3[0:40, : ]), axis=0)
    else:
        train_group_1 = np.concatenate((group_1[0:i*10, : ], group_1[i*10+10:50, : ]), axis=0)
        train_group_2 = np.concatenate((group_2[0:i*10, : ], group_2[i*10+10:50, : ]), axis=0)
        train_group_3 = np.concatenate((group_3[0:i*10, : ], group_3[i*10+10:50, : ]), axis=0)
        train_group = np.concatenate((train_group_1, train_group_2, train_group_3), axis=0)
    
    for j in test_group:
        test_target = j
        min = 10000
        min_class = 0
        for k in train_group:
            train_target = k
            vec = np.linalg.norm(test_target[:4]-train_target[:4])
            if(vec < min):
                min = vec
                min_class = train_target[4]
        if(test_target[4] == min_class):
            correct += 1
    
    average += correct
    accuracy = np.round(correct/30, 2)
    print('{0} : {1}%'.format(i,accuracy*100))

print('평균 : {}%'.format(average/150 * 100))


# In[ ]:





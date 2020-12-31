#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[105]:


import numpy as np
import sys, os
import struct


def preprocessing():
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
        if index != 1 and index != 5 and index != 8 :
            continue

        labelSet.append(index)

        arr = struct.unpack(len(s)*'B', s)
        dataSet.append(arr)
        dataCount = dataCount + 1
    
    partition_Index = dataCount // 2
    dataSet = np.array(dataSet) / 255
    
    train_dataSet = dataSet[0 : partition_Index]
    train_labelSet = labelSet[0 : partition_Index]
    test_dataSet = dataSet[partition_Index : dataCount - 1]
    test_labelSet = labelSet[partition_Index : dataCount - 1]
    
    return train_dataSet, train_labelSet, test_dataSet, test_labelSet
    
def sigmoid(x):
    dst = np.zeros(len(x))
    dst = 1 / (1 + np.exp(-x))
    return dst

def sigmoid_derivation(x):
    sig = sigmoid(x)
    sig_2 = 1-sigmoid(x)
    
    dst = np.zeros(len(x))
    for i in range(len(x)):
        dst[i] = sig[i] * sig_2[i]
    
    return dst

def initialize(row_size, col_size):
    dst = np.random.uniform(-1.0, 1.0, row_size * col_size)
    dst = dst.reshape(row_size, col_size)
    
    return dst

def bias(input, len):
    dst = np.zeros(len+1)
    dst[1:len + 1] = input[0:len]
    dst[0] = 1
    
    return dst
    
if __name__ == '__main__':
    INPUT_NODE = 784
    HIDDEN_NODE = 255
    OUTPUT_NODE = 3
    
    # Data 전처리
    train_data, train_label, test_data, test_label = preprocessing()
    TRAIN_DATACOUNT = len(train_data)
    TEST_DATACOUNT = len(test_data)
    #print(TRAIN_DATACOUNT)
    
    # 가중치 배열 초기화
    # hidden_grad = 785 * 255
    hidden_grad = initialize(INPUT_NODE + 1, HIDDEN_NODE)
    # ouput_grad = 256 * 3
    output_grad = initialize(HIDDEN_NODE + 1, OUTPUT_NODE)
    
    for epoch in range(5):
        for i in range(TRAIN_DATACOUNT):
            # sample data = 1 * 785
            sample_data = bias(train_data[i], INPUT_NODE)
            # sample label = int
            sample_label = train_label[i]

            # hidden_layer = 1 * 255
            hidden_layer = np.zeros(HIDDEN_NODE)
            # output_layer = 1 * 3
            output_layer = np.zeros(OUTPUT_NODE)

            # -> 전방 계산
            hidden_layer = np.dot(sample_data, hidden_grad)
            # hidden_layer = 255 -> 256
            hidden_layer = bias(hidden_layer, HIDDEN_NODE)
            hidden_sigmoid = sigmoid(hidden_layer)
            hidden_sigmoid[0] = 1

            output_layer = np.dot(hidden_sigmoid, output_grad)
            output_sigmoid = sigmoid(output_layer)

            # -> 오류 역전파
            value = 0
            max_value = np.max(output_sigmoid)
            if sample_label == 1:
                value = np.array([1.0,0,0])
            elif sample_label == 5:
                value = np.array([0,1.0,0])
            else:
                value = np.array([0,0,1.0])
            
            output_node = value - output_sigmoid
            output_delta = np.zeros(3)
            siged = sigmoid_derivation(output_layer)
            for x in range(3):
                output_delta[x] = output_node[x] * siged[x]
            delta_output_grad = np.zeros((256, 3))
            hidden_sigmoid = hidden_sigmoid.reshape(256,1)
            delta_output_grad = -1 * np.dot(hidden_sigmoid, output_delta.reshape(1,3))

            hidden_delta = np.dot(output_grad, output_delta) * sigmoid_derivation(hidden_layer)
            delta_hidden_grad = np.zeros((785,256))
            sample_data = sample_data.reshape(785,1)
            hidden_delta = hidden_delta.reshape(1,256)
            delta_hidden_grad = -1 * np.dot(sample_data, hidden_delta)
            

            # -> 가중치 갱신
            output_grad = output_grad - delta_output_grad
            hidden_grad = hidden_grad - delta_hidden_grad[:,1:256]
        print("epoch :",epoch+1)
        
    correct = 0
    for i in range(TEST_DATACOUNT):
        sample = test_data[i]
        label = test_label[i]
        
        input = np.zeros(785)
        sample = np.asarray(sample)
        input[1:785] = sample.reshape(1,784)
        input[0] = 1
        hidden_layer = np.dot(input, hidden_grad)
        hidden_layer = sigmoid(hidden_layer)
        
        hidden_input = np.zeros(256)
        hidden_input[1:256] = hidden_layer.reshape(1,255)
        hidden_input[0] = 1
        output_layer = np.dot(hidden_input, output_grad)
        output_layer = sigmoid(output_layer)
        
        max = np.max(output_layer)
        if label == 1 and max == output_layer[0]:
            correct = correct + 1
        elif label == 5 and max == output_layer[1]:
            correct = correct + 1
        elif label == 8 and max == output_layer[2]:
            correct = correct + 1
        
    print("testCount :", TEST_DATACOUNT)
    print("correct :", correct)


# In[ ]:





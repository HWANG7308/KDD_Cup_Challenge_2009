# -*- coding: utf-8 -*-
'''
split data into train/validation/test

Author:
organize all code: S1802373
'''
# import pandas as pd
# import matplotlib.pyplot as plt
import numpy as np
import csv
# from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA

def data_split(file_path):
    data=[]
    traffic_feature=[]
    traffic_target=[]
    csv_file = csv.reader(open(file_path))
    for content in csv_file:
        content=list(map(float,content))
        if len(content)!=0:
            data.append(content)
            traffic_feature.append(content[0:-1])
            traffic_target.append(content[-1])
    scaler = StandardScaler() # 标准化转换
    scaler.fit(traffic_feature)  # 训练标准化对象
    traffic_feature= scaler.transform(traffic_feature)   # 转换数据集

    # Split X to train, validation and test
    trainX, testX, trainY, testY = train_test_split(traffic_feature, traffic_target, test_size=0.2, random_state=1)
    # trainX, validateX, trainY, validateY = train_test_split(trainX, trainY, test_size=0.25, random_state=1)

    # double check the dataset has been split representatively
    # Ys = dict(train=trainY, valid=validateY, test=testY)
    Ys = dict(train=trainY, test=testY)
    for y in Ys:
        temp_data = Ys[y]
        print("%s set: sample size = %d, incidence rate = %f"
              % (y, len(temp_data), temp_data.count(1) / len(temp_data)))

    # return trainX, validateX, testX, trainY, validateY, testY
    return trainX, testX, trainY, testY

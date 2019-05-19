# -*- coding: utf-8 -*-
'''
main file

Author:
organize all code: S1802373
'''
import sys
import os
from tool import log
from split_data import data_split
from build_model import *
from draw_plot import *

def find_file(param):
    # change the following three file paths for different setting (with/without missing matrix; with/without PCA)
    _appetency = '../data/feature/small_train_appetency_n_no_pca.csv'
    _churn = '../data/feature/small_train_churn_n_no_pca.csv'
    _upselling = '../data/feature/small_train_upselling_n_no_pca.csv'

    if param == "a":
        task = "appetency"
        file_path = _appetency
        print("Appetency Start")
    elif param == "c":
        task = "churn"
        file_path = _churn
        print("Churn Start")
    elif param == "u":
        task = "upselling"
        file_path = _upselling
        print("Upselling Start")
    else:
        print("Wrong")
        exit()
    return file_path, task

def prepare_dir(task):
    if not os.path.exists("../result"):
        os.mkdir("../result")
    if not os.path.exists("../result/graph"):
        os.mkdir("../result/graph")
    if not os.path.exists("../result/text"):
        os.mkdir("../result/text")

    file_result = "../result/text/" + task + "_result.txt"
    if os.path.isfile(file_result):
        os.remove(file_result)

def main(firstparam):
    logger = log()

    file_path, task = find_file(firstparam)

    prepare_dir(task)

    # trainX, validateX, testX, trainY, validateY, testY = data_split(file_path)
    trainX, testX, trainY, testY = data_split(file_path)
    logger.info('Finished randomly splitting data set'+'\n')

    gnb, dt, lr, rfc, vc, AdaDT, bagging_lr = classifiers()

    # Result
    results = []
    for clf, name in ((gnb, 'Gaussian Naive Bayes'),
                      (dt, 'Decision Tree'),
                      (lr, 'Logistic Regression'),
                      (rfc, 'Random Forest'),
                      (vc, 'Voting Classifier'),
                      (AdaDT, 'Adaboosting classifier'),
                      # (svc, 'SVM  classifier'),
                      (bagging_lr, 'Bagging')):
        results.append(benchmark(clf, trainX, trainY, testX, testY, logger, task))
        # if clf == lr:
        #     print(clf.coef_)

    drawModelComparison(results, task)

    y_test = []
    y_score = []

    for classifier in (gnb,dt,lr,rfc,vc,AdaDT,bagging_lr):
        y_test.append(ROCplot(classifier, trainX, testX, trainY, testY)[0])
        y_score.append(ROCplot(classifier, trainX, testX, trainY, testY)[1])

    draw_roc(y_test, y_score, task)

if __name__ == "__main__":
    main(sys.argv[1])

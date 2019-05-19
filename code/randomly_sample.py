# -*- coding: utf-8 -*-
'''
Randomly sampling as baseline

Author:
organize all code: S1802373, S1809576
'''
import numpy as np
from scipy.stats import bernoulli
import sys
import os
from tool import log
from split_data import data_split
from build_model import *
from draw_plot import *

def random_sample(size):
    np.random.seed(0)
    np.set_printoptions(threshold=np.inf)
    random_sam = np.zeros(size)
    p = 0.9
    s = size
    a = bernoulli.rvs(p, loc=0, size=s, random_state=None)
    yes = 0
    for i in range(size):
        if a[i] == 1:
            random_sam[i] = 1.0
        elif a[i] == 0:
            random_sam[i] = -1.0
    return random_sam

def random_compare(testY):
    size = len(testY)
    random_sam = random_sample(size)
    score = float(accuracy_score(testY, random_sam, normalize=False)) / len(testY)
    print('Accuracy: ', score)

def find_file(param):
    _appetency = '../data/feature/small_train_appetency.csv'
    _churn = '../data/feature/small_train_churn.csv'
    _upselling = '../data/feature/small_train_upselling.csv'

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

    trainX, validateX, testX, trainY, validateY, testY = data_split(file_path)
    logger.info('Finished randomly splitting data set'+'\n')

    random_compare(testY)

    # gnb, lr, rfc, vc, AdaDT, bagging_lr = classifiers()
    #
    # Result
    # results = []
    # for clf, name in ((gnb, 'Gaussian Naive Bayes'),
    #                   (lr, 'Logistic Regression'),
    #                   (rfc, 'Random Forest'),
    #                   (vc, 'Voting Classifier'),
    #                   (AdaDT, 'Adaboosting classifier'),
    #                   # (svc, 'SVM  classifier'),
    #                   (bagging_lr, 'Bagging')):
        # results.append(benchmark(clf, trainX, trainY, testX, testY, logger, task))
    #
    # drawModelComparison(results, task)
    #
    # y_test = []
    # y_score = []
    #
    # for classifier in (gnb,lr,rfc,vc,AdaDT,bagging_lr):
    #     y_test.append(ROCplot(classifier, trainX, testX, trainY, testY)[0])
    #     y_score.append(ROCplot(classifier, trainX, testX, trainY, testY)[1])
    #
    # draw_roc(y_test, y_score, task)

if __name__ == "__main__":
    main(sys.argv[1])

# -*- coding: utf-8 -*-
'''
Draw ROC curve and model comparison

Author:
organize all code: S1802373
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def drawModelComparison(results, task):
    indices = np.arange(len(results))
    results = [[result[i] for result in results] for i in range(4)]
    clf, score, train_time, test_time = results

    train_time = np.array(train_time) / np.max(train_time)
    test_time = np.array(test_time) / np.max(test_time)

    plt.figure(figsize=(12, 8))
    plt.title("Model Comparison for " + task + " labels")
    plt.barh(indices, score, .2, label="score", color='navy')
    plt.barh(indices + .3, train_time, .2, label="training time", color='c')
    plt.barh(indices + .6, test_time, .2, label="test time", color='darkorange')
    plt.yticks(())
    plt.legend(loc='best')
    plt.subplots_adjust(left=.25)
    plt.subplots_adjust(top=.95)
    plt.subplots_adjust(bottom=.05)

    for i, c in zip(indices, clf):
        plt.text(-.3, i, c)

    plt.savefig("../result/graph/Model_Comparison_" + task + ".pdf")

def ROCplot(classifier, trainX, testX, trainY, testY):
    y_score = classifier.fit(trainX, trainY).predict_proba(testX)

    # y_test = testY.as_matrix()
    y_test = np.array(testY)
    y_test[y_test == 'no'] = 0
    y_test[y_test == 'yes'] = 1
    y_test = y_test.T
    return y_test, y_score

def draw_roc(ytest, yscore, task):
    plt.figure()
    label = ['gnb','dt','lr','rfc','vc','AdaDT','bagging']
    color = ['red', 'navy', 'black', 'yellow', 'aqua', 'darkorange', 'cornflowerblue']
    for i in range(len(ytest)):
        y_test = ytest[i]
        y_score = yscore[i]
        # fpr, tpr, _ = roc_curve(y_test, y_score[:, 1], pos_label=2)
        fpr, tpr, _ = roc_curve(y_test.tolist(), y_score[:, 1])
        roc_auc = auc(fpr, tpr)
        lw = 2
        plt.plot(fpr, tpr, color=color[i],
             lw=lw, label= label[i]+' ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC ' + task)
        plt.legend(loc="lower right")
    plt.savefig("../result/graph/ROC_" + task + ".pdf")

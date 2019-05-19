# -*- coding: utf-8 -*-
'''
Generate classification model

Author:
organize all code: S1802373
'''
# Refer to
# Hui Ge - Fit Naive Bayes Model
# Train and test your model using (trainX, trainY), (validateX,validateY), (testX, testY)
# import os
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from time import time
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
# from sklearn.metrics import roc_auc_score

def benchmark(clf, trainX, trainY, testX, testY, logger, task):
    clf_descr = str(clf).split('(')[0]
    logger.info('Start to fit %s' % clf_descr)
    # Train the clf, and record the training time
    t0 = time()
    clf.fit(trainX, trainY)
    train_time = time() - t0
    print('Training time: %0.3fs' % train_time)

    # Fit the clf to the test dataset, and record the testing time
    t0 = time()
    predict = clf.predict(testX)
    test_time = time() - t0
    print('Testing time: %0.3fs' % test_time)

    _yes = 0
    _no = 0
    for i in predict:
        if i == 1.0:
            _yes += 1
        elif i == -1.0:
            _no += 1
    # print('yes: ', _yes)
    # print('no: ', _no)

    yes_ = 0
    no_ = 0
    for j in testY:
        if j == 1.0:
            yes_ += 1
        elif j == -1.0:
            no_ += 1

    score = float(accuracy_score(testY, predict, normalize=False)) / len(testY)
    print('Accuracy of {0}: {1:.2%}'.format(clf_descr, score))
    con_mat = confusion_matrix(testY, predict)
    logger.info('Finished fitting %s' % clf_descr +'\n')

    file_name = "../result/text/" + task + "_result.txt"
    with open(file_name, "a") as f:
        f.write("Training time: %0.3fs" % train_time + "\n")
        f.write("Testing time: %0.3fs" % test_time + "\n")
        f.write("Accuracy of {0}: {1:.2%}".format(clf_descr, score) + "\n")
        f.write("yes_test: " + str(yes_) + "     no_test: " + str(no_) + "\n")
        f.write("yes_predict: " + str(_yes) + "     no_predict: " + str(_no) + "\n")
        f.write("confusion matrix: " + "\n")
        f.write(str(con_mat[0]) + "\n")
        f.write(str(con_mat[1]) + "\n")

    return clf_descr, score, train_time, test_time

def classifiers():
    # Gaussian Naive Bayes
    gnb = GaussianNB()

    # Decision Tree
    dt = DecisionTreeClassifier(random_state=0)
    # parameter_grid = {'max_depth': [1, 2, 3, 4, 5], 'max_features': [1, 2, 3, 4]}

    # Logistic Regression
    lr = LogisticRegression(C=10, solver='sag', tol=0.1)
    # lr_param = [{'tol': [0.1, 1e-2, 1e-3, 1e-4], 'C': [1, 10, 100, 1000], 'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag']}]
    # scores = ['precision', 'recall']
    # for score in scores:
    #     print("# Tuning hyper-parameters for %s" % score)
    #     print()
    #
    #     lr = GridSearchCV(LogisticRegression(), lr_param, cv=5, scoring='%s_macro' % score)
    #     lr.fit(trainX, trainY)
    #     print("Best parameters set found on development set:")
    #     print()
    #     print(lr.best_params_)
    #     print()

    # Random Forest Classifier
    # rfc = RandomForestClassifier(max_depth=15, n_estimators=20)
    rfc = RandomForestClassifier(max_depth=5, max_features=1, n_estimators=20)
    # rfc = RandomForestClassifier(max_depth=5, max_features=10, n_estimators=10)
    # rfc_param = [{'n_estimators': [10, 20, 30, 50, 100, 200], 'max_depth': [5, 8, 10, 15]}]
    # scores = ['precision', 'recall']
    # for score in scores:
    #     print("# Tuning hyper-parameters for %s" % score)
    #     print()
    #
    #     rfc = GridSearchCV(RandomForestClassifier(), rfc_param, cv=5, scoring='%s_macro' % score)
    #     rfc.fit(trainX, trainY)
    #     print("Best parameters set found on development set:")
    #     print()
    #     print(rfc.best_params_)
    #     print()

    # Voting classifier
    vc = VotingClassifier(estimators=[('lr', lr), ('gnb', gnb), ('rfc', rfc)], voting='soft')

    # Adaboosting classifier
    # AdaDT = AdaBoostClassifier(DecisionTreeClassifier(max_depth=8),n_estimators=10,learning_rate=0.5)
    AdaDT = AdaBoostClassifier(DecisionTreeClassifier(max_depth=8),n_estimators=50,learning_rate=0.05)
    # AdaDT_param = [{'n_estimators': [10, 20, 30, 50], 'max_depth': [5, 8, 10, 15], 'learning_rate': [0.05, 0.1, 0.2, 0.5]}]
    # scores = ['precision', 'recall']
    # for score in scores:
    #     print("# Tuning hyper-parameters for %s" % score)
    #     print()
    #
    #     AdaDT = GridSearchCV(AdaBoostClassifier(DecisionTreeClassifier()))
    #     AdaDT.fit(trainX, trainY)
    #     print("Best parameters set found on development set:")
    #     print()
    #     print(AdaDT.best_params_)
    #     print()

    # Bagging classifier
    bagging_lr = BaggingClassifier(LogisticRegression(C=10, solver='sag', tol=0.1))

    # #SVM classifier
    # svc = SVC(class_weight='balanced',kernel='poly',C=10.0)

    return gnb, dt, lr, rfc, vc, AdaDT, bagging_lr

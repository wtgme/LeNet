# -*- coding: utf-8 -*-
"""
Created on 15:25, 06/10/16

@author: wt
"""

import multiprocessing
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import numpy as np


def classifier_cv(X, y, K=5):
    skf = StratifiedKFold(n_splits=K)
    accuracys = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        logistic = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg', n_jobs=multiprocessing.cpu_count())
        # svc_lin = SVC(kernel='linear', class_weight='balanced')
        y_lin = logistic.fit(X_train, y_train).predict(X_test)
        score = accuracy_score(y_lin, y_test)
        print "Fold Accuracy: %0.4f" % (score)
        accuracys.append(score)
    print("Overall Accuracy: %0.4f (+/- %0.4f)" % (np.mean(accuracys), np.std(accuracys)))

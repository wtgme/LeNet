# -*- coding: utf-8 -*-
"""
Created on 4:52 PM, 5/2/16

@author: tw
"""
import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
from gensim.models import Word2Vec
import numpy as np
from sklearn import cross_validation
from collections import OrderedDict
from sklearn.svm import SVC

def svm_cv(X, y, kernel='linear'):
    print X.shape
    print y.shape
    # Cross validation with SVM
    clf = SVC(kernel=kernel, class_weight='balanced')
    #When the cv argument is an integer, cross_val_score
    # uses the KFold or StratifiedKFold strategies by default,
    # the latter being used if the estimator derives from ClassifierMixin.
    scores = cross_validation.cross_val_score(clf, X, y, scoring='accuracy', cv=5, n_jobs=5)
    print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std()))


def word_vect(filename):
    documents = []
    with open(filename, 'r') as fo:
        for line in fo.readlines():
            tokens = line.strip().split('\t')
            if tokens[2] != str(0):
                documents.append(tokens[1].split())

    simple_models = [Word2Vec(documents, size=100, window=5, min_count=1, workers=8, sg=0, cbow_mean=0), # CBOW, SUM
              Word2Vec(documents, size=100, window=5, min_count=1, workers=8, sg=0, cbow_mean=1), # CBOW, mean
              Word2Vec(documents, size=100, window=5, min_count=1, workers=8, sg=1), # skip-gram
            ]
    models_by_name = OrderedDict((str(i), simple_models[i]) for i in xrange(len(simple_models)))
    print len(models_by_name)
    for name, model in models_by_name.items():
        print name
        X, y = [], []
        with open(filename, 'r') as fo:
            for line in fo.readlines():
                tokens = line.strip().split('\t')
                if tokens[2] != str(0):
                    words = tokens[1].split()
                    values = []
                    for word in words:
                        values.append(model[word])
                    values = np.array(values)
                    X.append(np.mean(values, axis=0))
                    y.append(int(tokens[2]))
        svm_cv(np.array(X), np.array(y))

if __name__ == '__main__':
    word_vect('data/cora.data')
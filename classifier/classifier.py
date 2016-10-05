# -*- coding: utf-8 -*-
"""
Created on 10:40 PM, 10/4/16

@author: tw
"""
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn import cross_validation
from sklearn.svm import SVC


def readfile(filename):
    datamap = {}
    with open(filename, 'r') as fo:
        for line in fo.readlines():
            tokens = line.strip().split('\t')
            datamap[tokens[0]] = tokens[1]
    return datamap


def unify_files(textfile, idlabelfile):
    id_text = readfile(textfile)
    id_label = readfile(idlabelfile)
    labelmap = {}
    for id in id_text.keys():
        text = id + '\t' + id_text[id] + '\t'
        label = id_label.get(id, None)
        if label is not None:
            lid = labelmap.get(label, None)
            if lid is None:
                lid = len(labelmap)+1
                labelmap[label] = lid
            text += str(lid)
        else:
            text += str(0)
        print text
    print labelmap

def transform(textfile, idlabelfile):
    id_text = readfile(textfile)

    '''Tranform from word vector'''
    corpus = []
    for id in id_text.keys():
        corpus.append(id_text[id])
    vectorizer = TfidfVectorizer(min_df=1)
    X = vectorizer.fit_transform(corpus).toarray()

    id_label = readfile(idlabelfile)
    Y, labelmap = [], {}
    for id in id_text.keys():
        label = id_label.get(id, None)
        if label is not None:
            lid = labelmap.get(label, None)
            if lid is None:
                lid = len(labelmap)+1
                labelmap[label] = lid
            Y.append(lid)
        else:
            Y.append(np.nan)
    y = np.array(Y)
    return X[~np.isnan(y)], y[~np.isnan(y)]


def read_SVM_format(vectfile):
    X_train, y_train = load_svmlight_file(vectfile)
    X_train = X_train.toarray()
    return X_train, y_train



def svm_cv(X, y, kernel='linear'):
    # Cross validation with SVM
    clf = SVC(kernel=kernel, class_weight='balanced')
    #When the cv argument is an integer, cross_val_score
    # uses the KFold or StratifiedKFold strategies by default,
    # the latter being used if the estimator derives from ClassifierMixin.
    scores = cross_validation.cross_val_score(clf, X, y, scoring='accuracy', cv=5, n_jobs=5)
    print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std()))

    # scores = cross_validation.cross_val_score(clf, X, y, scoring='precision_weighted', cv=5, n_jobs=5)
    # print("Precision: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std()))
    #
    # scores = cross_validation.cross_val_score(clf, X, y, scoring='recall_weighted', cv=5, n_jobs=5)
    # print("Recall: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std()))
    #
    # scores = cross_validation.cross_val_score(clf, X, y, scoring='f1_weighted', cv=5, n_jobs=5)
    # print("F1: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std()))

    # scores = cross_validation.cross_val_score(clf, X, y, scoring='roc_auc_weighted', cv=5, n_jobs=5)
    # print("AUC: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std()))

    return scores

if __name__ == '__main__':

    '''Row text: word vector'''
    # X, y = transform('data/text.data', 'data/idlabel.data')
    # print X.shape
    # print y.shape
    # svm_cv(X, y)

    '''Doc2vec SVM'''
    X, y = read_SVM_format('data/doc2vec.data')
    print X.shape
    print y.shape
    svm_cv(X, y)



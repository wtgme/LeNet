# -*- coding: utf-8 -*-
"""
Created on 4:52 PM, 5/2/16

@author: tw
"""
import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))

import pickle
from collections import OrderedDict
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import multiprocessing
from sklearn import linear_model
import util


def label_generate(filename, starK=10, endK=11):
    """Pre-cluster data, and assign predicted labels to each sample"""
    ids, texts, labels = [], [], []
    with open(filename, 'r') as fo:
        for line in fo.readlines():
            tokens = line.strip().split('\t')
            if tokens[2] != str(0):
                ids.append(tokens[0])
                texts.append(tokens[1])
                labels.append(tokens[2])
    df = pd.DataFrame({'ID': ids,
                       'Text': texts,
                       'Label': labels})
    # print df
    vectorizer = TfidfVectorizer(min_df=1)
    # vectorizer = CountVectorizer(min_df=1)
    X = vectorizer.fit_transform(df['Text']).toarray()
    print X.shape
    label_list, scores = [], []
    for k in xrange(starK, endK):
        # kmeans_model = KMeans(n_clusters=k, random_state=1).fit(X)
        kmeans_model = MiniBatchKMeans(n_clusters=k, init='k-means++',
                         init_size=1000, batch_size=1000, compute_labels=True).fit(X)
        labels = kmeans_model.labels_
        score = metrics.calinski_harabaz_score(X, labels)
        label_list.append(labels)
        scores.append(score)
    # print scores
    max_score = max(scores)
    index = scores.index(max_score)
    df['Predict'] = label_list[index]
    print 'Best K:', range(starK, endK)[index]
    return df


def doc_vect(filename):
    """Using ground-truth labels or predicted labels in Word2Vec"""
    df = label_generate(filename)
    names = list(df.columns.values)
    # print names
    textid = names.index('Text')
    predid = names.index('Label')
    documents = []
    from gensim.models.doc2vec import TaggedDocument
    for line in df.itertuples():
        # print line
        # print line[textid+1]
        # print line[predid+1]
        sentence = TaggedDocument(line[textid+1].split(), [str(line[predid+1])])
        documents.append(sentence)
    print len(documents)
    from gensim.models.doc2vec import Doc2Vec
    import multiprocessing
    cores = multiprocessing.cpu_count()
    simple_models = [
                # PV-DM w/concatenation - window=5 (both sides) approximates paper's 10-word total window size
                Doc2Vec(documents, dm=1, dm_concat=1, size=100, window=5, negative=5, hs=0, min_count=1, workers=cores),
                # PV-DBOW
                Doc2Vec(documents, dm=0, size=100, negative=5, hs=0, min_count=1, workers=cores),
                # PV-DM w/average
                Doc2Vec(documents, dm=1, dm_mean=1, size=100, window=5, negative=5, hs=0, min_count=1, workers=cores),
                    ]

    models_by_name = OrderedDict((str(model), model) for model in simple_models)
    from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
    models_by_name['dbow+dmm'] = ConcatenatedDoc2Vec([simple_models[1], simple_models[2]])
    models_by_name['dbow+dmc'] = ConcatenatedDoc2Vec([simple_models[1], simple_models[0]])

    pickle.dump(models_by_name, open('data/doc2vec.pick', 'w'))



def classification(filename):
    models_by_name = pickle.load(open('data/doc2vec.pick', 'r'))
    for name, model in models_by_name.items():
        print name
        X, y = [], []
        with open(filename, 'r') as fo:
            for line in fo.readlines():
                tokens = line.strip().split('\t')
                if tokens[2] != str(0):
                    words = tokens[1].split()
                    docvect = model.infer_vector(words)
                    # docvect = model.docvecs[tokens[0]]
                    X.append(docvect)
                    y.append(int(tokens[2]))
        util.classifier_cv(np.array(X), np.array(y))

'''-----------------------------------------------------------
Modify to only use labels of training data
'''
def read_data(filename):
    X, y = [], []
    with open(filename, 'r') as fo:
        for line in fo.readlines():
            tokens = line.strip().split('\t')
            if tokens[2] != str(0):
                X.append(tokens[1])
                y.append(int(tokens[2]))
    return np.array(X), np.array(y)


def vect(X_train, y_train):
    documents = []
    from gensim.models.doc2vec import TaggedDocument
    for line in zip(X_train, y_train):
        sentence = TaggedDocument(line[0].split(), [str(line[1])])
        documents.append(sentence)
    from gensim.models.doc2vec import Doc2Vec
    import multiprocessing
    cores = multiprocessing.cpu_count()
    simple_models = [
                # PV-DM w/concatenation - window=5 (both sides) approximates paper's 10-word total window size
                Doc2Vec(documents, dm=1, dm_concat=1, size=100, window=5, negative=5, hs=0, min_count=1, workers=cores),
                # PV-DBOW
                Doc2Vec(documents, dm=0, size=100, negative=5, hs=0, min_count=1, workers=cores),
                # PV-DM w/average
                Doc2Vec(documents, dm=1, dm_mean=1, size=100, window=5, negative=5, hs=0, min_count=1, workers=cores),
                    ]

    models_by_name = OrderedDict((str(model), model) for model in simple_models)
    from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
    models_by_name['dbow+dmm'] = ConcatenatedDoc2Vec([simple_models[1], simple_models[2]])
    models_by_name['dbow+dmc'] = ConcatenatedDoc2Vec([simple_models[1], simple_models[0]])
    return models_by_name


def train_label_classification(X, y):
    skf = StratifiedKFold(n_splits=5)
    accuracys = []
    for train_index, test_index in skf.split(X, y):
        acc_list = []
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        vect_models = vect(X_train, y_train)

        for name, model in vect_models.items():
            X_train_vec, X_test_vec = [], []
            for x in X_train:
                words = x.split()
                docvect = model.infer_vector(words)
                X_train_vec.append(docvect)
            for x in X_test:
                words = x.split()
                docvect = model.infer_vector(words)
                X_test_vec.append(docvect)
            logistic = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg', n_jobs=multiprocessing.cpu_count())
            # svc_lin = SVC(kernel='linear', class_weight='balanced')
            print len(X_train_vec), len(X_test_vec), len(y_train)
            y_lin = logistic.fit(np.array(X_train_vec), y_train).predict(np.array(X_test_vec))
            score = accuracy_score(y_lin, y_test)
            # print "Fold Accuracy: %0.4f" % (score)
            acc_list.append(score)
        accuracys.append(acc_list)
    accuracys = np.array(accuracys)
    i = 0
    for name, model in vect_models.items():
        print name
        print("Overall Accuracy: %0.4f (+/- %0.4f)" % (np.mean(accuracys[:, i]), np.std(accuracys[:, i])))
        i += 1


def vect_all(X_train, y_train, X_test, y_predict):
    documents = []
    from gensim.models.doc2vec import TaggedDocument
    for line in zip(X_train, y_train):
        sentence = TaggedDocument(line[0].split(), [str(line[1])])
        documents.append(sentence)
    for line in zip(X_test, y_predict):
        sentence = TaggedDocument(line[0].split(), [str(line[1])])
        documents.append(sentence)
    print len(documents)
    from gensim.models.doc2vec import Doc2Vec

    cores = multiprocessing.cpu_count()
    simple_models = [
                # PV-DM w/concatenation - window=5 (both sides) approximates paper's 10-word total window size
                Doc2Vec(documents, dm=1, dm_concat=1, size=100, window=5, negative=5, hs=0, min_count=1, workers=cores),
                # PV-DBOW
                Doc2Vec(documents, dm=0, size=100, negative=5, hs=0, min_count=1, workers=cores),
                # PV-DM w/average
                Doc2Vec(documents, dm=1, dm_mean=1, size=100, window=5, negative=5, hs=0, min_count=1, workers=cores),
                    ]

    models_by_name = OrderedDict((str(model), model) for model in simple_models)
    from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
    models_by_name['dbow+dmm'] = ConcatenatedDoc2Vec([simple_models[1], simple_models[2]])
    models_by_name['dbow+dmc'] = ConcatenatedDoc2Vec([simple_models[1], simple_models[0]])
    return models_by_name


def pre_classify(X_train, y_train, X_test, y_test):
    corpus = np.append(X_train, X_test)
    vectorizer = TfidfVectorizer(min_df=1)
    X = vectorizer.fit_transform(corpus).toarray()
    """SVM classifier: too slow"""
    # svc_lin = SVC(kernel='linear', class_weight='balanced')
    # y_lin = svc_lin.fit(X[:len(X_train), :], y_train).predict(X[len(X_train):, :])
    """Parallel KNN: more fast"""
    neigh = KNeighborsClassifier(n_neighbors=5, n_jobs=multiprocessing.cpu_count())
    y_lin = neigh.fit(X[:len(X_train), :], y_train).predict(X[len(X_train):, :])
    print "Pre-classification accuracy: %0.4f" % accuracy_score(y_lin, y_test)
    return y_lin


def pre_class_classification(X, y):
    skf = StratifiedKFold(n_splits=5)
    accuracys = []
    for train_index, test_index in skf.split(X, y):
        acc_list = []
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        y_lin = pre_classify(X_train, y_train, X_test, y_test)
        vect_models = vect_all(X_train, y_train, X_test, y_lin)

        for name, model in vect_models.items():
            X_train_vec, X_test_vec = [], []
            for x in X_train:
                words = x.split()
                docvect = model.infer_vector(words)
                X_train_vec.append(docvect)
            for x in X_test:
                words = x.split()
                docvect = model.infer_vector(words)
                X_test_vec.append(docvect)
            logistic = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg', n_jobs=multiprocessing.cpu_count())
            # svc_lin = SVC(kernel='linear', class_weight='balanced')
            y_lin = logistic.fit(np.array(X_train_vec), y_train).predict(np.array(X_test_vec))
            score = accuracy_score(y_lin, y_test)
            print "Fold Accuracy: %0.4f" % (score)
            acc_list.append(score)
        accuracys.append(acc_list)
    accuracys = np.array(accuracys)
    i = 0
    for name, model in vect_models.items():
        print name
        print("Overall Accuracy: %0.4f (+/- %0.4f)" % (np.mean(accuracys[:, i]), np.std(accuracys[:, i])))
        i += 1


def train_label_run(filename):
    X, y = read_data(filename)
    train_label_classification(X, y)


def pre_class_run(filename):
    X, y = read_data(filename)
    pre_class_classification(X, y)

if __name__ == '__main__':

    # '''Doc2Vec traing'''
    # doc_vect('data/cora.data')
    #
    # # '''Verify model'''
    # # varify()
    #
    # # '''Out put files'''
    # # output('data/cora.data')
    # classification('datalea/cora.data')

    print 'Only user training data for label'
    train_label_run('data/cora.data')
    print 'Pre-classify test data first, and then all label training'
    pre_class_run('data/cora.data')




# -*- coding: utf-8 -*-
"""
Created on 4:52 PM, 5/2/16

@author: tw
"""
from gensim.models import Word2Vec
import numpy as np
from collections import OrderedDict
import util


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
        util.classifier_cv(np.array(X), np.array(y))

if __name__ == '__main__':
    word_vect('data/cora.data')
# -*- coding: utf-8 -*-
"""
Created on 10:40 PM, 10/4/16

@author: tw
"""
from sklearn.feature_extraction.text import TfidfVectorizer


def readfile(filename):
    datamap = {}
    with open(filename, 'r') as fo:
        for line in fo.readlines():
            tokens = line.strip().split('\t')
            datamap[tokens[0]] = tokens[1]
    return datamap


def transform():
    id_text = readfile('text.data')

    corpus = []
    for id in id_text.keys():
        corpus.append(id_text[id])
    vectorizer = TfidfVectorizer(min_df=1)
    X = vectorizer.fit_transform(corpus).toarray()
    print X.shape

    id_label = readfile('idlabel.data')
    Y, labelmap = [], {}
    for id in id_text.keys():
        label = id_label.get(id, None)
        if label is not None:
            lid = labelmap.get(label, None)
            if lid is None:
                lid = len(labelmap)+1
                labelmap[label] = len(labelmap)+1
            Y.append(lid)
        else:
            Y.append(None)
    return Y



if __name__ == '__main__':
    transform()


# -*- coding: utf-8 -*-
"""
Created on 22:12, 06/10/16

@author: wt
"""

from gensim.models.doc2vec import Doc2Vec
import multiprocessing
from gensim.models.doc2vec import TaggedDocument
from scipy import spatial

def test():
    documents = []
    documents.append(TaggedDocument('this book is good'.split(), ['1']))
    documents.append(TaggedDocument('my rooms are good'.split(), ['1']))
    documents.append(TaggedDocument('this book is bad'.split(), ['0']))
    documents.append(TaggedDocument('my rooms are bad'.split(), ['0']))
    # documents.append(TaggedDocument('you look good'.split(), ['1']))
    # documents.append(TaggedDocument('you look bad'.split(), ['0']))
    # documents.append(TaggedDocument('my rooms are good'.split(), ['1']))
    # documents.append(TaggedDocument('my rooms are bad'.split(), ['0']))

    model = Doc2Vec(documents, dm=1, dm_concat=1, size=10, window=5, negative=5, hs=0, min_count=1, workers=1)
    print('%s:\n %s' % (model, model.docvecs.most_similar('1')))
    v1 = model.infer_vector('this book is good'.split())
    v2 = model.infer_vector('this book is bad'.split())
    print v1, v2
    print (1 - spatial.distance.cosine(v1, v2))

def test2():
    documents = []
    documents.append(TaggedDocument('this book is good'.split(), ['0']))
    documents.append(TaggedDocument('this book is bad'.split(), ['1']))
    documents.append(TaggedDocument('my rooms are good'.split(), ['2']))
    documents.append(TaggedDocument('my rooms are bad'.split(), ['3']))
    # documents.append(TaggedDocument('my rooms are good'.split(), ['0']))
    # documents.append(TaggedDocument('my rooms are bad'.split(), ['0']))
    # documents.append(TaggedDocument('my rooms are good'.split(), ['0']))
    # documents.append(TaggedDocument('my rooms are bad'.split(), ['0']))

    model = Doc2Vec(documents, dm=1, dm_concat=1, size=10, window=5, negative=5, hs=0, min_count=1, workers=1)
    print('%s:\n %s' % (model, model.docvecs.most_similar('0')))
    v1 = model.infer_vector('this book is good'.split())
    v2 = model.infer_vector('this book is bad'.split())
    print v1, v2
    print (1 - spatial.distance.cosine(v1, v2))

test()
test2()
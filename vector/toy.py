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
    documents.append(TaggedDocument('this book is good , I like it'.split(), ['g']))
    documents.append(TaggedDocument('this room is good , I like it'.split(), ['g']))
    documents.append(TaggedDocument('this book is bad , I do not like it'.split(), ['b']))
    documents.append(TaggedDocument('this room is bad , I do not like it'.split(), ['b']))
    # documents.append(TaggedDocument('you look good'.split(), ['1']))
    # documents.append(TaggedDocument('you look bad'.split(), ['0']))
    # documents.append(TaggedDocument('my rooms are good'.split(), ['1']))
    # documents.append(TaggedDocument('my rooms are bad'.split(), ['0']))

    model = Doc2Vec(documents, dm=1, dm_concat=1, size=100, window=3, negative=5, hs=0, iter=50, min_count=1, workers=1)
    print('%s:\n %s' % (model, model.docvecs.most_similar('g')))
    v1 = model.infer_vector('this book is good'.split())
    v2 = model.infer_vector('this book is bad'.split())
    # print (1 - spatial.distance.cosine(v1, v2))
    print model.similarity('bad', 'good')
    print model.similarity('book', 'room')
    # print model.similarity('book', 'room') - model.similarity('bad', 'good')
    # print model.docvecs.similarity('1', '0')


def test2():
    documents = []
    documents.append(TaggedDocument('this book is good , I like it'.split(), ['0']))
    documents.append(TaggedDocument('this room is good , I like it'.split(), ['1']))
    documents.append(TaggedDocument('this book is bad , I do not like it'.split(), ['2']))
    documents.append(TaggedDocument('this room is bad , I do not like it'.split(), ['3']))
    # documents.append(TaggedDocument('my rooms are good'.split(), ['0']))
    # documents.append(TaggedDocument('my rooms are bad'.split(), ['0']))
    # documents.append(TaggedDocument('my rooms are good'.split(), ['0']))
    # documents.append(TaggedDocument('my rooms are bad'.split(), ['0']))

    model = Doc2Vec(documents, dm=1, dm_concat=1, size=100, window=3, negative=5, hs=0, iter=50, min_count=1, workers=1)
    print('%s:\n %s' % (model, model.docvecs.most_similar('0')))
    v1 = model.infer_vector('this book is good'.split())
    v2 = model.infer_vector('this book is bad'.split())
    # print (1 - spatial.distance.cosine(v1, v2))
    print model.similarity('bad', 'good')
    print model.similarity('book', 'room')
    # print model.similarity('book', 'room') - model.similarity('bad', 'good')
    # print model.docvecs.similarity('0', '1')
    # print model.docvecs.similarity('0', '2')

test()
test2()
# print 0.884254323202 - 0.745608090702
# print 0.912312923256 - 0.866802435586
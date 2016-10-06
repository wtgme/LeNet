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
import util



def doc_vect(filename):
    documents = []
    from gensim.models.doc2vec import TaggedDocument
    with open(filename, 'r') as fo:
        for line in fo.readlines():
            tokens = line.strip().split('\t')
            if tokens[2] != str(0):
                sentence = TaggedDocument(tokens[1].split(), [tokens[2]])
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

    # # speed setup by sharing results of 1st model's vocabulary scan
    # simple_models[0].build_vocab(documents)  # PV-DM/concat requires one special NULL word so it serves as template
    # print(simple_models[0])
    # for model in simple_models[1:]:
    #     model.reset_from(simple_models[0])
    #     print(model)
    for model in simple_models[:1]:
        print model
        for label in range(1, 11):
            inferred_docvec = model.docvecs[str(label)]
            print label
            print('%s:\n %s' % (model, model.most_similar(str(label))))

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


def output(filename):
    models_by_name = pickle.load(open('data/doc2vec.pick', 'r'))
    with open(filename, 'r') as fo:
        for line in fo.readlines():
            tokens = line.strip().split('\t')
            if tokens[2] != str(0):
                words = tokens[1].split()
                for name, model in models_by_name.items():
                    if '+' not in name:
                        docvect = model.infer_vector(words)
                        print tokens[0], tokens[2]
                        print('%s:\n %s' % (model, model.docvecs.most_similar([docvect], topn=3)))

                # sb = tokens[2]
                # for i in xrange(len(docvect)):
                #     sb += ' '+str(i+1)+':'+str(docvect[i])
                # print sb


if __name__ == '__main__':
    '''Doc2Vec traing'''
    doc_vect('data/cora.data')

    # '''Verify model'''
    # varify()

    # '''Out put files'''
    # output('data/cora.data')
    # classification('data/cora.data')

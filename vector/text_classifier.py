# -*- coding: utf-8 -*-
"""
Created on 15:03, 11/10/16

@author: wt
"""

from collections import OrderedDict
from gensim.models.doc2vec import Doc2Vec
import multiprocessing
from gensim.models.doc2vec import TaggedDocument
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
import util
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def doc_vect(alldocs):
    print 'Doc2Vec with lineNO as ID'
    train_docs = [doc for doc in alldocs if doc.split == 'train']
    test_docs = [doc for doc in alldocs if doc.split == 'test']
    print('%d docs: %d train-sentiment, %d test-sentiment' % (len(alldocs), len(train_docs), len(test_docs)))
    documents = []
    for doc in alldocs:
        sentence = TaggedDocument(doc.words, doc.tags)
        documents.append(sentence)
    print len(documents)
    cores = multiprocessing.cpu_count()
    simple_models = [
                # PV-DM w/concatenation - window=5 (both sides) approximates paper's 10-word total window size
                Doc2Vec(documents, dm=1, dm_concat=1, size=400, window=5, negative=5, hs=1, sample=1e-3, iter=20, min_count=1, workers=cores),
                # PV-DBOW
                Doc2Vec(documents, dm=0, size=400, window=5, negative=5, hs=1, sample=1e-3, iter=20, min_count=1, workers=cores),
                # PV-DM w/average
                Doc2Vec(documents, dm=1, dm_mean=1, size=400, window=5, negative=5, hs=1, sample=1e-3, iter=20, min_count=1, workers=cores),
                    ]

    models_by_name = OrderedDict((str(model), model) for model in simple_models)
    models_by_name['dbow+dmm'] = ConcatenatedDoc2Vec([simple_models[1], simple_models[2]])
    models_by_name['dbow+dmc'] = ConcatenatedDoc2Vec([simple_models[1], simple_models[0]])

    for name, model in models_by_name.items():
        print name
        train_targets, train_regressors = zip(*[(doc.sentiment, model.docvecs[doc.tags[0]]) for doc in train_docs])
        test_targets, test_regressors = zip(*[(doc.sentiment, model.docvecs[doc.tags[0]]) for doc in test_docs])
        util.logit(train_regressors, train_targets, test_regressors, test_targets)


def label_doc_vect(alldocs):
    print 'Label-doc-vec'
    train_docs = [doc for doc in alldocs if doc.split == 'train']
    test_docs = [doc for doc in alldocs if doc.split == 'test']
    print('%d docs: %d train-sentiment, %d test-sentiment' % (len(alldocs), len(train_docs), len(test_docs)))
    documents = []
    for doc in alldocs:
        if doc.split == 'train':
            sentence = TaggedDocument(doc.words, doc.tags+['l'+str(doc.sentiment)])
        else:
            sentence = TaggedDocument(doc.words, doc.tags)
        documents.append(sentence)
    print len(documents)
    cores = multiprocessing.cpu_count()
    simple_models = [
                # PV-DM w/concatenation - window=5 (both sides) approximates paper's 10-word total window size
                Doc2Vec(documents, dm=1, dm_concat=1, size=400, window=5, negative=5, hs=1, sample=1e-3, iter=20, min_count=1, workers=cores),
                # PV-DBOW
                Doc2Vec(documents, dm=0, size=400, window=5, negative=5, hs=1, sample=1e-3, iter=20, min_count=1, workers=cores),
                # PV-DM w/average
                Doc2Vec(documents, dm=1, dm_mean=1, size=400, window=5, negative=5, hs=1, sample=1e-3, iter=20, min_count=1, workers=cores),
                    ]

    models_by_name = OrderedDict((str(model), model) for model in simple_models)
    models_by_name['dbow+dmm'] = ConcatenatedDoc2Vec([simple_models[1], simple_models[2]])
    models_by_name['dbow+dmc'] = ConcatenatedDoc2Vec([simple_models[1], simple_models[0]])

    for name, model in models_by_name.items():
        print name
        train_targets, train_regressors = zip(*[(doc.sentiment, model.docvecs[doc.tags[0]]) for doc in train_docs])
        test_targets, test_regressors = zip(*[(doc.sentiment, model.docvecs[doc.tags[0]]) for doc in test_docs])
        util.logit(train_regressors, train_targets, test_regressors, test_targets)


def pre_class(train_docs, test_docs, non_docs):
    train_y, train_X = zip(*[(doc.sentiment, ' '.join(doc.words)) for doc in train_docs])
    test_y, test_X = zip(*[(doc.sentiment, ' '.join(doc.words)) for doc in test_docs+non_docs])
    y_lin = util.pre_classify_text(train_X, train_y, test_X, None)
    return y_lin


def label_vect(alldocs):
    print 'Label2Vec with pre-classification'
    train_docs = [doc for doc in alldocs if doc.split == 'train']
    test_docs = [doc for doc in alldocs if doc.split == 'test']
    non_docs = [doc for doc in alldocs if doc.split == 'extra']
    print('%d docs: %d train-sentiment, %d test-sentiment' % (len(alldocs), len(train_docs), len(test_docs)))
    ylin = pre_class(train_docs, test_docs, non_docs)
    documents = []
    for doc in train_docs:
        sentence = TaggedDocument(doc.words, [str(doc.sentiment)])
        documents.append(sentence)
    i = 0
    for doc in test_docs+non_docs:
        sentence = TaggedDocument(doc.words, [str(ylin[i])])
        documents.append(sentence)
        i += 1
    print len(documents)
    cores = multiprocessing.cpu_count()
    simple_models = [
                # PV-DM w/concatenation - window=5 (both sides) approximates paper's 10-word total window size
                Doc2Vec(documents, dm=1, dm_concat=1, size=100, window=10, negative=5, hs=1, sample=1e-3, min_count=1, workers=cores),
                # PV-DBOW
                Doc2Vec(documents, dm=0, size=100, window=10, negative=5, hs=1, sample=1e-3, min_count=1, workers=cores),
                # PV-DM w/average
                Doc2Vec(documents, dm=1, dm_mean=1, size=100, window=10, negative=5, hs=1, sample=1e-3, min_count=1, workers=cores),
                    ]

    models_by_name = OrderedDict((str(model), model) for model in simple_models)
    models_by_name['dbow+dmm'] = ConcatenatedDoc2Vec([simple_models[1], simple_models[2]])
    models_by_name['dbow+dmc'] = ConcatenatedDoc2Vec([simple_models[1], simple_models[0]])

    for name, model in models_by_name.items():
        print name
        train_targets, train_regressors = zip(*[(doc.sentiment, model.infer_vector(doc.words)) for doc in train_docs])
        test_targets, test_regressors = zip(*[(doc.sentiment, model.infer_vector(doc.words)) for doc in test_docs])
        util.logit(train_regressors, train_targets, test_regressors, test_targets)


def label_vect_no_class(alldocs):
    print 'Label2Vec without pre-classification'
    train_docs = [doc for doc in alldocs if doc.split == 'train']
    test_docs = [doc for doc in alldocs if doc.split == 'test']
    print('%d docs: %d train-sentiment, %d test-sentiment' % (len(alldocs), len(train_docs), len(test_docs)))
    documents = []
    for doc in alldocs:
        if doc.split == 'train':
            sentence = TaggedDocument(doc.words, [str(doc.sentiment)])
        else:
            sentence = TaggedDocument(doc.words, [])
        documents.append(sentence)
    print len(documents)
    cores = multiprocessing.cpu_count()
    simple_models = [
                # PV-DM w/concatenation - window=5 (both sides) approximates paper's 10-word total window size
                Doc2Vec(documents, dm=1, dm_concat=1, size=400, window=10, negative=5, hs=1, sample=1e-3, iter=20, min_count=1, workers=cores),
                # PV-DBOW
                Doc2Vec(documents, dm=0, size=400, window=10, negative=5, hs=1, sample=1e-3, iter=20, min_count=1, workers=cores),
                # PV-DM w/average
                Doc2Vec(documents, dm=1, dm_mean=1, size=400, window=10, negative=5, hs=1, sample=1e-3, iter=20, min_count=1, workers=cores),
                    ]

    models_by_name = OrderedDict((str(model), model) for model in simple_models)
    models_by_name['dbow+dmm'] = ConcatenatedDoc2Vec([simple_models[1], simple_models[2]])
    models_by_name['dbow+dmc'] = ConcatenatedDoc2Vec([simple_models[1], simple_models[0]])

    for name, model in models_by_name.items():
        print name
        train_targets, train_regressors = zip(*[(doc.sentiment, model.infer_vector(doc.words)) for doc in train_docs])
        test_targets, test_regressors = zip(*[(doc.sentiment, model.infer_vector(doc.words)) for doc in test_docs])
        util.logit(train_regressors, train_targets, test_regressors, test_targets)



if __name__ == '__main__':
    data = util.get_ng_data()
    doc_vect(data)
    # label_vect(data)
    # label_vect_no_class(data)
    label_doc_vect(data)
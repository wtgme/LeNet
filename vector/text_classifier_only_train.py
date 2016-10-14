# -*- coding: utf-8 -*-
"""
Created on 15:03, 11/10/16

@author: wt
"""

import locale
import glob
import os.path
import requests
import tarfile
from collections import OrderedDict
from gensim.models.doc2vec import Doc2Vec
import multiprocessing
from gensim.models.doc2vec import TaggedDocument
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
import util
import logging
from collections import namedtuple
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# Convert text to lower-case and strip punctuation/symbols from words
def normalize_text(text):
    norm_text = text.lower()
    # Replace breaks with spaces
    norm_text = norm_text.replace('<br />', ' ')
    # Pad punctuation with spaces on both sides
    for char in ['.', '"', ',', '(', ')', '!', '?', ';', ':']:
        norm_text = norm_text.replace(char, ' ' + char + ' ')
    return norm_text

# ##ONLY RUN with Python3.0
# def transform():
#     dirname = 'aclImdb'
#     filename = 'aclImdb_v1.tar.gz'
#     locale.setlocale(locale.LC_ALL, 'C')
#
#     # Convert text to lower-case and strip punctuation/symbols from words
#     def normalize_text(text):
#         norm_text = text.lower()
#         # Replace breaks with spaces
#         norm_text = norm_text.replace('<br />', ' ')
#         # Pad punctuation with spaces on both sides
#         for char in ['.', '"', ',', '(', ')', '!', '?', ';', ':']:
#             norm_text = norm_text.replace(char, ' ' + char + ' ')
#         return norm_text
#
#     if not os.path.isfile('aclImdb/alldata-id.txt'):
#         if not os.path.isdir(dirname):
#             if not os.path.isfile(filename):
#                 # Download IMDB archive
#                 url = 'http://ai.stanford.edu/~amaas/data/sentiment/' + filename
#                 r = requests.get(url)
#                 with open(filename, 'wb') as f:
#                     f.write(r.content)
#             tar = tarfile.open(filename, mode='r')
#             tar.extractall()
#             tar.close()
#
#         # Concat and normalize test/train data
#         folders = ['train/pos', 'train/neg', 'test/pos', 'test/neg', 'train/unsup']
#         alldata = u''
#         for fol in folders:
#             temp = u''
#             output = fol.replace('/', '-') + '.txt'
#             # Is there a better pattern to use?
#             txt_files = glob.glob('/'.join([dirname, fol, '*.txt']))
#
#             for txt in txt_files:
#                 with open(txt, 'r', encoding='utf-8') as t:
#                     control_chars = [chr(0x85)]
#                     t_clean = t.read()
#
#                     for c in control_chars:
#                         t_clean = t_clean.replace(c, ' ')
#                     temp += t_clean
#                 temp += "\n"
#             temp_norm = normalize_text(temp)
#             with open('/'.join([dirname, output]), 'w', encoding='utf-8') as n:
#                 n.write(temp_norm)
#             alldata += temp_norm
#         with open('/'.join([dirname, 'alldata-id.txt']), 'w', encoding='utf-8') as f:
#             for idx, line in enumerate(alldata.splitlines()):
#                 num_line = "_*{0} {1}\n".format(idx, line)
#                 f.write(num_line)
#     import os.path
#     assert os.path.isfile("aclImdb/alldata-id.txt"), "alldata-id.txt unavailable"


def doc_vect(alldocs):
    print 'Doc2Vec Each Tag is ID'
    train_docs = [doc for doc in alldocs if doc.split == 'train']
    test_docs = [doc for doc in alldocs if doc.split == 'test']
    print('%d docs: %d train-sentiment, %d test-sentiment' % (len(alldocs), len(train_docs), len(test_docs)))
    documents = []
    for doc in train_docs:
        sentence = TaggedDocument(doc.words, doc.tags)
        documents.append(sentence)
    print len(documents)
    cores = multiprocessing.cpu_count()
    simple_models = [
                # PV-DM w/concatenation - window=5 (both sides) approximates paper's 10-word total window size
                Doc2Vec(documents, dm=1, dm_concat=1, size=100, window=10, negative=5, hs=1, sample=1e-3, iter=20, min_count=1, workers=cores),
                # PV-DBOW
                Doc2Vec(documents, dm=0, size=100, window=10, negative=5, hs=1, sample=1e-3, iter=20, min_count=1, workers=cores),
                # PV-DM w/average
                Doc2Vec(documents, dm=1, dm_mean=1, size=100, window=10, negative=5, hs=1, sample=1e-3, iter=20, min_count=1, workers=cores),
                    ]

    models_by_name = OrderedDict((str(model), model) for model in simple_models)
    models_by_name['dbow+dmm'] = ConcatenatedDoc2Vec([simple_models[1], simple_models[2]])
    models_by_name['dbow+dmc'] = ConcatenatedDoc2Vec([simple_models[1], simple_models[0]])

    for name, model in models_by_name.items():
        print name
        train_targets, train_regressors = zip(*[(doc.sentiment, model.docvecs[doc.tags[0]]) for doc in train_docs])
        test_targets, test_regressors = zip(*[(doc.sentiment, model.infer_vector(doc.words)) for doc in test_docs])
        util.logit(train_regressors, train_targets, test_regressors, test_targets)


def label_doc_vect(alldocs):
    print 'Label_doc_Vec'
    train_docs = [doc for doc in alldocs if doc.split == 'train']
    test_docs = [doc for doc in alldocs if doc.split == 'test']
    print('%d docs: %d train-sentiment, %d test-sentiment' % (len(alldocs), len(train_docs), len(test_docs)))
    documents = []
    for doc in train_docs:
        if doc.split == 'train':
            sentence = TaggedDocument(doc.words, doc.tags+['l'+str(doc.sentiment)])
        else:
            sentence = TaggedDocument(doc.words, doc.tags)
        documents.append(sentence)
    print len(documents)
    cores = multiprocessing.cpu_count()
    simple_models = [
                # PV-DM w/concatenation - window=5 (both sides) approximates paper's 10-word total window size
                Doc2Vec(documents, dm=1, dm_concat=1, size=100, window=10, negative=5, hs=1, sample=1e-3, iter=20, min_count=1, workers=cores),
                # PV-DBOW
                Doc2Vec(documents, dm=0, size=100, window=10, negative=5, hs=1, sample=1e-3, iter=20, min_count=1, workers=cores),
                # PV-DM w/average
                Doc2Vec(documents, dm=1, dm_mean=1, size=100, window=10, negative=5, hs=1, sample=1e-3, iter=20, min_count=1, workers=cores),
                    ]

    models_by_name = OrderedDict((str(model), model) for model in simple_models)
    models_by_name['dbow+dmm'] = ConcatenatedDoc2Vec([simple_models[1], simple_models[2]])
    models_by_name['dbow+dmc'] = ConcatenatedDoc2Vec([simple_models[1], simple_models[0]])

    for name, model in models_by_name.items():
        print name
        train_targets, train_regressors = zip(*[(doc.sentiment, model.docvecs[doc.tags[0]]) for doc in train_docs])
        test_targets, test_regressors = zip(*[(doc.sentiment, model.infer_vector(doc.words)) for doc in test_docs])
        util.logit(train_regressors, train_targets, test_regressors, test_targets)


def pre_class(train_docs, test_docs, non_docs):
    train_y, train_X = zip(*[(doc.sentiment, ' '.join(doc.words)) for doc in train_docs])
    test_y, test_X = zip(*[(doc.sentiment, ' '.join(doc.words)) for doc in test_docs+non_docs])
    y_lin = util.pre_classify_text(train_X, train_y, test_X, None)
    return y_lin


def label_vect(alldocs):
    print 'Label2Vec with Pre-classification'
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
                Doc2Vec(documents, dm=1, dm_concat=1, size=100, window=10, negative=5, hs=1, sample=1e-3, iter=20, min_count=1, workers=cores),
                # PV-DBOW
                Doc2Vec(documents, dm=0, size=100, window=10, negative=5, hs=1, sample=1e-3, iter=20, min_count=1, workers=cores),
                # PV-DM w/average
                Doc2Vec(documents, dm=1, dm_mean=1, size=100, window=10, negative=5, hs=1, sample=1e-3, iter=20, min_count=1, workers=cores),
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
    print 'Lable2Vec without Pre-classification'
    train_docs = [doc for doc in alldocs if doc.split == 'train']
    test_docs = [doc for doc in alldocs if doc.split == 'test']
    print('%d docs: %d train-sentiment, %d test-sentiment' % (len(alldocs), len(train_docs), len(test_docs)))
    documents = []
    for doc in train_docs:
        if doc.split == 'train':
            sentence = TaggedDocument(doc.words, [str(doc.sentiment)])
        else:
            sentence = TaggedDocument(doc.words, [])
        documents.append(sentence)
    print len(documents)
    cores = multiprocessing.cpu_count()
    simple_models = [
                # PV-DM w/concatenation - window=5 (both sides) approximates paper's 10-word total window size
                Doc2Vec(documents, dm=1, dm_concat=1, size=100, window=10, negative=5, hs=1, sample=1e-3, iter=20, min_count=1, workers=cores),
                # PV-DBOW
                Doc2Vec(documents, dm=0, size=100, window=10, negative=5, hs=1, sample=1e-3, iter=20, min_count=1, workers=cores),
                # PV-DM w/average
                Doc2Vec(documents, dm=1, dm_mean=1, size=100, window=10, negative=5, hs=1, sample=1e-3, iter=20, min_count=1, workers=cores),
                    ]

    models_by_name = OrderedDict((str(model), model) for model in simple_models)
    models_by_name['dbow+dmm'] = ConcatenatedDoc2Vec([simple_models[1], simple_models[2]])
    models_by_name['dbow+dmc'] = ConcatenatedDoc2Vec([simple_models[1], simple_models[0]])

    for name, model in models_by_name.items():
        print name
        train_targets, train_regressors = zip(*[(doc.sentiment, model.infer_vector(doc.words)) for doc in train_docs])
        test_targets, test_regressors = zip(*[(doc.sentiment, model.infer_vector(doc.words)) for doc in test_docs])
        util.logit(train_regressors, train_targets, test_regressors, test_targets)


def get_imdb_data():
    SentimentDocument = namedtuple('SentimentDocument', 'words tags split sentiment')
    alldocs = []  # will hold all docs in original order
    with open('aclImdb/alldata-id.txt', 'r') as alldata:
        for line_no, line in enumerate(alldata):
            tokens = line.split()
            words = tokens[1:]
            tags = [line_no]  # `tags = [tokens[0]]` would also work at extra memory cost
            split = ['train', 'test', 'extra', 'extra'][line_no // 25000]  # 25k train, 25k test, 25k extra
            sentiment = [1.0, 0.0, 1.0, 0.0, None, None, None, None][
                line_no // 12500]  # [12.5K pos, 12.5K neg]*2 then unknown
            alldocs.append(SentimentDocument(words, tags, split, sentiment))
    return alldocs

def get_ng_data():
    from sklearn.datasets import fetch_20newsgroups
    remove = ('headers', 'footers', 'quotes')
    data_train = fetch_20newsgroups(subset='train', shuffle=True, random_state=42,
                                remove=remove)
    data_test = fetch_20newsgroups(subset='test', shuffle=True, random_state=42,
                               remove=remove)
    y_train, y_test = data_train.target, data_test.target
    SentimentDocument = namedtuple('SentimentDocument', 'words tags split sentiment')
    alldocs = []
    for line_no, line in enumerate(data_train.data):
            words = normalize_text(line).split()
            tags = [line_no]  # `tags = [tokens[0]]` would also work at extra memory cost
            split = 'train'
            sentiment = y_train[line_no]  # [12.5K pos, 12.5K neg]*2 then unknown
            alldocs.append(SentimentDocument(words, tags, split, sentiment))
    train_len = len(data_train.data)
    for line_no, line in enumerate(data_test.data):
            words = normalize_text(line).split()
            tags = [line_no+train_len]  # `tags = [tokens[0]]` would also work at extra memory cost
            split = 'test'
            sentiment = y_test[line_no]  # [12.5K pos, 12.5K neg]*2 then unknown
            alldocs.append(SentimentDocument(words, tags, split, sentiment))
    return alldocs



if __name__ == '__main__':
    data = get_ng_data()
    doc_vect(data)
    # label_vect(data)
    label_vect_no_class(data)
    label_doc_vect(data)

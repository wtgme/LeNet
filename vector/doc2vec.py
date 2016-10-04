# -*- coding: utf-8 -*-
"""
Created on 4:52 PM, 5/2/16

@author: tw
"""

import pickle

def pre_process(texts):
    from collections import defaultdict
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1
    texts = [[token for token in text if frequency[token] > 1] for text in texts]
    return texts
    # dictionary = corpora.Dictionary(texts)
    # # dictionary.save('/tmp/deerwester.dict') # store the dictionary, for future reference
    # corpus = [dictionary.doc2bow(text) for text in texts]
    # # corpora.MmCorpus.serialize('/tmp/deerwester.mm', corpus) # store to disk, for later use
    # return corpus, dictionary


def doc_vect(filename):
    documents = []
    from gensim.models.doc2vec import TaggedDocument
    with open(filename, 'r') as fo:
        for line in fo.readlines():
            tokens = line.split('\t')
            sentence = TaggedDocument(tokens[1].split(), [tokens[0]])
            documents.append(sentence)
    print len(documents)
    from gensim.models.doc2vec import Doc2Vec
    model = Doc2Vec(documents, min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=8)
    model.save('data/doc2vec.d2v')
    # pickle.dump(model, open('data/doc2vec.pick', 'w'))


def varify():
    from gensim.models.doc2vec import Doc2Vec
    model = Doc2Vec.load('data/doc2vec.d2v')
    documents = pickle.load(open('data/fedcorpus.pick', 'r'))
    for i in xrange(3):
        inferred_docvec = model.infer_vector(documents[i].words)
        print documents[i].tags
        print('%s:\n %s' % (model, model.docvecs.most_similar([inferred_docvec], topn=3)))



if __name__ == '__main__':
    '''Doc2Vec traing'''
    doc_vect('data/fed.data')

    # for word in model.vocab:
    #     print word
    # print model.most_similar(positive=['ed', 'anorexic'], negative=['fitness', 'health'])
    '''Verify model'''
    # varify()

    # profile('fed', 'com')

    # s = '''Female//27yrs//BPD//EDNOS//Self Harm//SW: 145 LBS//CW: 100 LBS//UGW:85 LBS ~~I Will Not Die Fat~~Trigger Warning~~ Avi Is Me~~'''
    # print process(s)





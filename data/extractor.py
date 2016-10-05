# -*- coding: utf-8 -*-
"""
Created on 15:36, 04/10/16

@author: wt
"""
import re
from nltk.tokenize import RegexpTokenizer
from nltk import SnowballStemmer
from nltk.corpus import stopwords
import string
import pandas as pd

tknzr = RegexpTokenizer(r'\w+')
stemmer = SnowballStemmer("english")
cachedStopWords = stopwords.words("english")
printable = set(string.printable)


def process(text):
    text = text.encode('utf8')

    '''Remove non-English chars'''
    text = filter(lambda x: x in printable, text.lower())
    text = re.sub(r'\d+', '', text)
    tokens = tknzr.tokenize(text)
    words = []
    for token in tokens:
        if token in cachedStopWords:
            # print '========================='
            continue
        else:
            word = stemmer.stem(token)
            words.append(word)
    if len(words) >= 3:
        text = ' '.join(words)
        text += ' .'
        # print text
        return text
    else:
        return None


def extract_tile(filename):
    idtitle = {}
    with open(filename, 'r') as fo:
        for line in fo.readlines():
            tokens = line.split('\t')
            did = tokens[0]
            cite = tokens[2]
            m = re.search('<title> (.+?)</title>', cite)
            if m:
                title = m.group(1)
                proceesed = process(title)
                if proceesed:
                    idtitle[did] = process(title)
        for id in idtitle.keys():
            print id + '\t' + idtitle[id]


def extract_label(paperfile, classname):
    nameid = {}
    with open(paperfile, 'r') as fo:
        for line in fo.readlines():
            tokens = line.split('\t')
            did = tokens[0]
            name = tokens[1]
            nameid[name] = did
            # print name
    with open(classname, 'r') as fo:
        for line in fo.readlines():
            tokens = line.split('\t')
            fid = nameid.get(tokens[0], None)
            if fid:
                print fid + '\t' + tokens[1].strip()


def categorize(filename):
    # ids, labels = [], []
    with open(filename, 'r') as fo:
        for line in fo.readlines():
            tokens = line.strip().split('\t')
            print tokens[0] + '\t' + tokens[1].split('/')[1]
            # ids.append(tokens[0])
            # labels.append(tokens[1].split('/')[1])
            # labels.append(tokens[1])
    # df = pd.DataFrame({'ids': ids,
    #                     'labels': labels})
    # print df.describe()
    # print pd.value_counts(df['labels'])


if __name__ == '__main__':
    # extract_tile('cora/papers')
    # extract_label('cora/papers', 'cora/classifications')
    categorize('label.data')
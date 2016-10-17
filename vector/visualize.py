# -*- coding: utf-8 -*-
"""
Created on 9:33 PM, 10/14/16

@author: tw
"""

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# configuration
# currency = ["Schweiz","Franken","Deutschland","Euro","Grossbritannien","britische_Pfund","Japan","Yen","Russland","Rubel","USA","US-Dollar","Kroatien","Kuna"]
# capital  = ["Athen","Griechenland","Berlin","Deutschland","Ankara","Tuerkei","Bern","Schweiz","Hanoi","Vietnam","Lissabon","Portugal","Moskau","Russland","Stockholm","Schweden","Tokio","Japan","Washington","USA"]
# language = ["Deutschland","Deutsch","USA","Englisch","Frankreich","Franzoesisch","Griechenland","Griechisch","Norwegen","Norwegisch","Schweden","Schwedisch","Polen","Polnisch","Ungarn","Ungarisch"]
# matches = model.most_similar(positive=["Frau"], negative=[], topn=30)
# words = [match[0] for match in matches]

# function draw_words
# ... reduces dimensionality of vectors of given words either with PCA or with t-SNE and draws the words into a diagram
# @param word2vec model     to visualize vectors from
# @param list     words     list of word strings to visualize
# @param bool     pca       use PCA (True) or t-SNE (False) to reduce dimensionality
# @param bool     alternate use different color and label align for every second word
# @param bool     arrows    use arrows to connect related words (items that are next to each other in list)
# @param float    x1        x axis range (from)
# @param float    x2        x axis range (to)
# @param float    y1        y axis range (from)
# @param float    y2        y axis range (to)
# @param string   title     for diagram
def draw_words(vectors, words, pca=False, alternate=True, arrows=True, title=''):
    # get vectors for given words from model
    # vectors = [model[word] for word in words]
    # vectors = [model.infer_vector(word) for word in words]

    if pca:
        pca = PCA(n_components=2, whiten=True)
        vectors2d = pca.fit(vectors).transform(vectors)
    else:
        tsne = TSNE(n_components=2, random_state=0)
        vectors2d = tsne.fit_transform(vectors)


    first = True # color alternation to divide given groups
    for point, word in zip(vectors2d, words):
        # plot points
        plt.scatter(point[0], point[1], c='r' if first else 'g')
        # plot word annotations
        plt.annotate(
            word,
            xy = (point[0], point[1]),
            xytext = (-20, -20) if first else (20, -20),
            textcoords = 'offset points',
            ha = 'right' if first else 'left',
            va = 'bottom',
            size = "x-large",
            arrowprops=dict(arrowstyle="->")
        )
        first = not first if alternate else first

    # draw arrows
    if arrows:
        for i in xrange(0, len(words)-1, 2):
            a = vectors2d[i][0]
            b = vectors2d[i][1]
            c = vectors2d[i+1][0]
            d = vectors2d[i+1][1]
            plt.arrow(a, b, c-a, d-b,
                shape='full',
                lw=0.1,
                edgecolor='#bbbbbb',
                facecolor='#bbbbbb',
                length_includes_head=True,
                head_width=0.03,
                width=0.01
            )

    # draw diagram title
    if title:
        plt.title(title)

    plt.savefig(title+'.pdf')
    plt.clf()

'''# get trained model
model = gensim.models.Word2Vec.load_word2vec_format("model/SG-300-5-NS10-R50.model", binary=True)
# draw pca plots
draw_words(model, currency, True, True, True, -3, 3, -2, 2, r'$PCA\ Visualisierung:\ W\ddot{a}hrung$')
draw_words(model, capital, True, True, True, -3, 3, -2, 2.2, r'$PCA\ Visualisierung:\ Hauptstadt$')
draw_words(model, language, True, True, True, -3, 3, -2, 1.7, r'$PCA\ Visualisierung:\ Sprache$')'''
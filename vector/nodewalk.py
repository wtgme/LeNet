# -*- coding: utf-8 -*-
"""
Created on 6:50 PM, 9/26/16

@author: tw
"""
'''
Reference implementation of node2vec.

For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec
Knowledge Discovery and Data Mining (KDD), 2016
'''

import argparse
import networkx as nx
import node2vec
from gensim.models import Word2Vec


def parse_args():
    '''
	Parses the node2vec arguments.
	'''
    parser = argparse.ArgumentParser(description="Run node2vec.")

    parser.add_argument('--input', nargs='?', default='graph/karate.edgelist',
                        help='Input graph path')

    parser.add_argument('--output', nargs='?', default='out.emb',
                        help='Embeddings path')

    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--walk-length', type=int, default=80,
                        help='Length of walk per source. Default is 80.')

    parser.add_argument('--num-walks', type=int, default=10,
                        help='Number of walks per source. Default is 10.')

    parser.add_argument('--window-size', type=int, default=10,
                        help='Context size for optimization. Default is 10.')

    parser.add_argument('--iter', default=1, type=int,
                        help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter. Default is 1.')

    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=False)

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=False)

    return parser.parse_args()


def get_gaint_comp(DG):
    #If edges in both directions (u,v) and (v,u) exist in the graph,
    # attributes for the new undirected edge will be a combination of the
    # attributes of the directed edges.
    G = DG.to_undirected()
    print 'Getting Gaint Component.........................'
    print 'Network is connected:', (nx.is_connected(G))
    print 'The number of connected components:', (nx.number_connected_components(G))
    largest_cc = max(nx.connected_components(G), key=len)
    return DG.subgraph(largest_cc)


def read_graph(filename):
    DG = nx.DiGraph()
    with open(filename, 'r') as fo:
        for line in fo.readlines():
            tokens = line.strip().split('\t')
            DG.add_edge(tokens[0], tokens[1], weight=1)
    return DG


def main(args):
    '''
	Pipeline for representational learning for all nodes in a graph.
	'''
    # nx_G = read_graph()
    nx_G = read_graph()
    nx_G = get_gaint_comp(nx_G)
    G = node2vec.Graph(nx_G, args.directed, args.p, args.q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(args.num_walks, args.walk_length)
    return walks


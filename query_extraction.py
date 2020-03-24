#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io
import numpy as np
import argparse
from utils import *
from query_aux import *

parser = argparse.ArgumentParser(description='Extraction of queries from source, training and target language')
parser.add_argument("--src_emb_train", type=str, default='', help="Load source embeddings for training")
parser.add_argument("--tgt_emb_train", type=str, default='', help="Load target embeddings for validation")
parser.add_argument("--src_emb_test", type=str, default='', help="Load source embeddings for testing")
parser.add_argument("--tgt_emb_test", type=str, default='', help="Load target embeddings for testing")
parser.add_argument("--output_dir", type=str, default='', help="Output directory of lightsvm files")
parser.add_argument('--center', action='store_true', help='whether to center embeddings or not')
parser.add_argument("--dico_train", type=str, default='', help="Training dictionary for training language")
parser.add_argument("--dico_valid", type=str, default='', help="Validation dictionary for training language")
parser.add_argument("--dico_test", type=str, default='', help="Testing dictionary for final language")
parser.add_argument("--maxload", type=int, default=200000)

parser.add_argument("--query_relevance_type", type=str, default='', help="Type of query relevance: binary or continuous")
parser.add_argument("--query_size", type=int, default=10, help="Size of the query")
parser.add_argument("--add_csls_coord", type=bool, default=True, help="Whether to add to query coord CSLS distance")
parser.add_argument('--k_csls', type=int,default=10, help='Number of coord in query for CSLS distance (from 0 to k)')
parser.add_argument("--add_word_coord", type=bool, default=True, help="Whether to add to query coord word embedding")
parser.add_argument("--add_query_coord", type=bool, default=True, help="Whether to add to query coord query word embedding")


params = parser.parse_args()

###### MAIN ######

# There are two options: 

# The first one is to align english to spanish, to learn using the modified english embeddings 
# In the second time, we then align english to italian and use the learning for the 
# new english embeddings. In this scenario, english is the source language and we have 4 distincts embeddings to 
# consider

# The second setting is to align spanish to english, learn using spanish modified embeddings with fix english
# embeddings. Thus, there is only english target embeddings and 2 source embeddings


query_extractor = compute_binary_distance if params.query_relevance_type == 'binary' else compute_embedding_distance

print("Extraction of queries alignment on %s" % params.dico_train)

words_tgt_train, x_tgt_train = load_vectors(params.tgt_emb_train,
                                            maxload=params.maxload,
                                            center=params.center)
words_src_train, x_src_train = load_vectors(params.src_emb_train,
                                            maxload=params.maxload,
                                            center=params.center)

print("Loading and extracting train data")
src2tgt_train, lexicon_size = load_lexicon(params.dico_train, words_src_train, words_tgt_train)
query_extractor(x_src_train,
                x_tgt_train,
                params.output_dir+"train",
                src2tgt_train,
                add_csls_coord=params.add_csls_coord,
                k_csls=params.k_csls,
                add_word_coord=params.add_word_coord,
                add_query_coord=params.add_query_coord,
                query_size=params.query_size)
print("Training data extracted")

print("Loading and extracting validation data")
src2tgt_valid, lexicon_size = load_lexicon(params.dico_valid, words_src_train, words_tgt_train)
query_extractor(x_src_train,
                x_tgt_train,
                params.output_dir+"train",
                src2tgt_valid,
                add_csls_coord=params.add_csls_coord,
                k_csls=params.k_csls,
                add_word_coord=params.add_word_coord,
                add_query_coord=params.add_query_coord,
                query_size=params.query_size)
print("Validation data extracted")

print("Extraction of queries alignment on %s" % params.dico_test)

words_tgt_test, x_tgt_test = load_vectors(params.tgt_emb_test,
                                            maxload=params.maxload,
                                            center=params.center)
words_src_test, x_src_test = load_vectors(params.src_emb_test,
                                            maxload=params.maxload,
                                            center=params.center)
print("Loading and extracting testing data")
src2tgt_test, lexicon_size = load_lexicon(params.dico_test, words_src_test, words_tgt_test)
query_extractor(x_src_test,
                x_tgt_test,
                params.output_dir+"test",
                src2tgt_test,
                add_csls_coord=params.add_csls_coord,
                k_csls=params.k_csls,
                add_word_coord=params.add_word_coord,
                add_query_coord=params.add_query_coord,
                query_size=params.query_size)
print("Testing data extracted")
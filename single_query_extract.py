#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Modifications for Guinet et al. 

import io
import warnings
import numpy as np
import argparse
from utils import *
from query_aux import *

#Disable warnings for Meta-features
warnings.filterwarnings("ignore")

# to use bool for parsing
def str2bool(v):
    """Parse String to bool
    Args: 
        v: String or Bool
    Returns:
        bool
    Raises:
        ArgumentTypeError: If v is not a String nor a bool
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


parser = argparse.ArgumentParser(description="Extraction of queries simplified")
parser.add_argument(
    "--src_emb", type=str, default="", help="Load source embeddings for training"
)
parser.add_argument(
    "--tgt_emb", type=str, default="", help="Load target embeddings for validation"
)
parser.add_argument(
    "--filename", type=str, default="", help="Filename of lightsvm files extracted"
)
parser.add_argument(
    "--center", action="store_true", help="whether to center embeddings or not"
)
parser.add_argument(
    "--dico", type=str, default="", help="Dictionary for query extraction"
)
parser.add_argument("--maxload", type=int, default=200000)

parser.add_argument(
    "--query_relevance_type",
    type=str,
    default="",
    help="Type of query relevance: binary or continuous",
)
parser.add_argument("--query_size", type=int, default=10, help="Size of the query")
parser.add_argument(
    "--add_csls_coord",
    type=str2bool,
    default=True,
    help="Whether to add to query coord CSLS distance",
)
parser.add_argument(
    "--k_csls",
    type=int,
    default=10,
    help="Number of coord in query for CSLS distance (from 0 to k)",
)
parser.add_argument(
    "--testing_query",
    type=str2bool,
    default=False,
    help="Whether to impose the ground truth traduction presence in the query",
)
parser.add_argument(
    "--add_word_coord",
    type=str2bool,
    default=False,
    help="Whether to add to query coord word embedding",
)
parser.add_argument(
    "--discard_empty_query",
    type=str2bool,
    default=False,
    help="Whether to remove query without the right traduction or not",
)
parser.add_argument(
    "--use_csls",
    type=str2bool,
    default=False,
    help="Whether to use CSLS distance or CosineSim",
)
parser.add_argument(
    "--add_query_coord",
    type=str2bool,
    default=False,
    help="Whether to add to query coord query word embedding",
)

parser.add_argument(
    "--add_meta_features",
    type=str2bool,
    default=True,
    help="Whether to add to meta-features of the 2 clouds (source and target)",
)

parser.add_argument(
    "--center_meta_features",
    type=str2bool,
    default=True,
    help="Whether to add to center the meta-features of the target clouds",
)

parser.add_argument(
    "--nn_size_meta_features",
    type=int,
    default=10,
    help="Number of neighbors to use when computing meta-features",
)


params = parser.parse_args()

###### MAIN ######


query_extractor = (
    compute_binary_distance
    if params.query_relevance_type == "binary"
    else compute_embedding_distance
)

print("Extraction of queries alignment on %s" % params.dico)

words_tgt, x_tgt = load_vectors(
    params.tgt_emb, maxload=params.maxload, center=params.center
)
words_src, x_src = load_vectors(
    params.src_emb, maxload=params.maxload, center=params.center
)

print("Loading and extracting data")
src2tgt, lexicon_size = load_lexicon(params.dico, words_src, words_tgt)

query_extractor(
    x_src,
    x_tgt,
    params.filename,
    src2tgt,
    add_csls_coord=params.add_csls_coord,
    k_csls=params.k_csls,
    testing_query=params.testing_query,
    discard_empty_query=params.discard_empty_query,
    add_word_coord=params.add_word_coord,
    add_query_coord=params.add_query_coord,
    add_meta_features=params.add_meta_features,
    center_meta_features=params.center_meta_features,
    nn_size_meta_features=params.nn_size_meta_features,
    query_size=params.query_size,
    use_csls=params.use_csls
)
print("Query file extracted")

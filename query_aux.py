#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io
import munkres
import ot
import numpy as np
import pandas as pd
import scipy as sp
from scipy.spatial.distance import directed_hausdorff
from pymfe.mfe import MFE

def compute_NN_list(x_src, x_tgt, idx_src, bsz=100, nn_size=10):
    """
    Compute the list of the nearest neighboors in target space 
    for each word in source words
    """

    nn_list = dict()
    x_src /= np.linalg.norm(x_src, axis=1)[:, np.newaxis] + 1e-8
    x_tgt /= np.linalg.norm(x_tgt, axis=1)[:, np.newaxis] + 1e-8

    for i in range(0, len(idx_src), bsz):
        e = min(i + bsz, len(idx_src))
        scores = np.dot(x_tgt, x_src[idx_src[i:e]].T)
        ind = np.argpartition(scores, -nn_size, axis=0)[-nn_size:]
        #print(scores.shape) #200 000, 100
        #print(ind.shape) # 10,100
        for j in range(i, e):
            nn_list[idx_src[j]] = list(
                ind[:, j - i][np.argsort(scores[:, j - i][ind[:, j - i]])]
            )[::-1]
    return nn_list


def compute_csls_coord(x_src, x_tgt, lexicon, lexicon_size=-1, k=10, bsz=1024,return_similarity=False):
    """
    Compute similarity and CSLS penalty (from 0 to k) for all word vectors 
    """
    if lexicon_size < 0:
        lexicon_size = len(lexicon)
    idx_src = list(lexicon.keys())

    x_src /= np.linalg.norm(x_src, axis=1)[:, np.newaxis] + 1e-8
    x_tgt /= np.linalg.norm(x_tgt, axis=1)[:, np.newaxis] + 1e-8

    sr = x_src[list(idx_src)]

    similarities = 2 * np.dot(sr, x_tgt.T)

    # We compute x_tgt penaly for k steps of nn
    x_tgt_penalty = np.zeros((x_tgt.shape[0], k))
    for i in range(0, x_tgt.shape[0], bsz):
        j = min(i + bsz, x_tgt.shape[0])
        sc_batch = np.dot(x_tgt[i:j, :], x_src.T)
        relevant_nn_dotprod = np.partition(sc_batch, -k, axis=1)[:, -k:]
        relevant_nn_dotprod.sort()

        for l in range(0, k):
            x_tgt_penalty[i:j, l] = np.mean(relevant_nn_dotprod[:, l:], axis=1)

    # We compute x_src penaly for k steps of nn
    x_src_penalty = np.zeros((sr.shape[0], k))
    for i in range(0, sr.shape[0], bsz):
        j = min(i + bsz, sr.shape[0])
        sc_batch = np.dot(sr[i:j, :], x_tgt.T)
        relevant_nn_dotprod = np.partition(sc_batch, -k, axis=1)[:, -k:]
        relevant_nn_dotprod.sort()

        for l in range(0, k):
            x_src_penalty[i:j, l] = np.mean(relevant_nn_dotprod[:, l:], axis=1)

    if return_similarity:
        return similarities,x_src_penalty,x_tgt_penalty
    else:   
        return x_src_penalty, x_tgt_penalty


def compute_csls_list(x_src, x_tgt, lexicon,idx_src, nn_size = 10,lexicon_size=-1, k=10, bsz=1024):

    similarities,x_src_penalty,x_tgt_penalty = compute_csls_coord(x_src, x_tgt,
                                                                    lexicon,
                                                                    lexicon_size=lexicon_size,
                                                                    k=k,
                                                                    bsz=bsz,
                                                                    return_similarity=True)
    
    # x_src_penalty and x_tgt_penalty are of shape [number of words,k].
    # We need to keep only the first column (the penalty for each word) and add this penalties to the similarity
    x_src_penalty = x_src_penalty[:,0]
    x_tgt_penalty = x_tgt_penalty[:,0]
    # We then remove these to the similarities :
    similarities -= x_src_penalty[:,np.newaxis]
    similarities -= x_tgt_penalty[np.newaxis, :]
    # We now have a matrix with similarities[i,j] = csls(src[i],tgt[j])
    # We need to compute the list of the nearest neighboors (in terms of csls) in target space 
    #for each word in source words (meaning the biggest similarity)
    csls_list = dict()   
    for i in range(len(idx_src)):
        if (i % 1000 == 0):
            print("Step {}".format(i))
        # For the i-th source word, the csls similarity with each word from target :
        scores = similarities[i,:]
        ind = np.argpartition(scores, -nn_size, axis=0)[-nn_size:]
        #print(ind.shape) #10,
        #print(scores.shape) #200 000,
        csls_list[idx_src[i]] = list(ind[np.argsort(scores[ind])])[::-1]

    return csls_list                                                                                                                                                                                                                                                                                                                                   


def compute_binary_distance(
    x_src,
    x_tgt,
    file_name,
    lexicon,
    k_csls=10,
    discard_empty_query=False,
    testing_query=False,
    add_csls_coord=True,
    add_word_coord=False,
    add_query_coord=False,
    add_meta_features=True,
    center_meta_features=False,
    nn_size_meta_features=10,
    min_relevance=1,
    max_relevance=2,
    bsz=100,
    query_size=2,
    use_csls = False
):
    """
    Use a 0-1 loss for queries construction and save svmlight file
    (or 1-2 as 0 is considered as not relevant and therefore, as they are many 0, the prediction
    is nearly always 0)
    To be modfied for exact query exctraction
    """

    idx_src = list(lexicon.keys())
    print("Expected query size: %d" % (query_size * len(idx_src)))
    
    #Compute NN list ou CSLS-NN list
    nn_list = compute_NN_list(
        x_src, x_tgt, idx_src=idx_src, bsz=bsz, nn_size=query_size
    ) if use_csls == False else compute_csls_list(x_src, x_tgt, lexicon=lexicon, idx_src=idx_src)
    
    if add_meta_features == True:
        
        target_nn_list=compute_target_NN_list(x_tgt,nn_size=nn_size_meta_features)
        source_nn_list=compute_source_NN_list(x_src,idx_src,nn_size=nn_size_meta_features)
    
    query_id = 0
    file = open(file_name, "wb")
    # wb means we are writting in binary
    file.truncate(0)

    if add_csls_coord == True:

        x_src_penalty, x_tgt_penalty = compute_csls_coord(
            x_src, x_tgt, lexicon, lexicon_size=-1, k=k_csls, bsz=1024
        )
    query_count = len(idx_src)
    for ind_src, i in enumerate(idx_src):

        # Whether or not to add ground truth words in each query.
        # For the testing scenario, we do not want to force their 
        # presence whereas for the learning scenario, they are 
        # necessary
        if discard_empty_query == True:
            # Do not write queries with no relevent labels with (at the moment) the NN method 
            test = False 
            for e in lexicon[i]:
                if e in nn_list[i]:
                    test = True
            if test:
                query_list = nn_list[i]
                discard_curr_query = False
            else:
                discard_curr_query = True
                query_count -=1

        elif testing_query == True:
            
            query_list = nn_list[i]
            discard_curr_query = False
        else:
            
            # We consider ground truth words and fill the remaining with neighboors
            # in order to have a fixed length of query size
            target = list(lexicon[i])
            others_neigh = [elem for elem in nn_list[i] if elem not in lexicon[i]]
            query_list = target + others_neigh[: query_size - len(lexicon[i])]
            discard_curr_query = False
        
        # We compute the source word "voisinage"
        if add_meta_features == True:
                
                source_pt_neigh=source_nn_list[i]
                source_pt_emb=x_src[source_pt_neigh]
        
        if not discard_curr_query:
            
            for ind, j in enumerate(query_list):
                
                #All the bellow can be computed in parallel, to implement ?
                
                total_coord = []

                if add_csls_coord == True:

                    similarity = np.dot(x_src[i], x_tgt[j].T)
                    # We append NN similarity
                    total_coord.append(similarity)
                    # We append k CSLS similarity
                    total_coord = np.concatenate(
                        (
                            total_coord,
                            similarity - x_src_penalty[ind_src, :] - x_tgt_penalty[j, :],
                        ),
                        axis=None,
                    )

                if add_word_coord == True:

                    # We add the coord of the potential translation word
                    total_coord = np.concatenate((total_coord, x_tgt[j]), axis=None)

                if add_query_coord == True:

                    # We add the coord of the query word
                    total_coord = np.concatenate((total_coord, x_src[i]), axis=None)
                    
                if add_meta_features == True:
                    
                    target_pt_neigh=target_nn_list[j]
                    target_pt_emb=x_tgt[target_pt_neigh]
                    
                    if center_meta_features==True:
                            
                        #We align source and target word 
                        target_pt_emb-=(source_pt_emb[0]-target_pt_emb[0])
                    
                    #We compute mf for the two groups and for each single group
                    bigroup_mf=bigroup_meta_features(source_pt_emb,target_pt_emb)
                    target_mf=single_group_meta_features(target_pt_emb)
                    source_mf=single_group_meta_features(source_pt_emb)
                    
                    total_coord = np.concatenate((total_coord, bigroup_mf,
                                                  target_mf, source_mf), axis=None)


                # For the given traduction, the relevance is max_relevance
                if j in lexicon[i]:

                    line = svm_line(total_coord, query_id, max_relevance)
                    file.write(line)

                # For the other nearest neighboors, the relevance is min_relevance
                else:
                    
                    line = svm_line(total_coord, query_id, min_relevance)
                    file.write(line)

        query_id += 1
        
        #To keep track of progress
        if (query_id%100)==0:
            
            print("Query number {} done out of {}".format(query_id,query_size * len(idx_src)))
            
    print("Query kept {} out of {}".format(query_count,len(idx_src)))
    file.close()


def svm_line(coord_list, query_id, relevance):

    pairs = ["%d:%.16g" % (i, x) for i, x in enumerate(coord_list)]

    sep_line = []

    sep_line.append(str(relevance))

    sep_line.append("qid:%d" % query_id)
    sep_line.extend(pairs)
    sep_line.append("\n")

    line = " ".join(sep_line)

    return line.encode("ascii")

def bigroup_meta_features(source_pt_emb,target_pt_emb):
    
    y = [0]*source_pt_emb.shape[0] + [1]*source_pt_emb.shape[0]
    X=np.concatenate([source_pt_emb,target_pt_emb],axis=0)
    
    # Extract several meta-features (more than for single group)
    mfe=MFE(groups=["Statistical","complexity","concept","clustering"],suppress_warnings=True)
    mfe.fit(X, y)
    ft = mfe.extract()
    
    feat_list =[]
    
    interest_features = ['ch', 'cohesiveness.mean', 'cohesiveness.sd', 'conceptvar.mean',
       'conceptvar.sd', 'cor.mean', 'cor.sd', 'cov.mean', 'cov.sd',
       'eigenvalues.mean', 'eigenvalues.sd', 'f3.mean', 'f4.mean', 'gravity',
       'impconceptvar.mean', 'impconceptvar.sd', 'int', 'iq_range.mean',
       'iq_range.sd', 'kurtosis.mean', 'kurtosis.sd', 'mad.mean', 'mad.sd',
       'max.mean', 'max.sd', 'mean.mean', 'mean.sd', 'median.mean',
       'median.sd', 'min.mean', 'min.sd', 'nr_cor_attr', 'nr_norm',
       'nr_outliers', 'pb', 'range.mean', 'range.sd', 'sd.mean', 'sd.sd',
       'sil', 'skewness.mean', 'skewness.sd', 't4', 't_mean.mean', 't_mean.sd',
       'var.mean', 'var.sd', 'vdb', 'vdu', 'wg_dist.mean', 'wg_dist.sd']
    
    for feat,val in zip(ft[0],ft[1]):
        
        if feat in interest_features:
            
            feat_list.append(val)
    
    #We add 3 extra "distances"
    
    hung_dist=permutation_dist(source_pt_emb,target_pt_emb)
    wass_dist=wasserstein_dist(source_pt_emb,target_pt_emb)
    hauss_dist=hausdorff_dist(source_pt_emb,target_pt_emb)
    
    feat_list.append(hung_dist)
    feat_list.append(wass_dist)
    feat_list.append(hauss_dist)
    
    #return pd.Series(feat_list,index=interest_features+["hung_dist","wasser","hauss"])
    return feat_list


def single_group_meta_features(X):
    
    # Extract single group (source/target) features
    
    features = ["cohesiveness","cor", "cov", "eigenvalues", "nr_cor_attr", 
                "min", "mean", "median", "max", "iq_range", "kurtosis",
                "skewness", "t_mean", "var","sd", "range",
                "nr_norm", "nr_outliers"]
    mfe=MFE(features=features,suppress_warnings=True)
    
    mfe.fit(X, [0]*X.shape[0])
    ft = mfe.extract()

    #return pd.Series(ft[1],index=ft[0])
    return ft[1]

def hausdorff_dist(X1,X2):
    
    return directed_hausdorff(X1,X2)[0]

def permutation_dist(X1, X2):

    cost = (((X1[np.newaxis,:] - X2[:,np.newaxis,:])**2).sum(2))
    matrix = cost.tolist()
    m = munkres.Munkres()
    newind = m.compute(matrix)    
    costnew = 0.
    for (iold, inew) in newind:
        costnew += cost[iold, inew]

    return np.sqrt(costnew)

def wasserstein_dist(X1,X2):
    
    C1 = sp.spatial.distance.cdist(X1, X1)
    C2 = sp.spatial.distance.cdist(X2, X2)

    C1 /= C1.max()
    C2 /= C2.max()

    p = ot.unif(X1.shape[0])
    q = ot.unif(X2.shape[0])

    gw0, log0 = ot.gromov.gromov_wasserstein(
        C1, C2, p, q, 'square_loss', verbose=False, log=True)
    
    return log0['gw_dist']


def compute_target_NN_list(x_tgt, bsz=100, nn_size=10):
    """
    Compute the list of the nearest neighboors in target space 
    for each word in target words
    Include the index of the origin word
    """

    nn_list = dict()
    x_tgt /= np.linalg.norm(x_tgt, axis=1)[:, np.newaxis] + 1e-8

    for i in range(0, len(x_tgt), bsz):
        e = min(i + bsz, len(x_tgt))
        scores = np.dot(x_tgt, x_tgt[i:e].T)
        ind = np.argpartition(scores, -(nn_size+1), axis=0)[-(nn_size+1):]
        for j in range(i, e):
            nn_list[j] = list(
                ind[:, j - i][np.argsort(scores[:, j - i][ind[:, j - i]])]
            )[::-1]
    return nn_list

def compute_source_NN_list(x_src, idx_src, bsz=100, nn_size=10):
    """
    Compute the list of the nearest neighboors in source space 
    for each word in source words lexicon list
    Include the index of the origin word
    """

    nn_list = dict()
    x_src /= np.linalg.norm(x_src, axis=1)[:, np.newaxis] + 1e-8

    for i in range(0, len(idx_src), bsz):
        e = min(i + bsz, len(idx_src))
        scores = np.dot(x_src,x_src[idx_src[i:e]].T)
        ind = np.argpartition(scores, -(nn_size+1), axis=0)[-(nn_size+1):]
        for j in range(i, e):
            nn_list[idx_src[j]] = list(
                ind[:, j - i][np.argsort(scores[:, j - i][ind[:, j - i]])]
            )[::-1]
    return nn_list

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io
import numpy as np
import pandas as pd
from sklearn.datasets import dump_svmlight_file

def compute_NN_list(x_src, x_tgt,idx_src,bsz=100,nn_size=10):
    '''
    Compute the list of the nearest neighboors in target space 
    for each word in source words
    '''
    
    nn_list=dict()
    x_src /= np.linalg.norm(x_src, axis=1)[:, np.newaxis] + 1e-8
    x_tgt /= np.linalg.norm(x_tgt, axis=1)[:, np.newaxis] + 1e-8
    
    for i in range(0, len(idx_src), bsz):
        e = min(i + bsz, len(idx_src))
        scores = np.dot(x_tgt, x_src[idx_src[i:e]].T)
        ind=np.argpartition(scores, -nn_size,axis=0)[-nn_size:]
        for j in range(i, e):
            nn_list[idx_src[j]] = list(ind[:,j - i][np.argsort(scores[:,j - i][ind[:,j - i]])])[::-1]
    return nn_list

def compute_csls_coord(x_src, x_tgt, lexicon, lexicon_size=-1, k=10, bsz=1024):
    '''
    Compute similarity and CSLS penalty (from 0 to k) for all word vectors 
    '''
    if lexicon_size < 0:
        lexicon_size = len(lexicon)
    idx_src = list(lexicon.keys())

    x_src /= np.linalg.norm(x_src, axis=1)[:, np.newaxis] + 1e-8
    x_tgt /= np.linalg.norm(x_tgt, axis=1)[:, np.newaxis] + 1e-8

    sr = x_src[list(idx_src)]
    
    #similarities = 2 * np.dot(sr, x_tgt.T)

    #We compute x_tgt penaly for k steps of nn
    x_tgt_penalty = np.zeros((x_tgt.shape[0],k))
    for i in range(0, x_tgt.shape[0], bsz):
        j = min(i + bsz, x_tgt.shape[0])
        sc_batch = np.dot(x_tgt[i:j, :], x_src.T)
        relevant_nn_dotprod = np.partition(sc_batch, -k, axis=1)[:, -k:]
        relevant_nn_dotprod.sort()
        
        for l in range(0,k):
            x_tgt_penalty[i:j,l] = np.mean(relevant_nn_dotprod[:,l:],axis=1)
    
    #We compute x_src penaly for k steps of nn 
    x_src_penalty = np.zeros((sr.shape[0],k))
    for i in range(0, sr.shape[0], bsz):
        j = min(i + bsz, sr.shape[0])
        sc_batch = np.dot(sr[i:j, :], x_tgt.T)
        relevant_nn_dotprod = np.partition(sc_batch, -k, axis=1)[:, -k:]
        relevant_nn_dotprod.sort()
        
        for l in range(0,k):
            x_src_penalty[i:j,l] = np.mean(relevant_nn_dotprod[:,l:],axis=1)
            
    #similarities -= x_src_penalty[:,np.newaxis]
    #similarities -= x_tgt_penalty[np.newaxis, :]

    #return similarities,x_src_penalty,x_tgt_penalty
    return x_src_penalty,x_tgt_penalty


    
def compute_embedding_distance(x_src,
                               x_tgt,
                               file_name,
                               lexicon,
                               k_csls=10,
                               add_csls_coord=True,
                               add_word_coord=True,
                               add_query_coord=False,
                               bsz=100,
                               query_size=10):
    '''
    Use a embedding loss for queries construction and save svmlight file.
    More precisely, we use as score the maximum similarity with a given 
    traduction of the considered word in terms of embeddings
    It correspond to continuous relevance in the paper
    '''
    
    idx_src = list(lexicon.keys())
    print("Expected query size: %d" % (query_size*len(idx_src)))
    nn_list = compute_NN_list(x_src,x_tgt,idx_src=idx_src,bsz=bsz, nn_size=query_size)
    
    query_id=0
    file = open(file_name,'wb')
    #wb means we are writting in binary
    file.truncate(0)
    
    if add_csls_coord ==True:
        
        x_src_penalty,x_tgt_penalty = compute_csls_coord(x_src,
                                                         x_tgt,
                                                         lexicon,
                                                         lexicon_size=-1,
                                                         k=k_csls,
                                                         bsz=1024)
    
    for ind_src,i in enumerate(idx_src):
        
        # We consider ground truth words and fill the remaining with neighboors
        #in order to have a fixed length of query size
        target = list(lexicon[i])
        others_neigh = [elem for elem in nn_list[i] if elem not in lexicon[i]]
        query_list = target + others_neigh[:query_size-len(lexicon[i])]

        score=np.dot(x_tgt[query_list], x_tgt[target].T).max(axis=1) 
        #score=(score/score.min()).round().astype('int')
        sorted_score = sorted(score)
        
        
        for ind,j in enumerate(query_list):
            
            total_coord = []
            
            if add_csls_coord == True:
                
                similarity = np.dot(x_src[i],x_tgt[j].T)
                #We append NN similarity
                total_coord.append(similarity)
                #We append k CSLS similarity
                total_coord = np.concatenate((total_coord,
                                              similarity-x_src_penalty[ind_src,:]-x_tgt_penalty[j,:]),
                                           axis=None)

            if add_word_coord == True:
            
                # We add the coord of the query word
                total_coord = np.concatenate((total_coord,
                                              x_tgt[j]),
                                           axis=None)
                
                
            if add_query_coord == True:
            
                # We add the coord of the query word
                total_coord = np.concatenate((total_coord,
                                              x_src[i]),
                                           axis=None)
                
            
            #The relevance is the ranking of the embedding distance to ground truth
            if score[ind] >= 0.9:
                
                relevance = query_size 
                
            else:
                
                relevance = sorted_score.index(score[ind]) + 1

            line = svm_line(total_coord,query_id,relevance)
            file.write(line)

        query_id+=1
        
    file.close()
    
    
def compute_binary_distance(x_src,
                            x_tgt,
                            file_name,
                            lexicon,
                            k_csls=10,
                            add_csls_coord=True,
                            add_word_coord=True,
                            add_query_coord=False,
                            min_relevance=1,
                            max_relevance=2,
                            bsz=100,
                            query_size=10):
    '''
    Use a 0-1 loss for queries construction and save svmlight file
    (or 1-2 as 0 is considered as not relevant and therefore, as they are many 0, the prediction
    is nearly always 0)
    To be modfied for exact query exctraction
    '''
    
    idx_src = list(lexicon.keys())
    print("Expected query size: %d" % (query_size*len(idx_src)))
    nn_list = compute_NN_list(x_src,x_tgt,idx_src=idx_src,bsz=bsz, nn_size=query_size)
    
    query_id=0
    file = open(file_name,'wb')
    #wb means we are writting in binary
    file.truncate(0)
    
    if add_csls_coord ==True:
        
        x_src_penalty,x_tgt_penalty = compute_csls_coord(x_src,
                                                         x_tgt,
                                                         lexicon,
                                                         lexicon_size=-1,
                                                         k=k_csls,
                                                         bsz=1024)
    
    for ind_src,i in enumerate(idx_src):
        
        # We consider ground truth words and fill the remaining with neighboors
        # in order to have a fixed length of query size
        # To change, this is not totally correct
        target = list(lexicon[i])
        others_neigh = [elem for elem in nn_list[i] if elem not in lexicon[i]]
        query_list = target + others_neigh[:query_size-len(lexicon[i])]

    
        for ind,j in enumerate(query_list):
        
            total_coord = []
            
            if add_csls_coord == True:
                
                similarity = np.dot(x_src[i],x_tgt[j].T)
                #We append NN similarity
                total_coord.append(similarity)
                #We append k CSLS similarity
                total_coord = np.concatenate((total_coord,
                                              similarity-x_src_penalty[ind_src,:]-x_tgt_penalty[j,:]),
                                           axis=None)

            if add_word_coord == True:
            
                # We add the coord of the query word
                total_coord = np.concatenate((total_coord,
                                              x_tgt[j]),
                                           axis=None)
                
                
            if add_query_coord == True:
            
                # We add the coord of the query word
                total_coord = np.concatenate((total_coord,
                                              x_src[i]),
                                           axis=None)

        
            #For the given traduction, the relevance is max_relevance
            if j in lexicon[i]:

                line = svm_line(total_coord,query_id,max_relevance)
                file.write(line)
                
            
        
            #For the other nearest neighboors, the relevance is min_relevance
            else: 
                line = svm_line(total_coord,query_id,min_relevance)
                file.write(line)

        query_id+=1
        
    file.close()
        
        
def svm_line(coord_list,query_id,relevance):
    
    pairs = ['%d:%.16g'%(i,x) for i,x in enumerate(coord_list)]

    sep_line = []
    
    sep_line.append(str(relevance))
        
    sep_line.append('qid:%d'%query_id)
    sep_line.extend(pairs)
    sep_line.append('\n')

    line = ' '.join(sep_line)
    
    return line.encode('ascii')
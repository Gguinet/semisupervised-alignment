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


def compute_binary_distance(x_src,
                            x_tgt,
                            file_name,
                            lexicon,
                            add_query=False,
                            bsz=100,
                            query_size=10):
    '''
    Use a 0-1 loss for queries construction and save svmlight file
    '''
    
    idx_src = list(lexicon.keys())
    print("Expected query size: %d" % (query_size*len(idx_src)))
    nn_list = compute_NN_list(x_src,x_tgt,idx_src=idx_src,bsz=bsz, nn_size=query_size)
    
    query_id=0
    file = open(file_name,'wb')
    # wb means we are writting in binary
    file.truncate(0)
    
    for i in idx_src:

        
        if add_query==True:
            query_coord=x_src[i]
        else:
            query_coord=[]
        
        #For the given traduction, the relevance is 2
        for j in lexicon[i]:
            
            line = svm_line(np.concatenate((x_tgt[j],query_coord),
                                           axis=None),query_id,2)
            file.write(line)
        
        #For the other nearest neighboors, the relevance is 1
        #We choose a fixed lenght of query size
        others_neigh = [elem for elem in nn_list[i] if elem not in lexicon[i]]

        for j in others_neigh[:query_size-len(lexicon[i])]:

            line = svm_line(np.concatenate((x_tgt[j],query_coord),
                                       axis=None),query_id,1)
            file.write(line)
            
        query_id+=1
        
    file.close()
    
def compute_embedding_distance(x_src,
                               x_tgt,
                               file_name,
                               lexicon,
                               add_query=False,
                               bsz=100,
                               query_size=10):
    '''
    Use a embedding loss for queries construction and save svmlight file.
    More precisely, we use as score the maximum similarity with a given 
    traduction of the considered word in terms of embeddings
    '''
    
    idx_src = list(lexicon.keys())
    print("Expected query size: %d" % (query_size*len(idx_src)))
    nn_list = compute_NN_list(x_src,x_tgt,idx_src=idx_src,bsz=bsz, nn_size=query_size)
    
    query_id=0
    file = open(file_name,'wb')
    # wb means we are writting in binary
    file.truncate(0)
    
    
    for i in idx_src:
        
        # We consider ground truth words and fill the remaining with neighboors
        target = list(lexicon[i])
        others_neigh = [elem for elem in nn_list[i] if elem not in lexicon[i]]
        query_list = target + others_neigh[:query_size-len(lexicon[i])]

        score=np.dot(x_tgt[query_list], x_tgt[target].T).max(axis=1) 
        #score=(score/score.min()).round().astype('int')
        sorted_score = sorted(score)

        
        if add_query==True:
            query_coord=x_src[i]
        else:
            query_coord=[]

        for k,j in enumerate(query_list):
            
            #The relevance is the ranking of the embedding distance to ground truth
            relevance = sorted_score.index(score[k])

            line = svm_line(np.concatenate((x_tgt[j],query_coord),
                                           axis=None),query_id,relevance)
            file.write(line)

        query_id+=1
        
    file.close()
    
    
def compute_binary_distance_df(x_src,
                               x_tgt,
                               file_name,
                               lexicon,
                               add_query=False,
                               bsz=100,
                               query_size=10):
    '''
    Use a 0-1 loss for queries construction and save svmlight file using dataframe
    Give faster loading of data in the pipeline
    '''
    idx_src = list(lexicon.keys())
    print("Expected query size: %d" % (query_size*len(idx_src)))
    nn_list = compute_NN_list(x_src,x_tgt,idx_src=idx_src,bsz=bsz, nn_size=query_size)
    
    if add_query==True:
        feature_size=600
    else:
        feature_size=300
    
    final_df = pd.DataFrame(columns=range(feature_size))
    query_list,relevance_list=[],[]
    query_id,ind=0,0
   
    
    for i in idx_src:
        
        if add_query==True:
            query_coord=x_src[i]
        else:
            query_coord=[]
        
        for j in lexicon[i]:
            
            final_df.loc[ind]=np.concatenate((x_tgt[j],query_coord),axis=None)
            relevance_list.append(1)
            query_list.append(query_id)
            ind+=1

        for j in nn_list[i][:query_size-len(lexicon[i])]:
            
            if not j in lexicon[i]:
                
                final_df.loc[ind]=np.concatenate((x_tgt[j],query_coord),axis=None)
                relevance_list.append(0)
                query_list.append(query_id)
                ind+=1

        query_id+=1
    
    dump_svmlight_file(final_df,relevance_list,file_name,query_id=query_list)
        
        
def svm_line(coord_list,query_id,relevance=''):
    
    pairs = ['%d:%.16g'%(i,x) for i,x in enumerate(coord_list)]

    sep_line = []
    
    if relevance !='':
        sep_line.append(str(relevance))
        
    sep_line.append('qid:%d'%query_id)
    sep_line.extend(pairs)
    sep_line.append('\n')

    line = ' '.join(sep_line)
    
    return line.encode('ascii')
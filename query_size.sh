#!/bin/usr/env sh
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

set -e
s1=${1:-en}
t1=${2:-it}

s2=${1:-en}
t2=${2:-es}

echo "Example based on the ${s2}->${t2} alignment, using knowledge of ${s1}->${t1} "

if [ ! -d data/ ]; then
  mkdir -p data;
fi

if [ ! -d alignement/ ]; then
  mkdir -p alignement;
fi

if [ ! -d alignement/round1/ ]; then
  mkdir -p alignement/round1;
fi

if [ ! -d alignement/round2/ ]; then
  mkdir -p alignement/round2;
fi

if [ ! -d res/ ]; then
  mkdir -p res;
fi

if [ ! -d res/query_size/ ]; then
  mkdir -p res/query_size;
fi


if [ ! -d query_8/ ]; then
  mkdir -p query_8;
fi

if [ ! -d query_10/ ]; then
  mkdir -p query_10;
fi

if [ ! -d query_12/ ]; then
  mkdir -p query_12;
fi

if [ ! -d query_14/ ]; then
  mkdir -p query_14;
fi

if [ ! -d query_16/ ]; then
  mkdir -p query_16;
fi

if [ ! -d query_18/ ]; then
  mkdir -p query_18;
fi

if [ ! -d query_20/ ]; then
  mkdir -p query_20;
fi

if [ ! -d query_25/ ]; then
  mkdir -p query_25;
fi

if [ ! -d query_30/ ]; then
  mkdir -p query_30;
fi

if [ ! -d query_40/ ]; then
  mkdir -p query_40;
fi



dico_train=data/${s1}-${t1}.0-5000.txt
if [ ! -f "${dico_train}" ]; then
  DICO=$(basename -- "${dico_train}")
  wget -c "https://dl.fbaipublicfiles.com/arrival/dictionaries/${DICO}" -P data/
fi

dico_valid=data/${s1}-${t1}.5000-6500.txt
if [ ! -f "${dico_valid}" ]; then
  DICO=$(basename -- "${dico_valid}")
  wget -c "https://dl.fbaipublicfiles.com/arrival/dictionaries/${DICO}" -P data/
fi

dico_test=data/${s2}-${t2}.0-5000.txt
if [ ! -f "${dico_test}" ]; then
  DICO=$(basename -- "${dico_test}")
  wget -c "https://dl.fbaipublicfiles.com/arrival/dictionaries/${DICO}" -P data/
fi

src_emb1=data/wiki.${s1}.vec
if [ ! -f "${src_emb1}" ]; then
  EMB=$(basename -- "${src_emb1}")
  wget -c "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/${EMB}" -P data/
fi

tgt_emb1=data/wiki.${t1}.vec
if [ ! -f "${tgt_emb1}" ]; then
  EMB=$(basename -- "${tgt_emb1}")
  wget -c "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/${EMB}" -P data/
fi

src_emb2=data/wiki.${s2}.vec
if [ ! -f "${src_emb2}" ]; then
  EMB=$(basename -- "${src_emb2}")
  wget -c "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/${EMB}" -P data/
fi

tgt_emb2=data/wiki.${t2}.vec
if [ ! -f "${tgt_emb2}" ]; then
  EMB=$(basename -- "${tgt_emb2}")
  wget -c "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/${EMB}" -P data/
fi

output_src1=alignement/round1/${s1}.vec
output_tgt1=alignement/round1/${t1}.vec
output_src2=alignement/round2/${s2}.vec
output_tgt2=alignement/round2/${t2}.vec

#python3 align.py --src_emb "${src_emb}" --tgt_emb "${tgt_emb}" \
#  --dico_train "${dico_train}" --dico_test "${dico_test}" --output "${output}" \
#  --lr 25 --niter 10

#Aligning Embeddings
python3 unsup_align.py --model_src "${src_emb1}" --model_tgt "${tgt_emb1}" \
    --lexicon "${dico_train}" --output_src "${output_src1}" --output_tgt "${output_tgt1}" 
  
python3 unsup_align.py --model_src "${src_emb2}" --model_tgt "${tgt_emb2}" \
    --lexicon "${dico_test}" --output_src "${output_src2}" --output_tgt "${output_tgt2}"  
    
#Extracting queries with different size   

python3 query_extraction.py --src_emb_train "${output_src1}" --tgt_emb_train "${output_tgt1}" \
    --src_emb_test "${output_src2}" --tgt_emb_test "${output_tgt2}" --output_dir 'query_8/' \
    --dico_train "${dico_train}" --dico_valid "${dico_valid}" --dico_test "${dico_test}" \
    --query_size 8 --query_relevance_type 'binary' --add_csls_coord true \
    --k_csls 10 --add_word_coord false --add_query_coord false 
 
python3 query_extraction.py --src_emb_train "${output_src1}" --tgt_emb_train "${output_tgt1}" \
    --src_emb_test "${output_src2}" --tgt_emb_test "${output_tgt2}" --output_dir 'query_10/' \
    --dico_train "${dico_train}" --dico_valid "${dico_valid}" --dico_test "${dico_test}" \
    --query_size 10 --query_relevance_type 'binary' --add_csls_coord true \
    --k_csls 10 --add_word_coord false --add_query_coord false 
    
python3 query_extraction.py --src_emb_train "${output_src1}" --tgt_emb_train "${output_tgt1}" \
    --src_emb_test "${output_src2}" --tgt_emb_test "${output_tgt2}" --output_dir 'query_12/' \
    --dico_train "${dico_train}" --dico_valid "${dico_valid}" --dico_test "${dico_test}" \
    --query_size 12 --query_relevance_type 'binary' --add_csls_coord true \
    --k_csls 10 --add_word_coord false --add_query_coord false 
 
python3 query_extraction.py --src_emb_train "${output_src1}" --tgt_emb_train "${output_tgt1}" \
    --src_emb_test "${output_src2}" --tgt_emb_test "${output_tgt2}" --output_dir 'query_14/' \
    --dico_train "${dico_train}" --dico_valid "${dico_valid}" --dico_test "${dico_test}" \
    --query_size 14 --query_relevance_type 'binary' --add_csls_coord true \
    --k_csls 10 --add_word_coord false --add_query_coord false 
    
python3 query_extraction.py --src_emb_train "${output_src1}" --tgt_emb_train "${output_tgt1}" \
    --src_emb_test "${output_src2}" --tgt_emb_test "${output_tgt2}" --output_dir 'query_16/' \
    --dico_train "${dico_train}" --dico_valid "${dico_valid}" --dico_test "${dico_test}" \
    --query_size 16 --query_relevance_type 'binary' --add_csls_coord true \
    --k_csls 10 --add_word_coord false --add_query_coord false 
 
python3 query_extraction.py --src_emb_train "${output_src1}" --tgt_emb_train "${output_tgt1}" \
    --src_emb_test "${output_src2}" --tgt_emb_test "${output_tgt2}" --output_dir 'query_18/' \
    --dico_train "${dico_train}" --dico_valid "${dico_valid}" --dico_test "${dico_test}" \
    --query_size 18 --query_relevance_type 'binary' --add_csls_coord true \
    --k_csls 10 --add_word_coord false --add_query_coord false 
    
python3 query_extraction.py --src_emb_train "${output_src1}" --tgt_emb_train "${output_tgt1}" \
    --src_emb_test "${output_src2}" --tgt_emb_test "${output_tgt2}" --output_dir 'query_20/' \
    --dico_train "${dico_train}" --dico_valid "${dico_valid}" --dico_test "${dico_test}" \
    --query_size 20 --query_relevance_type 'binary' --add_csls_coord true \
    --k_csls 10 --add_word_coord false --add_query_coord false 
    
python3 query_extraction.py --src_emb_train "${output_src1}" --tgt_emb_train "${output_tgt1}" \
    --src_emb_test "${output_src2}" --tgt_emb_test "${output_tgt2}" --output_dir 'query_25/' \
    --dico_train "${dico_train}" --dico_valid "${dico_valid}" --dico_test "${dico_test}" \
    --query_size 25 --query_relevance_type 'binary' --add_csls_coord true \
    --k_csls 10 --add_word_coord false --add_query_coord false 
    
python3 query_extraction.py --src_emb_train "${output_src1}" --tgt_emb_train "${output_tgt1}" \
    --src_emb_test "${output_src2}" --tgt_emb_test "${output_tgt2}" --output_dir 'query_30/' \
    --dico_train "${dico_train}" --dico_valid "${dico_valid}" --dico_test "${dico_test}" \
    --query_size 30 --query_relevance_type 'binary' --add_csls_coord true \
    --k_csls 10 --add_word_coord false --add_query_coord false 
    
python3 query_extraction.py --src_emb_train "${output_src1}" --tgt_emb_train "${output_tgt1}" \
    --src_emb_test "${output_src2}" --tgt_emb_test "${output_tgt2}" --output_dir 'query_40/' \
    --dico_train "${dico_train}" --dico_valid "${dico_valid}" --dico_test "${dico_test}" \
    --query_size 40 --query_relevance_type 'binary' --add_csls_coord true \
    --k_csls 10 --add_word_coord false --add_query_coord false 
    
#Using different query size with two most efficient loss

python3 tf_ranking_libsvm.py --train_path 'query_8/train' --vali_path 'query_8/valid' \
    --test_path 'query_8/test' --output_dir 'res/query_size/ndcg_loss_1_size_8' --group_size 1 --loss "approx_ndcg_loss" \
    --num_train_steps 100000 --num_features 11 --query_relevance_type 'binary' --query_size 8
    
python3 tf_ranking_libsvm.py --train_path 'query_8/train' --vali_path 'query_8/valid' \
    --test_path 'query_8/test' --output_dir 'res/query_size/ndcg_loss_2_size_8' --group_size 2 --loss "approx_ndcg_loss" \
    --num_train_steps 100000 --num_features 11 --query_relevance_type 'binary' --query_size 8
    
    
    
python3 tf_ranking_libsvm.py --train_path 'query_10/train' --vali_path 'query_10/valid' \
    --test_path 'query_10/test' --output_dir 'res/query_size/ndcg_loss_1_size_10' --group_size 1 --loss "approx_ndcg_loss" \
    --num_train_steps 100000 --num_features 11 --query_relevance_type 'binary' --query_size 10
    
python3 tf_ranking_libsvm.py --train_path 'query_10/train' --vali_path 'query_10/valid' \
    --test_path 'query_10/test' --output_dir 'res/query_size/ndcg_loss_2_size_10' --group_size 2 --loss "approx_ndcg_loss" \
    --num_train_steps 100000 --num_features 11 --query_relevance_type 'binary' --query_size 10
    
    
    
python3 tf_ranking_libsvm.py --train_path 'query_12/train' --vali_path 'query_12/valid' \
    --test_path 'query_12/test' --output_dir 'res/query_size/ndcg_loss_1_size_12' --group_size 1 --loss "approx_ndcg_loss" \
    --num_train_steps 100000 --num_features 11 --query_relevance_type 'binary' --query_size 12
    
python3 tf_ranking_libsvm.py --train_path 'query_12/train' --vali_path 'query_12/valid' \
    --test_path 'query_12/test' --output_dir 'res/query_size/ndcg_loss_2_size_12' --group_size 2 --loss "approx_ndcg_loss" \
    --num_train_steps 100000 --num_features 11 --query_relevance_type 'binary' --query_size 12
    
    
    
python3 tf_ranking_libsvm.py --train_path 'query_14/train' --vali_path 'query_14/valid' \
    --test_path 'query_14/test' --output_dir 'res/query_size/ndcg_loss_1_size_14' --group_size 1 --loss "approx_ndcg_loss" \
    --num_train_steps 100000 --num_features 11 --query_relevance_type 'binary' --query_size 14
    
python3 tf_ranking_libsvm.py --train_path 'query_14/train' --vali_path 'query_14/valid' \
    --test_path 'query_14/test' --output_dir 'res/query_size/ndcg_loss_2_size_14' --group_size 2 --loss "approx_ndcg_loss" \
    --num_train_steps 100000 --num_features 11 --query_relevance_type 'binary' --query_size 14
    
    
    
python3 tf_ranking_libsvm.py --train_path 'query_16/train' --vali_path 'query_16/valid' \
    --test_path 'query_16/test' --output_dir 'res/query_size/ndcg_loss_1_size_16' --group_size 1 --loss "approx_ndcg_loss" \
    --num_train_steps 100000 --num_features 11 --query_relevance_type 'binary' --query_size 16
    
python3 tf_ranking_libsvm.py --train_path 'query_16/train' --vali_path 'query_16/valid' \
    --test_path 'query_16/test' --output_dir 'res/query_size/ndcg_loss_2_size_16' --group_size 2 --loss "approx_ndcg_loss" \
    --num_train_steps 100000 --num_features 11 --query_relevance_type 'binary' --query_size 16
    
    
    
python3 tf_ranking_libsvm.py --train_path 'query_18/train' --vali_path 'query_18/valid' \
    --test_path 'query_18/test' --output_dir 'res/query_size/ndcg_loss_1_size_18' --group_size 1 --loss "approx_ndcg_loss" \
    --num_train_steps 100000 --num_features 11 --query_relevance_type 'binary' --query_size 18
    
python3 tf_ranking_libsvm.py --train_path 'query_18/train' --vali_path 'query_18/valid' \
    --test_path 'query_18/test' --output_dir 'res/query_size/ndcg_loss_2_size_18' --group_size 2 --loss "approx_ndcg_loss" \
    --num_train_steps 100000 --num_features 11 --query_relevance_type 'binary' --query_size 18
    
    
    
python3 tf_ranking_libsvm.py --train_path 'query_20/train' --vali_path 'query_20/valid' \
    --test_path 'query_20/test' --output_dir 'res/query_size/ndcg_loss_1_size_20' --group_size 1 --loss "approx_ndcg_loss" \
    --num_train_steps 100000 --num_features 11 --query_relevance_type 'binary' --query_size 20
    
python3 tf_ranking_libsvm.py --train_path 'query_20/train' --vali_path 'query_20/valid' \
    --test_path 'query_20/test' --output_dir 'res/query_size/ndcg_loss_2_size_20' --group_size 2 --loss "approx_ndcg_loss" \
    --num_train_steps 100000 --num_features 11 --query_relevance_type 'binary' --query_size 20
    
    
    
python3 tf_ranking_libsvm.py --train_path 'query_25/train' --vali_path 'query_25/valid' \
    --test_path 'query_25/test' --output_dir 'res/query_size/ndcg_loss_1_size_25' --group_size 1 --loss "approx_ndcg_loss" \
    --num_train_steps 100000 --num_features 11 --query_relevance_type 'binary' --query_size 25
    
python3 tf_ranking_libsvm.py --train_path 'query_25/train' --vali_path 'query_25/valid' \
    --test_path 'query_25/test' --output_dir 'res/query_size/ndcg_loss_2_size_25' --group_size 2 --loss "approx_ndcg_loss" \
    --num_train_steps 100000 --num_features 11 --query_relevance_type 'binary' --query_size 25
    
    
    
python3 tf_ranking_libsvm.py --train_path 'query_30/train' --vali_path 'query_30/valid' \
    --test_path 'query_30/test' --output_dir 'res/query_size/ndcg_loss_1_size_30' --group_size 1 --loss "approx_ndcg_loss" \
    --num_train_steps 100000 --num_features 11 --query_relevance_type 'binary' --query_size 30
    
python3 tf_ranking_libsvm.py --train_path 'query_30/train' --vali_path 'query_30/valid' \
    --test_path 'query_30/test' --output_dir 'res/query_size/ndcg_loss_2_size_30' --group_size 2 --loss "approx_ndcg_loss" \
    --num_train_steps 100000 --num_features 11 --query_relevance_type 'binary' --query_size 30
    
    
python3 tf_ranking_libsvm.py --train_path 'query_40/train' --vali_path 'query_40/valid' \
    --test_path 'query_40/test' --output_dir 'res/query_size/ndcg_loss_1_size_40' --group_size 1 --loss "approx_ndcg_loss" \
    --num_train_steps 100000 --num_features 11 --query_relevance_type 'binary' --query_size 40
    
python3 tf_ranking_libsvm.py --train_path 'query_40/train' --vali_path 'query_40/valid' \
    --test_path 'query_40/test' --output_dir 'res/query_size/ndcg_loss_2_size_40' --group_size 2 --loss "approx_ndcg_loss" \
    --num_train_steps 100000 --num_features 11 --query_relevance_type 'binary' --query_size 40
 
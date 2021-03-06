#!/bin/usr/env sh

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

if [ ! -d query/ ]; then
  mkdir -p query;
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

python3 unsup_align.py --model_src "${src_emb1}" --model_tgt "${tgt_emb1}" \
    --lexicon "${dico_train}" --output_src "${output_src1}" --output_tgt "${output_tgt1}" 
  
python3 unsup_align.py --model_src "${src_emb2}" --model_tgt "${tgt_emb2}" \
    --lexicon "${dico_test}" --output_src "${output_src2}" --output_tgt "${output_tgt2}"  

python3 query_extraction.py --src_emb_train "${output_src1}" --tgt_emb_train "${output_tgt1}" \
    --src_emb_test "${output_src2}" --tgt_emb_test "${output_tgt2}" --output_dir 'query/' \
    --dico_train "${dico_train}" --dico_valid "${dico_valid}" --dico_test "${dico_test}" \
    --query_size 10 --query_relevance_type 'binary' --add_csls_coord true \
    --k_csls 10 --testing_query true --add_word_coord false --add_query_coord false 

# Group size 1 
echo "Group size 1"

python3 tf_ranking_libsvm.py --train_path 'query/train' --vali_path 'query/valid' \
    --test_path 'query/test' --output_dir 'res/groupsize_1/approx_ndcg_loss' --group_size 1 --loss "approx_ndcg_loss" \
    --num_train_steps 100000 --num_features 11 --query_relevance_type 'binary' --query_size 10
    
python3 tf_ranking_libsvm.py --train_path 'query_binary/train' --vali_path 'query_binary/valid' \
    --test_path 'query_binary/test' --output_dir 'res/groupsize_1/list_mle_loss' --group_size 1 --loss "list_mle_loss" \
    --num_train_steps 100000 --num_features 11 --query_relevance_type 'binary' --query_size 10
    
# Group size 2
echo "Group size 2"

python3 tf_ranking_libsvm.py --train_path 'query/train' --vali_path 'query/valid' \
    --test_path 'query/test' --output_dir 'res/groupsize_2/approx_ndcg_loss' --group_size 2 --loss "approx_ndcg_loss" \
    --num_train_steps 100000 --num_features 11 --query_relevance_type 'binary' --query_size 10
    
python3 tf_ranking_libsvm.py --train_path 'query_binary/train' --vali_path 'query_binary/valid' \
    --test_path 'query_binary/test' --output_dir 'res/groupsize_2/list_mle_loss' --group_size 2 --loss "list_mle_loss" \
    --num_train_steps 100000 --num_features 11 --query_relevance_type 'binary' --query_size 10
    
# Group size 3
echo "Group size 1"

python3 tf_ranking_libsvm.py --train_path 'query/train' --vali_path 'query/valid' \
    --test_path 'query/test' --output_dir 'res/groupsize_3/approx_ndcg_loss' --group_size 3 --loss "approx_ndcg_loss" \
    --num_train_steps 100000 --num_features 11 --query_relevance_type 'binary' --query_size 10
    
python3 tf_ranking_libsvm.py --train_path 'query_binary/train' --vali_path 'query_binary/valid' \
    --test_path 'query_binary/test' --output_dir 'res/groupsize_3/list_mle_loss' --group_size 3 --loss "list_mle_loss" \
    --num_train_steps 100000 --num_features 11 --query_relevance_type 'binary' --query_size 10
    
# Group size 4
echo "Group size 4"

python3 tf_ranking_libsvm.py --train_path 'query/train' --vali_path 'query/valid' \
    --test_path 'query/test' --output_dir 'res/groupsize_4/approx_ndcg_loss' --group_size 4 --loss "approx_ndcg_loss" \
    --num_train_steps 100000 --num_features 11 --query_relevance_type 'binary' --query_size 10
    
python3 tf_ranking_libsvm.py --train_path 'query_binary/train' --vali_path 'query_binary/valid' \
    --test_path 'query_binary/test' --output_dir 'res/groupsize_4/list_mle_loss' --group_size 4 --loss "list_mle_loss" \
    --num_train_steps 100000 --num_features 11 --query_relevance_type 'binary' --query_size 10

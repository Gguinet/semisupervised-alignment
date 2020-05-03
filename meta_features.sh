#!/bin/usr/env sh

set -e
s1=$1
t1=$2

s2=$1
t2=$3

echo "Starting $1-$3 induction, using $2 as pivot"


if [ ! -d data/ ]; then
  mkdir -p data;
fi


if [ ! -d res/${s2}-${t2}/ ]; then
  mkdir -p res/${s2}-${t2};
fi

if [ ! -d query/${s1}-${t1}/ ]; then
  mkdir -p query/${s1}-${t1};
fi
if [ ! -d query/${s2}-${t2}/ ]; then
  mkdir -p query/${s2}-${t2};
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

#Aligning embeddings

output_src1=alignment/${s1}-${t1}/${s1}.vec
output_tgt1=alignment/${s1}-${t1}/${t1}.vec

if [ ! -d alignment/${s1}-${t1}/ ]; then
  mkdir -p alignment/${s1}-${t1}
  python3 unsup_align.py --model_src "${src_emb1}" --model_tgt "${tgt_emb1}" \
    --lexicon "${dico_train}" --output_src "${output_src1}" --output_tgt "${output_tgt1}" ;
fi

output_src2=alignment/${s2}-${t2}/${s2}.vec
output_tgt2=alignment/${s2}-${t2}/${t2}.vec

if [ ! -d alignment/${s2}-${t2}/ ]; then
  mkdir -p alignment/${s2}-${t2}
  python3 unsup_align.py --model_src "${src_emb2}" --model_tgt "${tgt_emb2}" \
    --lexicon "${dico_test}" --output_src "${output_src2}" --output_tgt "${output_tgt2}"  ;
fi


# Query Extraction 
# For the testing, we do not force the presence of ground truth in each query


################################# Benchmarking #################################

train_path=query/${s1}-${t1}/benchmark/train
if [ ! -f "${train_path}" ]; then
    python3 single_query_extract.py --src_emb "${output_src1}" --tgt_emb "${output_tgt1}" \
        --filename "${train_path}" --dico "${dico_train}" --query_size 10 \
        --query_relevance_type 'binary' --add_csls_coord true --k_csls 10 \
        --testing_query false --add_word_coord false --add_query_coord false \
        --discard_empty_query false --add_meta_features false --center_meta_features false \
        --nn_size_meta_features 10;
fi

test_path=query/${s2}-${t2}/benchmark/test
if [ ! -f "${test_path}" ]; then
    python3 single_query_extract.py --src_emb "${output_src2}" --tgt_emb "${output_tgt2}" \
        --filename "${test_path}" --dico "${dico_test}" --query_size 10 \
        --query_relevance_type 'binary' --add_csls_coord true --k_csls 10 \
        --testing_query true --add_word_coord false --add_query_coord false \
        --discard_empty_query false --add_meta_features false --center_meta_features false \
        --nn_size_meta_features 10;
fi
    
# BLI Induction

output_dir1=res/${s2}-${t2}/${t1}/benchmark/approx_ndcg_loss_group_4

python3 tf_ranking_libsvm.py --train_path "${train_path}" --vali_path "${test_path}" \
    --test_path "${test_path}" --output_dir "${output_dir1}" --group_size 4 --loss "approx_ndcg_loss" \
    --num_train_steps 100000 --num_features 11 --query_relevance_type 'binary' --query_size 10
    
    
################################# Experiment 1 #################################


#Using meta-features, small size of neigbhors for mf (10), not centered, small NN

train_path=query/${s1}-${t1}/mf_small/train
if [ ! -f "${train_path}" ]; then
    python3 single_query_extract.py --src_emb "${output_src1}" --tgt_emb "${output_tgt1}" \
        --filename "${train_path}" --dico "${dico_train}" --query_size 10 \
        --query_relevance_type 'binary' --add_csls_coord true --k_csls 10 \
        --testing_query false --add_word_coord false --add_query_coord false \
        --discard_empty_query false --add_meta_features true --center_meta_features false \
        --nn_size_meta_features 10;
fi

test_path=query/${s2}-${t2}/mf_small/test
if [ ! -f "${test_path}" ]; then
    python3 single_query_extract.py --src_emb "${output_src2}" --tgt_emb "${output_tgt2}" \
        --filename "${test_path}" --dico "${dico_test}" --query_size 10 \
        --query_relevance_type 'binary' --add_csls_coord true --k_csls 10 \
        --testing_query true --add_word_coord false --add_query_coord false \
        --discard_empty_query false --add_meta_features true --center_meta_features false \
        --nn_size_meta_features 10;
fi
    
# BLI Induction

output_dir2=res/${s2}-${t2}/${t1}/mf_small/approx_ndcg_loss_group_4

python3 tf_ranking_libsvm.py --train_path "${train_path}" --vali_path "${test_path}" \
    --test_path "${test_path}" --output_dir "${output_dir2}" --group_size 4 --loss "approx_ndcg_loss" \
    --num_train_steps 100000 --num_features 127 --query_relevance_type 'binary' --query_size 10
    

################################# Experiment 2 #################################
    
    
#Using meta-features, big size of neigbhors for mf (100), not centered, small NN

train_path=query/${s1}-${t1}/mf_big/train
if [ ! -f "${train_path}" ]; then
    python3 single_query_extract.py --src_emb "${output_src1}" --tgt_emb "${output_tgt1}" \
        --filename "${train_path}" --dico "${dico_train}" --query_size 10 \
        --query_relevance_type 'binary' --add_csls_coord true --k_csls 10 \
        --testing_query false --add_word_coord false --add_query_coord false \
        --discard_empty_query false --add_meta_features true --center_meta_features false \
        --nn_size_meta_features 100;
fi

test_path=query/${s2}-${t2}/mf_big/test
if [ ! -f "${test_path}" ]; then
    python3 single_query_extract.py --src_emb "${output_src2}" --tgt_emb "${output_tgt2}" \
        --filename "${test_path}" --dico "${dico_test}" --query_size 10 \
        --query_relevance_type 'binary' --add_csls_coord true --k_csls 10 \
        --testing_query true --add_word_coord false --add_query_coord false \
        --discard_empty_query false --add_meta_features true --center_meta_features false \
        --nn_size_meta_features 100;
fi
    
# BLI Induction

output_dir3=res/${s2}-${t2}/${t1}/mf_big/approx_ndcg_loss_group_4

python3 tf_ranking_libsvm.py --train_path "${train_path}" --vali_path "${test_path}" \
    --test_path "${test_path}" --output_dir "${output_dir3}" --group_size 4 --loss "approx_ndcg_loss" \
    --num_train_steps 100000 --num_features 127 --query_relevance_type 'binary' --query_size 10
    
    

################################# Experiment 3 #################################
    
    
#Using meta-features, big size of neigbhors for mf (100), centered, small NN

train_path=query/${s1}-${t1}/mf_big_centered/train
if [ ! -f "${train_path}" ]; then
    python3 single_query_extract.py --src_emb "${output_src1}" --tgt_emb "${output_tgt1}" \
        --filename "${train_path}" --dico "${dico_train}" --query_size 10 \
        --query_relevance_type 'binary' --add_csls_coord true --k_csls 10 \
        --testing_query false --add_word_coord false --add_query_coord false \
        --discard_empty_query false --add_meta_features true --center_meta_features true \
        --nn_size_meta_features 100;
fi

test_path=query/${s2}-${t2}/mf_big_centered/test
if [ ! -f "${test_path}" ]; then
    python3 single_query_extract.py --src_emb "${output_src2}" --tgt_emb "${output_tgt2}" \
        --filename "${test_path}" --dico "${dico_test}" --query_size 10 \
        --query_relevance_type 'binary' --add_csls_coord true --k_csls 10 \
        --testing_query true --add_word_coord false --add_query_coord false \
        --discard_empty_query false --add_meta_features true --center_meta_features true \
        --nn_size_meta_features 100;
fi
    
# BLI Induction

output_dir4=res/${s2}-${t2}/${t1}/mf_big_centered/approx_ndcg_loss_group_4

python3 tf_ranking_libsvm.py --train_path "${train_path}" --vali_path "${test_path}" \
    --test_path "${test_path}" --output_dir "${output_dir4}" --group_size 4 --loss "approx_ndcg_loss" \
    --num_train_steps 100000 --num_features 127 --query_relevance_type 'binary' --query_size 10
  
  
################################# Experiment 4 #################################
    
    
#Using meta-features, small size of neigbhors for mf (10), centered, small NN

train_path=query/${s1}-${t1}/mf_small_centered/train
if [ ! -f "${train_path}" ]; then
    python3 single_query_extract.py --src_emb "${output_src1}" --tgt_emb "${output_tgt1}" \
        --filename "${train_path}" --dico "${dico_train}" --query_size 10 \
        --query_relevance_type 'binary' --add_csls_coord true --k_csls 10 \
        --testing_query false --add_word_coord false --add_query_coord false \
        --discard_empty_query false --add_meta_features true --center_meta_features true \
        --nn_size_meta_features 10;
fi

test_path=query/${s2}-${t2}/mf_small_centered/test
if [ ! -f "${test_path}" ]; then
    python3 single_query_extract.py --src_emb "${output_src2}" --tgt_emb "${output_tgt2}" \
        --filename "${test_path}" --dico "${dico_test}" --query_size 10 \
        --query_relevance_type 'binary' --add_csls_coord true --k_csls 10 \
        --testing_query true --add_word_coord false --add_query_coord false \
        --discard_empty_query false --add_meta_features true --center_meta_features true \
        --nn_size_meta_features 10;
fi
    
# BLI Induction

output_dir5=res/${s2}-${t2}/${t1}/mf_small_centered/approx_ndcg_loss_group_4

python3 tf_ranking_libsvm.py --train_path "${train_path}" --vali_path "${test_path}" \
    --test_path "${test_path}" --output_dir "${output_dir5}" --group_size 4 --loss "approx_ndcg_loss" \
    --num_train_steps 100000 --num_features 127 --query_relevance_type 'binary' --query_size 10
    

################################# Experiment 5 #################################


#Using meta-features, small size of neigbhors for mf (10), not centered, big NN

train_path=query/${s1}-${t1}/mf_small/train
if [ ! -f "${train_path}" ]; then
    python3 single_query_extract.py --src_emb "${output_src1}" --tgt_emb "${output_tgt1}" \
        --filename "${train_path}" --dico "${dico_train}" --query_size 10 \
        --query_relevance_type 'binary' --add_csls_coord true --k_csls 10 \
        --testing_query false --add_word_coord false --add_query_coord false \
        --discard_empty_query false --add_meta_features true --center_meta_features false \
        --nn_size_meta_features 10;
fi

test_path=query/${s2}-${t2}/mf_small/test
if [ ! -f "${test_path}" ]; then
    python3 single_query_extract.py --src_emb "${output_src2}" --tgt_emb "${output_tgt2}" \
        --filename "${test_path}" --dico "${dico_test}" --query_size 10 \
        --query_relevance_type 'binary' --add_csls_coord true --k_csls 10 \
        --testing_query true --add_word_coord false --add_query_coord false \
        --discard_empty_query false --add_meta_features true --center_meta_features false \
        --nn_size_meta_features 10;
fi
    
# BLI Induction

output_dir6=res/${s2}-${t2}/${t1}/mf_small_bigNN/approx_ndcg_loss_group_4

python3 tf_ranking_libsvm_bigNN.py --train_path "${train_path}" --vali_path "${test_path}" \
    --test_path "${test_path}" --output_dir "${output_dir6}" --group_size 4 --loss "approx_ndcg_loss" \
    --num_train_steps 100000 --num_features 127 --query_relevance_type 'binary' --query_size 10
    

################################# Experiment 6 #################################
    
    
#Using meta-features, big size of neigbhors for mf (100), not centered, big NN

train_path=query/${s1}-${t1}/mf_big/train
if [ ! -f "${train_path}" ]; then
    python3 single_query_extract.py --src_emb "${output_src1}" --tgt_emb "${output_tgt1}" \
        --filename "${train_path}" --dico "${dico_train}" --query_size 10 \
        --query_relevance_type 'binary' --add_csls_coord true --k_csls 10 \
        --testing_query false --add_word_coord false --add_query_coord false \
        --discard_empty_query false --add_meta_features true --center_meta_features false \
        --nn_size_meta_features 100;
fi

test_path=query/${s2}-${t2}/mf_big/test
if [ ! -f "${test_path}" ]; then
    python3 single_query_extract.py --src_emb "${output_src2}" --tgt_emb "${output_tgt2}" \
        --filename "${test_path}" --dico "${dico_test}" --query_size 10 \
        --query_relevance_type 'binary' --add_csls_coord true --k_csls 10 \
        --testing_query true --add_word_coord false --add_query_coord false \
        --discard_empty_query false --add_meta_features true --center_meta_features false \
        --nn_size_meta_features 100;
fi
    
# BLI Induction

output_dir7=res/${s2}-${t2}/${t1}/mf_big_bigNN/approx_ndcg_loss_group_4

python3 tf_ranking_libsvm_bigNN.py.py --train_path "${train_path}" --vali_path "${test_path}" \
    --test_path "${test_path}" --output_dir "${output_dir7}" --group_size 4 --loss "approx_ndcg_loss" \
    --num_train_steps 100000 --num_features 127 --query_relevance_type 'binary' --query_size 10
    
    
################################# Experiment 7 #################################


#Using meta-features, small size of neigbhors for mf (10), not centered, extra big NN

train_path=query/${s1}-${t1}/mf_small/train
if [ ! -f "${train_path}" ]; then
    python3 single_query_extract.py --src_emb "${output_src1}" --tgt_emb "${output_tgt1}" \
        --filename "${train_path}" --dico "${dico_train}" --query_size 10 \
        --query_relevance_type 'binary' --add_csls_coord true --k_csls 10 \
        --testing_query false --add_word_coord false --add_query_coord false \
        --discard_empty_query false --add_meta_features true --center_meta_features false \
        --nn_size_meta_features 10;
fi

test_path=query/${s2}-${t2}/mf_small/test
if [ ! -f "${test_path}" ]; then
    python3 single_query_extract.py --src_emb "${output_src2}" --tgt_emb "${output_tgt2}" \
        --filename "${test_path}" --dico "${dico_test}" --query_size 10 \
        --query_relevance_type 'binary' --add_csls_coord true --k_csls 10 \
        --testing_query true --add_word_coord false --add_query_coord false \
        --discard_empty_query false --add_meta_features true --center_meta_features false \
        --nn_size_meta_features 10;
fi
    
# BLI Induction

output_dir8=res/${s2}-${t2}/${t1}/mf_small_extraNN/approx_ndcg_loss_group_4

python3 tf_ranking_libsvm_ExtraNN.py --train_path "${train_path}" --vali_path "${test_path}" \
    --test_path "${test_path}" --output_dir "${output_dir8}" --group_size 4 --loss "approx_ndcg_loss" \
    --num_train_steps 100000 --num_features 127 --query_relevance_type 'binary' --query_size 10
    

################################# Experiment 8 #################################
    
    
#Using meta-features, big size of neigbhors for mf (100), not centered, extra big NN

train_path=query/${s1}-${t1}/mf_big/train
if [ ! -f "${train_path}" ]; then
    python3 single_query_extract.py --src_emb "${output_src1}" --tgt_emb "${output_tgt1}" \
        --filename "${train_path}" --dico "${dico_train}" --query_size 10 \
        --query_relevance_type 'binary' --add_csls_coord true --k_csls 10 \
        --testing_query false --add_word_coord false --add_query_coord false \
        --discard_empty_query false --add_meta_features true --center_meta_features false \
        --nn_size_meta_features 100;
fi

test_path=query/${s2}-${t2}/mf_big/test
if [ ! -f "${test_path}" ]; then
    python3 single_query_extract.py --src_emb "${output_src2}" --tgt_emb "${output_tgt2}" \
        --filename "${test_path}" --dico "${dico_test}" --query_size 10 \
        --query_relevance_type 'binary' --add_csls_coord true --k_csls 10 \
        --testing_query true --add_word_coord false --add_query_coord false \
        --discard_empty_query false --add_meta_features true --center_meta_features false \
        --nn_size_meta_features 100;
fi
    
# BLI Induction

output_dir9=res/${s2}-${t2}/${t1}/mf_big_extraNN/approx_ndcg_loss_group_4

python3 tf_ranking_libsvm_ExtraNN.py --train_path "${train_path}" --vali_path "${test_path}" \
    --test_path "${test_path}" --output_dir "${output_dir9}" --group_size 4 --loss "approx_ndcg_loss" \
    --num_train_steps 100000 --num_features 127 --query_relevance_type 'binary' --query_size 10
    

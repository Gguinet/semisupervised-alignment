set -e
s1=$1
t1=$2

s2=$1
t2=$3

echo "Starting $1-$3 induction, using $2 as pivot"


if [ ! -d data/ ]; then
  mkdir -p data;
fi


if [ ! -d res_sup/${s2}-${t2}/ ]; then
  mkdir -p res_sup/${s2}-${t2};
fi

if [ ! -d query_sup/${s1}-${t1}/ ]; then
  mkdir -p query_sup/${s1}-${t1};
fi
if [ ! -d query_sup/${s2}-${t2}/ ]; then
  mkdir -p query_sup/${s2}-${t2};
fi

dico_train=data/${s1}-${t1}.0-5000.txt
if [ ! -f "${dico_train}" ]; then
  DICO=$(basename -- "${dico_train}")
  wget -c "https://dl.fbaipublicfiles.com/arrival/dictionaries/${DICO}" -P data/
fi

dico_test=data/${s2}-${t2}.0-5000.txt
if [ ! -f "${dico_test}" ]; then
  DICO=$(basename -- "${dico_test}")
  wget -c "https://dl.fbaipublicfiles.com/arrival/dictionaries/${DICO}" -P data/
fi

dico_val1=data/${s1}-${t1}.5000-6500.txt
if [ ! -f "${dico_val1}" ]; then
  DICO=$(basename -- "${dico_val1}")
  wget -c "https://dl.fbaipublicfiles.com/arrival/dictionaries/${DICO}" -P data/
fi

dico_val2=data/${s2}-${t2}.5000-6500.txt
if [ ! -f "${dico_val2}" ]; then
  DICO=$(basename -- "${dico_val2}")
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

output_src1=alignment_sup/${s1}-${t1}/${s1}.vec
output_tgt1=alignment_sup/${s1}-${t1}/${t1}.vec

if [ ! -d alignment_sup/${s1}-${t1}/ ]; then
  mkdir -p alignment_sup/${s1}-${t1}
  python3 align.py --src_emb "${src_emb1}" --tgt_emb "${tgt_emb1}" \
                   --dico_train "${dico_train}" --dico_test "${dico_val1}" \
                   --output_src "${output_src1}" --output_tgt "${output_tgt1}" ;
fi

output_src2=alignment_sup/${s2}-${t2}/${s2}.vec
output_tgt2=alignment_sup/${s2}-${t2}/${t2}.vec

if [ ! -d alignment_sup/${s2}-${t2}/ ]; then
  mkdir -p alignment_sup/${s2}-${t2}
  python3 align.py --src_emb "${src_emb2}" --tgt_emb "${tgt_emb2}" \
                   --dico_train "${dico_test}" --dico_test "${dico_val2}" \
                   --output_src "${output_src2}" --output_tgt "${output_tgt2}" ;
fi


# Query Extraction 
# For the testing, we do not force the presence of ground truth in each query

train_path=query_sup/${s1}-${t1}/train
if [ ! -f "${train_path}" ]; then
    python3 single_query_extract.py --src_emb "${output_src1}" --tgt_emb "${output_tgt1}" \
        --filename "${train_path}" --dico "${dico_train}" --query_size 10 \
        --query_relevance_type 'binary' --add_csls_coord true --k_csls 10 \
        --testing_query false --add_word_coord false --add_query_coord false ;
fi

test_path=query_sup/${s2}-${t2}/test
if [ ! -f "${test_path}" ]; then
    python3 single_query_extract.py --src_emb "${output_src2}" --tgt_emb "${output_tgt2}" \
        --filename "${test_path}" --dico "${dico_test}" --query_size 10 \
        --query_relevance_type 'binary' --add_csls_coord true --k_csls 10 \
        --testing_query true --add_word_coord false --add_query_coord false \
        --discard_empty_query true ;
fi

    
# BLI Induction
#queryy_full

output_dir1=res_sup/${s2}-${t2}/${t1}/approx_ndcg_loss_group_2

python3 tf_ranking_libsvm.py --train_path "${train_path}" --vali_path "${test_path}" \
    --test_path "${test_path}" --output_dir "${output_dir1}" --group_size 2 --loss "approx_ndcg_loss" \
    --num_train_steps 100000 --num_features 11 --query_relevance_type 'binary' --query_size 10
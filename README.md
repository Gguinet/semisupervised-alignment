# semisupervised-alignement

## Instructions for google collab

**Settings for Virtual Machine:**

* europe-west2-c (London)
* 8 precesseurs virtuels, n1-highmem-8, haute capacité de mémoire
* Ubuntu, Version 18.04, 200 Go mémoire
* Pare-feu: Autoriser traffic HTTPS et HTTP

**Ligne de commande:**

```
sudo apt install python
sudo apt-get install python3-distutils
sudo curl "https://bootstrap.pypa.io/get-pip.py" -o "get-pip.py"
python3 get-pip.py
python3 -m pip install numpy
python3 -m pip install  tensorflow_ranking
python3 -m pip install POT
python3 -m pip install pandas
python3 -m pip install sklearn 
git clone https://github.com/Gguinet/semisupervised-alignement.git ("Entrer compte et mdp git")
cd semisupervised-alignement
sh main.sh
```

**Instructions Python:**

```
python3 query_extraction.py --src_emb_train "${output_src1}" --tgt_emb_train "${output_tgt1}" \
    --src_emb_test "${output_src2}" --tgt_emb_test "${output_tgt2}" --output_dir 'query/' \
    --dico_train "${dico_train}" --dico_valid "${dico_valid}" --dico_test "${dico_test}" \
    --query_size 10 --query_extractor 'binary'

python3 tf_ranking_libsvm.py --train_path 'query/train' --vali_path 'query/valid' \
    --test_path 'query/test' --output_dir 'tf_res' --group_size 1 --loss "approx_ndcg_loss" \
    --num_train_steps 100000
```

**Loss functions:**

* sigmoid_cross_entropy_loss (recommanded for binary relevance)
* approx_ndcg_loss (recommanded for ndcg optimization)
* pairwise_hinge_loss
* pairwise_logistic_loss
* pairwise_soft_zero_one_loss
* softmax_loss
* mean_squared_loss
* list_mle_loss
* approx_mrr_loss
* gumbel_approx_ndcg_loss
* neural_sort_cross_entropy_loss
* gumbel_neural_sort_cross_entropy_loss

**Parameters for optimizations:**

* Triplets of languages 
* Loss functions
* Query extractor: binary or continuous
* Size of queries
* Group Size: Try first 2 instead of 1 (exponential growth)

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
python3 -m pip install numpy tensorflow_ranking POT pandas sklearn 
git clone https://github.com/Gguinet/semisupervised-alignement.git ("Entrer compte et mdp git")
cd semisupervised-alignement
sh main.sh
```

**Instructions Python:**

```
python3 query_extraction.py --src_emb_train "${output_src1}" --tgt_emb_train "${output_tgt1}" \
    --src_emb_test "${output_src2}" --tgt_emb_test "${output_tgt2}" --output_dir 'query/' \
    --dico_train "${dico_train}" --dico_valid "${dico_valid}" --dico_test "${dico_test}" \
    --query_size 10 --query_relevance_type 'binary' --add_csls_coord false \
    --k_csls 10 --add_word_coord false --add_query_coord false 

python3 tf_ranking_libsvm.py --train_path 'query/train' --vali_path 'query/valid' \
    --test_path 'query/test' --output_dir 'tf_res' --group_size 1 --loss "approx_ndcg_loss" \
    --num_train_steps 100000 --query_relevance_type 'binary' --query_size 10
```

**Code framework:**

* ```Ablation_study``` collects the bash codes associated with the ablation tests.
* ```bli.sh``` takes as input two languages and run all the framework
* ```lang_impact.sh``` and ```lang_variation.sh``` run simulations for all languages
* ```main.sh``` is the older version of ```bli.sh```
* ```main_pre_align.sh``` and ```pre_alignment.sh``` are precomputing all the alignment for computations (to be deleted)
* ```query_extraction.py``` is extracting test, validation and training queries for two langues
* ```single_query_extract.py``` is only exctracting one query (usefull if re-use of query)
* ```query_aux.py``` is the auxiliary functions for query extraction
* ``tf_ranking_libsvm.py`` handle the learning to rank
* ``unsup_align.py`` is the facebook code for unsupervised alignment
* ``align.py`` and ``unsup_multialign.py`` are the supervised and multi alignment facebook code (not used)
* ``utils.py`` are auxiliary functions for alignment

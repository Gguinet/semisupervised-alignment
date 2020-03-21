# semisupervised-alignement

## Instructions for google collab

**Settings for Virtual Machine:**

* europe-west2-c (London)
* 8 precesseurs virtuels, n1-highmem-8, haute capacité de mémoire
* Ubuntu, Version 18.04, 200 Go mémoire
* Pare-feu: Autoriser traffic HTTPS et HTTP

**Ligne de commande Python:**

* sudo apt install python
* sudo apt-get install python3-distutils
* sudo curl "https://bootstrap.pypa.io/get-pip.py" -o "get-pip.py"
* python3 get-pip.py
* python3 -m pip install numpy
* python3 -m pip install  tensorflow_ranking
* python3 -m pip install POT
* python3 -m pip install pandas
* python3 -m pip install sklearn 
* git clone https://github.com/Gguinet/semisupervised-alignement.git ("Entrer compte et mdp git")
* cd semisupervised-alignement
* sh main.sh

# Install correct version of python and pip
sudo apt install python
sudo apt-get install python3-distutils
sudo curl "https://bootstrap.pypa.io/get-pip.py" -o "get-pip.py"
python3 get-pip.py

# Install modules 
python3 -m pip install numpy
python3 -m pip install tensorflow_ranking
python3 -m pip install POT
python3 -m pip install pandas
python3 -m pip install sklearn 
python3 -m pip install munkres
python3 -m pip install scipy
python3 -m pip install pymfe

# Clone the code
git clone https://github.com/Gguinet/semisupervised-alignement.git

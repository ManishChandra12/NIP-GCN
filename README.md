# NIP-GCN

## Setup
#### Requires python >= 3.8.5
1. Clone the repo
```
git clone https://github.com/ManishChandra12/NIP-GCN.git
```
2. Install virtualenv
```
pip install virtualenv
```
3. cd to the project directory
```
cd NIP-GCN
```
4. Create the virtual environment
```
virtualenv env
```
5. Activate the virtual environment
```
source env/bin/activate
```
6. Run the setup script
```
./setup.sh
```

## Training and Evaluation
7. Run the following to train and evaluate on `M10` dataset
```
./run.sh
```
Change `M10` in `run.sh` and `self.dataset` in `config.py` to `dblp`, `covid` or `covid_title` to train and evaluate on DBLP, Covid-Full and Covid-Title datasets respectively. Refer to `config.py` for other hyperparameter settings.

# cuub

Code to reproduce experiments from the paper [TODO](...).

## Setup

Execute the following to download the data from [here](https://www.cs.ubc.ca/~drgraham/datasets.html) and unpack it into a folder named `icar/`:
```
mkdir icar
wget https://www.cs.ubc.ca/~drgraham/datasets/dataset_icar.zip
unzip dataset_icar.zip -d icar/
```

## The Case of Few Configurations -- Comparison with UP

Compare CUUB to UP and the Naive procedure by duplicating the experiments from [Graham et al., 2023](https://arxiv.org/abs/2310.20401) (Figure ...), and also plotting the total time spent on each configuration as a function of its utility:
```
python up_experiment.py [minisat | cplex_rcw | cplex_region] [seed = {9858}]
```


## The Case of Many Configurations

Compare the many-configuration version of to the few-configuration version (Figure ...):
```
python many_experiment.py [minisat | cplex_rcw | cplex_region | synthetic] [seed = {9858}]
```

## Synthetic Data 

....
```
python synthetic_demo.py
```


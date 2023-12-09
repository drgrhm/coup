# cuub

Code to reproduce experiments from the paper [TODO](...).

## Setup

Execute the following to download the data from [here](https://www.cs.ubc.ca/~drgraham/datasets.html) and unpack it into a folder named `icar/`:
```
mkdir icar
wget https://www.cs.ubc.ca/~drgraham/datasets/dataset_icar.zip
unzip dataset_icar.zip -d icar/
```

Set up directories: 
```
mkdir dat img
```

## The Case of Few Configurations -- Comparison with UP

Compare CUUB to UP and the Naive procedure by duplicating the experiments from [Graham et al., 2023](https://arxiv.org/abs/2310.20401) (Figure ...):
```
python up_experiment.py [minisat | cplex_rcw | cplex_region] [seed = {9858}]
```

Compare the total time spent by CUUB and UP on each configuration (Figure ...):
```
python total_time_experiment.py [minisat | cplex_rcw | cplex_region] [seed = {9858}]
```

## The Case of Many Configurations

# COUP --- Continuous, Optimistic Utilitarian Configuration

Code to reproduce experiments from the paper [TODO](...).

## Setup

Execute the following to download the data from [here](https://www.cs.ubc.ca/~drgraham/datasets.html) and unpack it into a folder named `icar/`:
```
mkdir icar
wget https://www.cs.ubc.ca/~drgraham/datasets/dataset_icar.zip
unzip dataset_icar.zip -d icar/
```

## The Case of Few Configurations: OUP

Generate the plots for Figures ... comparing OUP with UP [Graham et al., 2023](https://arxiv.org/abs/2310.20401).
```
python up_experiment.py [minisat | cplex_rcw | cplex_region]
```


## The Case of Many ConfigurationsL COUP

Compare the many-configuration version of to the few-configuration version (Figure ...):
```
python many_experiment.py [minisat | cplex_rcw | cplex_region | synthetic]
```

## Synthetic Data 

....
```
python synthetic_demo.py
```


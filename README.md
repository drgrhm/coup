# cuub

Code to reproduce experiments from the paper [TODO](...).

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

Compare CUUB to UP and the Naive procedure by duplicating the experiments from [Graham et al., 2023](https://arxiv.org/abs/2310.20401) (Figure ...):
```
python up_experiment.py [minisat | cplex_rcw | cplex_region] [seed]
```

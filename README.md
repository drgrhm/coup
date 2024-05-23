# COUP - Continuous Optimistic Utilitarian Configuration

Code to reproduce experiments from the paper *Utilitarian Algorithm Configuration for Infinite Parameter Spaces*, available [here](...).

## Setup

Execute the following to download the data from [here](https://www.cs.ubc.ca/~drgraham/datasets.html) and unpack it into a folder named `icar/`:
```
mkdir icar
wget https://www.cs.ubc.ca/~drgraham/datasets/dataset_icar.zip
unzip dataset_icar.zip -d icar/
```

## Generating plots

Generate the plots for Figures 1, 2 and 6 comparing OUP with UP:
```
python up_experiment.py [minisat | cplex_rcw | cplex_region]
```

Generate the plots for Figure 3, comparing COUP with OUP:
```
python many_experiment.py [minisat | cplex_rcw | cplex_region]
```

Generate the plots for Figure 4, showing the explorative power of COUP:
```
python explore_experiment.py [minisat | cplex_rcw | cplex_region]
```

Generate the plots for Figure 5, showing the improvement of the new doubling condition:
```
python dubcond_experiment.py [minisat | cplex_rcw | cplex_region]
```



# cuub

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

Duplicate the experiments from UP paper [...], comparing to CUUB (Figure ...):
```
python up_experiment.py [minisat | cplex_rcw | cplex_region] [seed]
```

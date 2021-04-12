# SPatial transcriptomics imputation (spImpute)

## Required dependencies

  * pandas=0.25.3
  * numpy=1.18.5
  * matplotlib=3.3.4
  * statsmodels=0.12.0
  * scipy=1.6.1
  * tqdm=4.56.0

## Data prerequisites

  1. Raw count matrix without normalization, rows as samples and columns as genes
  2. First column as the coordinates of each sample with the format of 'XxY', eg. '12x23'

## Run example experiments

    python3 spImpute.py -f test_data/raw.csv -o test_data/imputed.csv -l cpm 

  ### Visualize major tissue components
  
    python3 visualize_components.py test_data/imputed_cluster_info.csv test_data/cluster.png
    

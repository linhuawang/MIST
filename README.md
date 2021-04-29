# Missing value Imputation for Spatially resolved Transcriptomics (MIST)

## Required dependencies

  * pandas=0.25.3
  * numpy=1.18.5
  * matplotlib=3.3.4
  * statsmodels=0.12.0
  * scipy=1.6.1
  * tqdm=4.56.0
  * imageio

## Data prerequisites

  1. Raw count matrix without normalization, rows as samples and columns as genes
  2. First column as the coordinates of each sample with the format of 'XxY', eg. '12x23'
  3. csv format


## Parameters 
  ### I/O parameters
  * -f: path to the input raw count matrix (csv).
  * -o: path to save the imputed data sets.

  ### Parameters affecting imputation values
  * -r: radius in Euclidean distance to consider as adjacent spots.
  * -s: whether to select thresholding parameter epsilon automatically or not. 0: no selection, use fixed. 1: select automatically.
  * -e: edge filtering parameter epsilon, range from 0 to 1. Only useful when -s was set to 0.
  * -l: normalization method. Must be one of "cpm", "logCPM", "logMed", "none". Default is "cpm".

  ### Other parameters
  * -n: number of processors to be used for parallel computing.

## Run example experiments
  
  The following code will impute the test data with 4 processors, save the imputed cpm data, raw data to the designated folder. Also, the component information will be saved to the same folder.
  
    python3 spImpute.py -f test_data/raw.csv -o test_data/imputed.csv -l cpm -n 4

  ### Visualize major tissue components
  
  The following code will take component information returned by the imputation pipeline and visualize the component information.
  
    python3 visualize_components.py test_data/imputed_cluster_info.csv test_data/cluster.png
    

# Missing value Imputation for Spatially resolved Transcriptomics (MIST)

We have moved our MIST package inside a more general object - ReST, which stands for Region-based
algorithms for Spatial Transcriptomics analysis. Please expect more contents to be added to ReST in the future.

## Required dependencies

  * pandas=0.25.3
  * numpy=1.18.5
  * matplotlib=3.3.4
  * statsmodels=0.12.0
  * scipy=1.6.1
  * tqdm=4.56.0
  * imageio
  * alphashape
  * descartes
  * joblib
  * gseapy

## Installation

  We recommended to create a conda environment with above-listed dependencies installed.

## Input data format

  1. For 10X Visium, Space Ranger `Folder` with the following contents:
    - [Folder]/spatial/tissue_positions_list.csv
    - [Folder]/filtered_feature_bc_matrix.h5
  2. `adata`: processed AnnData object with count, spot and gene meta data frame
  3. General spatial transcriptomics data
    - counts - gene expression data frame in Pandas.DataFrame format.
	  - coordinates - spot meta data frame, with x, y columns denoting coordinates.
	  - gene_df - gene meta data frame.

## Running MIST
  Please read `Tutorial 1 - MIST region detection, functional annotation and imputation (Melanoma).ipynb` for instructions.

<!-- ## Parameters 
  ### I/O parameters
  * -f: path to the input raw count matrix (csv).
  * -o: path to save the imputed data sets.

  ### Model Parameters
  * -r: radius in Euclidean distance to consider as adjacent spots.
  * -s: whether to select thresholding parameter epsilon automatically or not. 0: no selection, use fixed. 1: select automatically.
  * -e: edge filtering parameter epsilon, range from 0 to 1. Only useful when -s was set to 0.
  * -l: normalization method. Must be one of "cpm", "logCPM", "logMed", "none". Default is "cpm".

  ### Other parameters
  * -n: number of processors to be used for parallel computing. 1-10. Default is 1. 

## Run example experiments
  
  The following code will impute the test data with 4 processors, save the imputed cpm data, raw data to the designated folder. Also, the component information will be saved to the same folder.
  
    python3 MIST.py -f test_data/raw.csv -o test_data/imputed.csv -l cpm -n 4

  After running the above code, the following files will be generated:

    1. test_data/imputed.csv -- imputed, normalized, gene filtered expression.
    2. test_data/imputed_complete.csv -- imputed, normalized, gene expression.
    3. test_data/imputed_rawCount.csv -- imputed, raw gene counts.
    4. imputed_cluster_info.csv -- region assignment of every spot.

  ### Visualize major tissue components
  
  The following code will take component information returned by the imputation pipeline and visualize the component information.
  
    python3 visualize_components.py test_data/imputed_cluster_info.csv test_data/cluster.png
  
  The above code will visualize the detected regions by giving a figure like:

  [Cluster Visualization](test_data/output/cluster.png) -->

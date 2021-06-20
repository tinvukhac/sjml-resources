# Towards a Learned Cost Model for Distributed Spatial Join: Data, Code & Models

This is the end-point for the resources of a paper submitted to the 30th ACM International Conference on Information and Knowledge 
Management (CIKM 2021).

## Datasets and corresponding section in the paper
1. [Join input datasets](https://drive.google.com/drive/folders/1BT1UsrvG1MB1bWCVYDWk6XLOZFIOLcj0?usp=sharing) 

    1.1. [Synthetic datasets](https://drive.google.com/drive/folders/1_EoXOrBrJYFIVGXCnNRifSXQecJTjNC5?usp=sharing) (Section 3.1)

    1.2. [Real datasets](https://drive.google.com/drive/folders/1wY9F3p4qOdvxkjXsIl2GGHTk_lJYiRds?usp=sharing) (Section 3.2)

2. [Spatial join execution datasets](https://drive.google.com/drive/folders/1ITSpPZZGFwP7qqBIcOvctX2SC74sFIVu?usp=sharing) (Section 4.2)

3. [Datasets for spatial join cost models](https://drive.google.com/drive/folders/196Sj0JizSCYNrnpyR2AL9TRxvEg2ioNe?usp=sharing) (Section 4.3)

## Spatial data generators
1. Generate data using Spider
2. Generate data using open-source Python program
3. Generate data using Spark API

## Train and test spatial join cost estimation models
### Required Environment

* We recommend to use [PyCharm](https://www.jetbrains.com/pycharm/download/) as the IDE. 
But you could use other IDEs(e.g. IntelliJ) or any other code editors.
* In order to make it easier for you to install all required libraries (Keras, TensorFlow, scikit-learn, pandas, etc), 
we would recommend you to install [Anaconda](https://docs.anaconda.com/anaconda/install/). In particular, you could use an environment which is identical with ours as the following steps:
1. [Install Anaconda](https://docs.continuum.io/anaconda/install/)
2. Add conda to your $PATH variable: /home/your_username/anaconda3/condabin
3. Move to the project directory: cd */deep-spatial-join
4. Follow this tutorial to create an environment from our environment.yml file: [Creating an environment from an environment.yml file](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)
5. Activate the environment. Now you are ready to play with the model!  

### Brief description of the source code
* main.py: the endpoint to run the program.
* datasets.py: data pre-processing module
* data/histograms: contains csv files, which are the histograms of input datasets (to be fed into the CNN layers).
* data/tabular: contains csv files, which are the tabular feature of the input datasets (to be fed into the MLP layer).
* data/join_results: contains csv files which are the results of spatial join queries. Columns: dataset 1, dataset 2, join result size, # of MBR test, execution time.
* trained_models: where you save the trained models.
* utils: a bunch of scripts that we use to clean/fix data problems. You do not need to pay much of attention to these scripts.

### Train and test proposed models
1. Join selectivity estimation model
2. MBR tests selectivity estimation model
3. Algorithm selection model


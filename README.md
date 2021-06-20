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
Research on the spatial data generators were published at [Spatial Gems 2019](https://www.spatialgems.net/spatial-gems-collection) 
and [SIGSPATIAL 2020](https://sigspatial2020.sigspatial.org/accepted-papers/).
### 1. Generate data using Spider

Please go to [Spider Web](https://spider.cs.ucr.edu/) to generate and visualize your spatial datasets.

### 2. Generate data using [open-source Python program](https://github.com/tinvukhac/spatialdatagenerators). 
We also already added the program 'generator.py' to this repository. It should be executed with Python 3.

```shell script
# Show the instructions to use the program
python3 generator.py -h
# Generate a sample of 100 data points with uniform distribution
python3 generator.py --dist uniform --card 100 --dim 2 --geo point --output uniform_sample --format csv
```

### 3. Generate data using Spark API
We use [Beast](https://bitbucket.org/eldawy/beast/src/master/), our open-source system for Big Exploratory Analytics for Spatio-temporal data, 
to generate large spatial datasets using Spark API. To run a program built on top of Beast, Spark 3.0 and HDFS 3.2 are required in your machine.

First, you should create a maven project and add Beast as a maven dependency. The detailed instructions can be found [here](https://bitbucket.org/eldawy/beast/src/master/).
We also provide [a template project](https://bitbucket.org/eldawy/beast-examples/src/master/) that was already configured with required dependencies.

Once you have an Scala example that can use Beast's functionalities, you can easily generate a spatial dataset using the 
following commands:
```scala
import edu.ucr.cs.bdlab.beast._
import edu.ucr.cs.bdlab.beast.generator._
val generatedData: SpatialRDD = sparkContext.generateSpatialData(UniformDistribution, 100, 
  opts = Seq(SpatialGenerator.Dimensions -> 2, PointBasedGenerator.GeometryType -> "point"))
``` 

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
5. Activate the environment. Now you are ready to play with the models!  

### Brief description of the source code
* main.py: the endpoint to run the program.
* regression_model.py: implementation of regression models to estimate join selectivity and MBR tests selectivity.
* classification_model.py: implementation of classification models to estimate the best join algorithm in terms of running time.
* datasets.py: data pre-processing module
* data/histograms: contains csv files, which are the histograms of input datasets.
* data/tabular: contains csv files, which are the tabular feature of the input datasets (to be fed into the MLP layer).
* data/join_results: contains csv files which are the results of spatial join queries. Columns: dataset 1, dataset 2, join result size, # of MBR test, execution time.
* trained_models: where you save the trained models.
* utils: a bunch of scripts that we use to clean/fix data problems. You do not need to pay much of attention to these scripts.

### Train and test proposed models
1. Join selectivity estimation model
```shell script
python main.py --model random_forest --tab data/train_and_test_all_features_split/train_join_results_small_x_small.csv --path trained_models/model_join_selectivity.h5 --weights trained_models/model_weights_join_selectivity.h5 --target join_selectivity --train
python main.py --model random_forest --tab data/train_and_test_all_features_split/test_join_results_small_x_small.csv --path trained_models/model_join_selectivity.h5 --weights trained_models/model_weights_join_selectivity.h5 --target join_selectivity --no-train
```

2. MBR tests selectivity estimation model
```shell script
python main.py --model random_forest --tab data/train_and_test_all_features_split/train_join_results_small_x_small.csv --path trained_models/model_mbr_tests_selectivity.h5 --weights trained_models/model_weights_mbr_tests_selectivity.h5 --target mbr_tests_selectivity --train
python main.py --model random_forest --tab data/train_and_test_all_features_split/test_join_results_small_x_small.csv --path trained_models/model_mbr_tests_selectivity.h5 --weights trained_models/model_weights_mbr_tests_selectivity.h5 --target mbr_tests_selectivity --no-train
```

3. Algorithm selection model
```shell script
python main.py --model clf_random_forest --tab data/train_and_test_all_features_split/train_join_results_combined_v3.csv --path trained_models/model_best_algorithm.h5 --weights trained_models/model_weights_best_algorithm.h5 --target best_algorithm --train
python main.py --model clf_random_forest --tab data/train_and_test_all_features_split/test_join_results_combined_v3.csv --path trained_models/model_best_algorithm.h5 --weights trained_models/model_weights_best_algorithm.h5 --target best_algorithm --no-train

```


# Documentation

## Overview
This repo contains team **Gradient Surfers'** (*Lucas Trognon, Harold Benoit, Mahammad Ismayilzada*) code and data for the ML Project 1. Project description can be found in the `project_description.pdf` file.

## Repo Structure
This repo is organized as following:
* `run.py` - This script runs the ML Pipeline to produce exactly the same _.csv_ predictions used in our best submission.
* `implementations.py` - This file contains from-scratch numpy implementations of the ML algorithms used in this project.
* `helpers.py` - This file contains the code for various helper functions to complement ML algorithms and perform exploratory analysis, preprocessing, feature engineering and submission steps.
* `project1.ipynb` - This notebook contains the code for a step-by-step execution of the whole ML pipeline from loading data to feature engineering to model selection to preparation of the submission file.
* **`notebooks`** - This folder contains separate notebooks for different parts of the pipeline like EDA, model selection and model execution.
* **`submissions`** - This folder contains past submissions made to the AICrowd platform.
* **`meta`** - This folder contains the meeting notes of the team.
* **`data`** - This folder contains the training and test data for the project (Due to the large sizes of the training and test data, this folder does not exist on Github, but the code expects to find the data here)
* **`report`** - This folder contains files for the project report.

## Environment
This repo requires `Python v3.5+` and `numpy v1.20+` and few visualization libraries (`matplotlib`, `seaborn`) in order to run the `run.py` and the notebooks.

## Project

### Outline
This project is based on the popular [ML challenge](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs) to find the Higgs boson using original data from CERN. We are given a training and test data that contains the results of the CERN LHC's experiments on smashing protons to detect the Higgs boson. The purpose is to predict the presence of the Higgs boson given the data collected from various sensors used in the experiment. For this project, we have gone through a typical ML pipeline and implemented several regression and classification algorithms from scratch in python by only using numpy. Below we will describe the steps taken in more detail.

### Exploratory Data Analysis
At this step, we familiarize ourselves with the data by looking at various statistics and plots generated from the data. First of all, we have about **250,000** observations for the training and **568,238** events for the test data. Observations have been represented with 30 features where all variables are floating point, except `PRI_jet_num` which is integer. Target variable is encoded as ***(s, b)*** where ***s*** indicates a presence of a Higgs signal while ***b*** means we have just observed a background noise of other particles. A quick look at the distribution plots of the features reveal that some variables share one frequently occuring value **(-999)**. which are as per project description, values that are either meaningless or could not be computed for the given experiment. However, further investigation shows that these values are systematically missing and in fact we can see the correlation between missing values and the number of jets. This is also confirmed by the descriptions of the features in the original paper. This suggests that number of jets really plays a discriminatory role and splitting data based on these values and removing unnecessary columns for each subset of the dataset might prove to perform better. For more information on these results, please check out the `notebooks/exploration.ipynb`. 

### Preprocessing and Feature Engineering
First of all, we decided to treat the (-999) values as NaNs to make the processing easier. Then we computed the ratio of the NaN values for each feature. Results showed that there are some features containing more than 70% NaNs, some around 40% and some around 15%. Initially, we decided drop the columns in the first group (since large proportion of them are undefined, they are unlikely to provide a useful insight), encode the columns into binary features in the second group (these features could be indicative of the signal presence) and impute the features in the last group with the median values (since we are missing very few values) Although this preprocessing improved the model performance a lot, a revelation about the correlation between the number of jets and the rest of the features suggested a better approach. We split the dataset into 3 datasets based on the number of jets (`PRI_jet_num`) where each subset contained the observations with 0, 1 and more than 1 jet respectively. Then based on the original paper we identified which features were missing for each subset and we removed them accordingly from each subset in addition to the `PRI_jet_num` feature which was now explicitly encoded. As a result of this preprocessing, we get 3 subsets of the data that now contain -999 values only in the first column. This column can be either imputed or encoded or both. Then in order to normalize skewed distributions we apply log transformation to positive features, to make the comparison of features easier with different units, we standardize continuous features, add a bias column and lastly remove all outliers. For more information on these results, please check out the `notebooks/feature_engineering.ipynb`.

### Model Selection and Hyperparameter Tuning
As our problem is a classification problem, `logistic regression` is the most likely candidate to be the best performing model for this task among the methods allowed for this project, but we wanted to empirically show this using model selection techniques. We used K-fold (5 folds) cross validation to determine the best performing model and Grid Search to tune the hyperparameters such as the polynomial *degree* for feature expansion, learning rate (*gamma*) and the regularization parameter *lambda*. Logistic regression indeed proved to be the best model according to our results. Additionally, we ran multiple models depending on the preprocessing level of the input data from the simple raw data to the complex feature engineered data. For more information on these results, please check out the notebooks under the name of different ML algorithms in the **notebooks** folder.


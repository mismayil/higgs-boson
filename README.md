# Documentation

## Overview
This repo contains team **Gradient Surfers'** (Lucas Trognon, Harold Benoit, Mahammad Ismayilzada) code and data for the ML Project 1. Project description can be found in the `project_description.pdf` file.

## Repo structure
This repo is organized as following:
* `run.py` - This script runs the ML Pipeline to produce exactly the same _.csv_ predictions used in our best submission.
* `implementations.py` - This file contains from-scratch numpy implementations of the ML algorithms used in this project.
* `helpers.py` - This file contains the code for various helper functions to complement ML algorithms and perform exploratory analysis, preprocessing, feature engineering and submission steps.
* `project1.ipynb` - This notebook contains the code for a step-by-step execution of the whole ML pipeline from loading data to feature engineering to model selection to preparation of the submission file.
* **`notebooks`** - This folder contains separate notebooks for different parts of the pipeline like EDA, model selection and model execution.
* **`submissions`** - This folder contains past submissions made to the AICrowd platform.
* **`meta`** - This folder contains the meeting notes of the team.
* **`data`** - This folder contains the training and test data for the project.
* **`report`** - This folder contains files for the project report.

## Project

### Outline
This project is based on the popular [ML challenge](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs) to find the Higgs boson using original data from CERN. We are given a training and test data that contains the results of the CERN LHC's experiments on smashing protons to detect the Higgs boson. The purpose is to predict the presence of the Higgs boson given the data collected from various sensors used in the experiment. For this project, we have gone through a typical ML pipeline and implemented several regression and classification algorithms from scratch in python by only using numpy. Below we will describe the steps taken in more detail.

### Exploratory Data Analysis
At this step, we familiarize ourselves with the data by looking at various statistics and plots generated from the data. First of all, we have about **250,000** observations for the training and **568,238** events for the test data. Observations have been represented with 30 features where all variables are floating point, except `PRI_jet_num` which is integer. Target variable is encoded as ***(s, b)*** where ***s*** indicates a presence of a Higgs signal while ***b*** means we have just observed a background noise of other particles. A quick look at the distribution plots of the features reveal that some variables share one frequently occuring value **(-999)**. which are as per project description, values that are either meaningless or could not be computed for the given experiment.

### Preprocessing and Feature engineering
First of all, we decided to treat the (-999) values as NaNs to make the processing easier. Then we computed the ratio of the NaN values for each feature. Results showed that there are some features containing more than 70% NaNs, some around 40% and some around 15%. We decided drop the columns in the first group (since large proportion of them are undefined, they are unlikely to provide a useful insight), encode the columns into binary features in the second group (these features could be indicative of the signal presence) and impute the features in the last group with the median values (since we are missing very few values) Then in order to minimize the effect of the outliers and make the comparison of features easier, we standardize the continuous features and lastly add a bias column.
### Model selection and hyperparameter tuning
As our problem is a classification problem, `logistic regression` is the most likely candidate to be the best performing model for this task among the methods allowed for this project, but we wanted to empirically show this using model selection techniques. We used K-fold cross validation to determine the best performing model and Grid Search to tune the hyperparameters. Logistic regression indeed proved to be the best model according to our results.


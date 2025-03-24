REQUIRED PACKAGES
- pyspark-dist-explore (pip install pyspark-dist-explore)

Files:
my_functions.py
It contains some custom functions used in the computation.

data_understanding.py
Compute statistics and distribution of the variables for the weather, building and joined datasets

data_preparation.py
Create the final dataset (cleaned_data.csv) joining the three datasets and cleaning the data. The code produce also a small version of the final dataset (cleaned_data_small.csv) containing a few thousands of records just for developing purposes

preprocessing.py
Compute one hot encoding for categorical attributes and split the data in training (full_train.json) and test (test.json) set starting from the cleaned dataset (clean_data.csv)

[model]_validation.py
Starting from the train data (full_train.json) split the data in training and validation set and tune the hyperparameters

[model]_test.py
Test the validated model training the model on train data (full_train.json) and testing it on unseen test data (test.json)

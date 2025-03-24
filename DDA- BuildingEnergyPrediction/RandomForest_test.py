from pyspark import SparkContext
from pyspark.sql import *
from pyspark.mllib.tree import RandomForest
import matplotlib.pyplot as plt
from math import log
import pandas as pd
import os
import time
import pickle
from pyspark_dist_explore import distplot
from my_functions import from_json_to_labeled_rdd, get_log_time

#######################################################################################
###########################  SPARK SESSION AND LOG FILE  ##############################

#create log file
current_time = time.localtime()
log_file_name = 'RandomForest_test_'+get_log_time()[:10].replace('/','')+'.log'
log_file_name = os.path.join('log', log_file_name)
lf = open(log_file_name, 'w')

#set spark context and spark session
sc = SparkContext()
spark = SparkSession.builder.appName("RandomForest_test").getOrCreate()
lf.write(get_log_time()+': SparkContext: {}\nSparkSession: {}'.format(sc, spark))

#######################################################################################


#######################################################################################
#################################  READING DATASETS ###################################

lf.write('\n\n')
lf.write(get_log_time()+': Reading datasets')
file_name = os.path.join('data','full_train_trees.json')
train = from_json_to_labeled_rdd(spark, file_name, n_features=13, sparse=False)
file_name = os.path.join('data','test_trees.json')
test = from_json_to_labeled_rdd(spark, file_name, n_features=13, sparse=False)

#######################################################################################


#######################################################################################
####################################  TEST MODEL ######################################

#loading categorical features mapping and computing categorical features info
lf.write('\n\n')
lf.write(get_log_time()+': Loading categorical features mapping and computing categorical features info')
with open('categ_mapping', 'rb') as file:
    categ_mapping  = pickle.load(file)

categ_feat_info = {ind:len(val[1]) for ind, val in categ_mapping.items()}

#fit the model
lf.write('\n\n')
lf.write(get_log_time()+': Fitting the model with maxDepth=22')
model = RandomForest.trainRegressor(train, categoricalFeaturesInfo=categ_feat_info, numTrees=4, maxDepth=22)

#extract predictions on test set
lf.write('\n\n')
lf.write(get_log_time()+': Predicting values on test sets')
#set to zero the negative predicted values
preds = model.predict(test.map(lambda x: x.features)).map(lambda x: x if x>0 else 0)
#extract labels
actuals = test.map(lambda x: x.label)
#merge actuals values and predicted ones
lf.write('\n\n')
lf.write(get_log_time()+': Merging actuals and predicted values into an RDD')
actuals_preds = actuals.zipWithIndex().map(lambda x: (x[1],x[0])).join(preds.zipWithIndex().map(lambda x: (x[1],x[0]))).map(lambda x: x[1])
#compute the test error
lf.write('\n\n')
lf.write(get_log_time()+': Computing the root mean square logarithmic error')
rmslq = actuals_preds.map(lambda x: (log(x[0]+1)-log(x[1]+1))**2).reduce(lambda x,y: x+y)/actuals_preds.count()
lf.write('\n\n')
lf.write(get_log_time()+': RMSLE: {}'.format(rmslq))

#compute residuals and save them
lf.write('\n\n')
lf.write(get_log_time()+': computing residuals and save them')
row = Row("residuals")
residuals = actuals_preds.map(lambda x: x[1] - x[0]).map(row).toDF()
save_table(residuals, os.path.join('results','residuals_rdd'))

#######################################################################################

##### close log file
lf.write('\n\n')
lf.write(get_log_time()+': SUCCESS\n')
lf.close()

spark.stop()
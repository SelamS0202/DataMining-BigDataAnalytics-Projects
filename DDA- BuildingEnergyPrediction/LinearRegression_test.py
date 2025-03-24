from pyspark import SparkContext
from pyspark.sql import *
from pyspark.mllib.regression import LinearRegressionWithSGD
import matplotlib.pyplot as plt
from math import log
import pandas as pd
import os
import time
import pickle
from my_functions import from_json_to_labeled_rdd, get_log_time

#######################################################################################
###########################  SPARK SESSION AND LOG FILE  ##############################

#create log file
current_time = time.localtime()
log_file_name = 'LinearRegression_test_'+get_log_time()[:10].replace('/','')+'.log'
log_file_name = os.path.join('log', log_file_name)
lf = open(log_file_name, 'w')

#set spark context and spark session
sc = SparkContext()
spark = SparkSession.builder.appName("LinearRegression_test").getOrCreate()
lf.write(get_log_time()+': SparkContext: {}\nSparkSession: {}'.format(sc, spark))

#######################################################################################


#######################################################################################
#################################  READING DATASETS ###################################

lf.write('\n\n')
lf.write(get_log_time()+': Reading datasets')
file_name = os.path.join('data','full_train.json')
train = from_json_to_labeled_rdd(spark, file_name, n_features=32, sparse=True)
file_name = os.path.join('data','test.json')
test = from_json_to_labeled_rdd(spark, file_name, n_features=32, sparse=True)

#######################################################################################


#######################################################################################
####################################  TEST MODEL ######################################

#fit the model
lf.write('\n\n')
lf.write(get_log_time()+': Fitting the model with regularization {} with value {}'.format('l1', 0.1))
model = LinearRegressionWithSGD.train(train, regParam=0.1, regType='l1', intercept=True)

#predicting values and merging with actual ones
lf.write('\n\n')
lf.write(get_log_time()+': Predicting values of test set and merging with actual ones')
actuals_preds = test.map(lambda x: (x.label, model.predict(x.features))).map(lambda x: x if x[1]>0 else (x[0],0.0))
#compute the test error
lf.write('\n\n')
lf.write(get_log_time()+': Computing the root mean square logarithmic error')
rmslq = actuals_preds.map(lambda x: (log(x[0]+1)-log(x[1]+1))**2).reduce(lambda x,y: x+y)/actuals_preds.count()
lf.write('\n\n')
lf.write(get_log_time()+': RMSLE: {}'.format(rmslq))

#######################################################################################

##### close log file
lf.write('\n\n')
lf.write(get_log_time()+': SUCCESS\n')
lf.close()

spark.stop()
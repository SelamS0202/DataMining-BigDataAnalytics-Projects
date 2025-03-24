from pyspark import SparkContext
from pyspark.sql import *
from pyspark.mllib.regression import LinearRegressionWithSGD
import matplotlib.pyplot as plt
from math import log
import pandas as pd
import numpy as np
import os
import time
import pickle
from itertools import product
from my_functions import from_json_to_labeled_rdd, get_log_time


#######################################################################################
###########################  SPARK SESSION AND LOG FILE  ##############################

#create log file
current_time = time.localtime()
log_file_name = 'LinearRegression_validation_'+get_log_time()[:10].replace('/','')+'.log'
log_file_name = os.path.join('log', log_file_name)
lf = open(log_file_name, 'w')

#set spark context and spark session
sc = SparkContext()
spark = SparkSession.builder.appName("LinearRegression_validation").getOrCreate()
lf.write(get_log_time()+': SparkContext: {}\nSparkSession: {}'.format(sc, spark))

#######################################################################################


#######################################################################################
#################################  READING DATASETS ###################################

lf.write('\n\n')
lf.write(get_log_time()+': Reading datasets')
file_name = os.path.join('data','full_train.json')
data = from_json_to_labeled_rdd(spark, file_name, n_features=32, sparse=True)
train, validation = data.randomSplit([0.8, 0.2])
n_train = train.count()
n_validation = validation.count()

#######################################################################################


#######################################################################################
########################## VALIDATE LINEAR REGRESSION ##################################

lf.write('\n\n')
lf.write(get_log_time()+': Tuning hyperparameter depth of the trees')

####tune the hyperparameter regularization
reg_types = ['l1', 'l2', None]
reg_values = [10**-i for i in range(1,5)]
no_reg_flag = 0
min_rmslq_validation = 10**6 #big value

for hp in product(reg_types, reg_values):
    #do the iteration with reg_value=0.0 only with regularization L1 (with L2 would be the same since the reg value is 0)
    if hp[0] is None:
        if no_reg_flag == 1:
            continue
        else:
            no_reg_flag = 1
    
    lf.write('\n\n')
    if hp[0] is None:
        lf.write(get_log_time()+': Fitting the model with no regularization')
    else:
        lf.write(get_log_time()+': Fitting the model with regularization {} with value {}'.format(hp[0], hp[1]))
    model = LinearRegressionWithSGD.train(train, regParam=hp[1], regType=hp[0], intercept=True)

    #extract prediction on training and validation set
    lf.write('\n\n')
    lf.write(get_log_time()+': Predicting values and merging with actual ones on training and validation sets')
    #set to zero the negative predicted values
    actuals_preds_train = train.map(lambda x: (x.label, model.predict(x.features))).map(lambda x: x if x[1]>0 else (x[0],0.0))
    actuals_preds_validation = validation.map(lambda x: (x.label, model.predict(x.features))).map(lambda x: x if x[1]>0 else (x[0],0.0))
    #compute the training and validation root mean square logarithmic error
    lf.write('\n\n')
    lf.write(get_log_time()+': Computing the root mean square logarithmic error')
    rmslq_train = actuals_preds_train.map(lambda x: (log(x[0]+1)-log(x[1]+1))**2).reduce(lambda x,y: x+y)/n_train
    rmslq_validation = actuals_preds_validation.map(lambda x: (log(x[0]+1)-log(x[1]+1))**2).reduce(lambda x,y: x+y)/n_validation
    lf.write('\n\n')
    lf.write(get_log_time()+': RMSLE train: {}\t RMSLE validation: {}'.format(rmslq_train, rmslq_validation))

    #get best combination of hyperparameters
    if rmslq_validation < min_rmslq_validation:
        min_rmslq_validation = rmslq_validation
        best_hp = hp


lf.write('\n\n')
if best_hp[0] is None:
    lf.write(get_log_time()+': Best model: no regularization')
else:
    lf.write(get_log_time()+': Best model: regularization {} with value {}'.format(best_hp[0], best_hp[1]))

#######################################################################################

##### close log file
lf.write('\n\n')
lf.write(get_log_time()+': SUCCESS\n')
lf.close()

spark.stop()
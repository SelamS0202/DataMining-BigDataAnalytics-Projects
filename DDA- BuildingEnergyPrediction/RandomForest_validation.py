from pyspark import SparkContext
from pyspark.sql import *
from pyspark.mllib.tree import RandomForest
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
log_file_name = 'RandomForest_validation_'+get_log_time()[:10].replace('/','')+'.log'
log_file_name = os.path.join('log', log_file_name)
lf = open(log_file_name, 'w')

#set spark context and spark session
sc = SparkContext()
spark = SparkSession.builder.appName("RandomForest_validation").getOrCreate()
lf.write(get_log_time()+': SparkContext: {}\nSparkSession: {}'.format(sc, spark))

#######################################################################################


#######################################################################################
#################################  READING DATASETS ###################################

lf.write('\n\n')
lf.write(get_log_time()+': Reading datasets')
file_name = os.path.join('data','full_train_trees.json')
data = from_json_to_labeled_rdd(spark, file_name, n_features=13, sparse=False)
train, validation = data.randomSplit([0.8, 0.2])

#extract labels of training and validation set
actuals_train = train.map(lambda x: x.label)
actuals_validation = validation.map(lambda x: x.label)

#count values
lf.write('\n\n')
lf.write(get_log_time()+': Counting istances of training and validation set')
n_train = actuals_train.count()
n_validation = actuals_validation.count()
lf.write('\n\n')
lf.write(get_log_time()+': There are {} istances of training and {} istances of validation set'.format(n_train, n_validation))

#set indexes for merging with predictions
lf.write('\n\n')
lf.write(get_log_time()+': Construct rdd with indexes for actual values')
actuals_train_with_index = actuals_train.zipWithIndex().map(lambda x: (x[1],x[0]))
actuals_validation_with_index = actuals_validation.zipWithIndex().map(lambda x: (x[1],x[0]))

#######################################################################################


#######################################################################################
########################## VALIDATE RANDOM FOREST #####################################

#loading categorical features mapping and computing categorical features info
lf.write('\n\n')
lf.write(get_log_time()+': Loading categorical features mapping and computing categorical features info')
with open('categ_mapping', 'rb') as file:
    categ_mapping  = pickle.load(file)

categ_feat_info = {ind:len(val[1]) for ind, val in categ_mapping.items()}

#create list for training and validation root mean squared logarithmic errors
rmslq_values_train = list()
rmslq_values_validation = list()

####tune the hyperparameter depth of the trees
depth_vals = range(15,22)

lf.write('\n\n')
lf.write(get_log_time()+': Tuning hyperparameter depth of the trees from {} to {}'.format(depth_vals[0], depth_vals[-1]))


for n in depth_vals:
    lf.write('\n\n')
    lf.write(get_log_time()+': Fitting the model with depth '+str(n))
    try:
        model = RandomForest.trainRegressor(train, categoricalFeaturesInfo=categ_feat_info, numTrees=4, maxDepth=n)
    except:
        lf.write('\n\n')
        lf.write(get_log_time()+': Model too complex, there is not enough memory! Stop iterations')
        break

    #extract prediction on training and validation set
    lf.write('\n\n')
    lf.write(get_log_time()+': Predicting values on training and validation sets')
    #set to zero the negative predicted values
    preds_train = model.predict(train.map(lambda x: x.features)).map(lambda x: x if x>0 else 0)
    preds_validation = model.predict(validation.map(lambda x: x.features)).map(lambda x: x if x>0 else 0)

    #set indexes for merging with actuals
    lf.write('\n\n')
    lf.write(get_log_time()+': Construct rdd with indexes for predicted values')
    preds_train_with_index = preds_train.zipWithIndex().map(lambda x: (x[1],x[0]))
    preds_validation_with_index = preds_validation.zipWithIndex().map(lambda x: (x[1],x[0]))

    #merge actuals values and predicted ones
    lf.write('\n\n')
    lf.write(get_log_time()+': Merging actuals and predicted values into an RDD')
    actuals_preds_train = actuals_train_with_index.join(preds_train_with_index).map(lambda x: x[1])
    actuals_preds_validation = actuals_validation_with_index.join(preds_validation_with_index).map(lambda x: x[1])

    #compute the training and validation root mean square logarithmic error
    lf.write('\n\n')
    lf.write(get_log_time()+': Computing the root mean square logarithmic error')
    rmslq_train = actuals_preds_train.map(lambda x: (log(x[0]+1)-log(x[1]+1))**2).reduce(lambda x,y: x+y)/n_train
    rmslq_validation = actuals_preds_validation.map(lambda x: (log(x[0]+1)-log(x[1]+1))**2).reduce(lambda x,y: x+y)/n_validation
    lf.write('\n\n')
    lf.write(get_log_time()+': RMSLE train: {}\t RMSLE validation: {}'.format(rmslq_train, rmslq_validation))
    rmslq_values_train.append(rmslq_train)
    rmslq_values_validation.append(rmslq_validation)


#sava results
lf.write('\n\n')
lf.write(get_log_time()+': Saving the results')
with open(os.path.join('results', 'RMSLE_train_RandomForest.list'), 'wb') as file:
    pickle.dump(rmslq_values_train, file)
with open(os.path.join('results', 'RMSLE_validation_RandomForest.list'), 'wb') as file:
    pickle.dump(rmslq_values_validation, file)

#plot the results
lf.write('\n\n')
lf.write(get_log_time()+': Plotting the results')
try:
    plt.plot(depth_vals, rmslq_values_train, label='Training set', color='b')
    plt.plot(depth_vals, rmslq_values_validation, label='Validation set', color='r')
    plt.title('RMSLE Decision Tree')
    plt.xlabel('Max depth of the tree')
    plt.legend()
    file_name = os.path.join('plots', 'RandomForest_validation')
    plt.savefig(file_name)
except:
    lf.write('\n\n')
    lf.write(get_log_time()+': Can not plot the results because memory error')

#######################################################################################

##### close log file
lf.write('\n\n')
lf.write(get_log_time()+': SUCCESS\n')
lf.close()

spark.stop()
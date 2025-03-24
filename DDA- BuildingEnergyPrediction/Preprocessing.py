from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.mllib.regression import  LabeledPoint
from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import MinMaxScaler
from pyspark.mllib.tree import RandomForest, RandomForestModel
import time
import os
import pickle
from my_functions import save_table, get_log_time, as_mllib

####################################################################################
########################  SPARK SESSION AND LOG FILE  ##############################

#create log file
current_time = time.localtime()
log_file_name = 'preprocessing_'+get_log_time()[:10].replace('/','')+'.log'
log_file_name = os.path.join('log', log_file_name)
lf = open(log_file_name, 'w')

#set spark context and spark session
sc = SparkContext()
spark = SparkSession.builder.appName("preprocessing").getOrCreate()
lf.write(get_log_time()+': SparkContext: {}\nSparkSession: {}'.format(sc, spark))
####################################################################################

####################################################################################
##############################  READING DATASETS ###################################

lf.write('\n\n')
lf.write(get_log_time()+': Reading datasets')
#read the datasets
df = spark.read.csv(os.path.join('data','cleaned_data.csv'), header=True)

##### cast columns
lf.write('\n\n')
lf.write(get_log_time()+': Casting columns')

df = df.withColumn('site_id', df['site_id'].cast(IntegerType()))
df = df.withColumn('timestamp', df['timestamp'].cast(TimestampType()))
df = df.withColumn('wind_speed', df['wind_speed'].cast(FloatType()))
df = df.withColumn('wind_direction', df['wind_direction'].cast(FloatType()))
df = df.withColumn('sea_level_pressure', df['sea_level_pressure'].cast(FloatType()))
#df = df.withColumn('precip_depth_1_hr', df['precip_depth_1_hr'].cast(FloatType()))
#df = df.withColumn('cloud_coverage', df['cloud_coverage'].cast(FloatType()))
df = df.withColumn('air_temperature', df['air_temperature'].cast(FloatType()))
df = df.withColumn('dew_temperature', df['dew_temperature'].cast(FloatType()))
df = df.withColumn('site_id', df['site_id'].cast(IntegerType()))
df = df.withColumn('building_id', df['building_id'].cast(IntegerType()))
df = df.withColumn('square_feet', df['square_feet'].cast(FloatType()))
#df = df.withColumn('year_built', df['year_built'].cast(FloatType()))
#df = df.withColumn('floor_count', df['floor_count'].cast(FloatType()))
df = df.withColumn('building_id', df['building_id'].cast(IntegerType()))
#df = df.withColumn('meter', df['meter'].cast(IntegerType()))
df = df.withColumn('timestamp', df['timestamp'].cast(TimestampType()))
df = df.withColumn('meter_reading', df['meter_reading'].cast(FloatType()))
#df = df.withColumn('quarter', df['quarter'].cast(IntegerType()))
#df = df.withColumn('day_of_week', df['day_of_week'].cast(IntegerType()))
df = df.withColumn('hour', df['hour'].cast(IntegerType()))
df = df.withColumn('prev1week_meter_reading', df['prev1week_meter_reading'].cast(FloatType()))
df = df.withColumn('prev2week_meter_reading', df['prev2week_meter_reading'].cast(FloatType()))

#######################################################################################



#######################################################################################
##############################  PREPROCESSING #########################################

#define numeric and categorical attributes
categ_attrs = ['quarter','day_of_week','primary_use','meter']
numeric_attr = ['wind_speed','wind_direction','sea_level_pressure','air_temperature','dew_temperature','square_feet','hour','prev1week_meter_reading','prev2week_meter_reading']

#####preprocessing: defining stages for preprocessing data: one hot encoding for categorical attributes and standardization for numerical ones
lf.write('\n\n')
lf.write(get_log_time()+': Preprocessing data')
indexed_cols = [col+'_ind' for col in categ_attrs]
vectorized_cols = [col+'_vec' for col in categ_attrs]

#put all the numeric attributes into a vector in order to fit the standard scaler
numeric_assembler = VectorAssembler(inputCols=numeric_attr, outputCol="numeric_features")

#standard scaler with mean and standard deviation
scaler = MinMaxScaler(inputCol='numeric_features', outputCol='scaled_features', withMean=True)

#map categorical features that are strings to integer values because OneHotEncoderEstimator need integers
indexers = [StringIndexer(inputCol=col, outputCol=col+'_ind') for col in categ_attrs]
for indexer in indexers:
	indexer.setStringOrderType('alphabetAsc')

#one hot encoding for categorical features
encoder = OneHotEncoderEstimator(inputCols=indexed_cols, outputCols=vectorized_cols)

#put all the preprocessed features into a vector
assembler = VectorAssembler(inputCols=vectorized_cols+['scaled_features'], outputCol="features")
assembler_trees = VectorAssembler(inputCols=indexed_cols+numeric_attr, outputCol="features_trees")

#define the preprocessing pipeline that contains all the above defined steps
preproc_stages =  [numeric_assembler, scaler]+indexers+[encoder, assembler, assembler_trees]
preprocessing = Pipeline(stages = preproc_stages)

#fit the preprocessing
lf.write('\n\n')
lf.write(get_log_time()+': Fit the preprocessing')
preprocessingModel = preprocessing.fit(df)

#transform the data
lf.write('\n\n')
lf.write(get_log_time()+': Transforming data')
df = preprocessingModel.transform(df)

#select final dataset and split into training, validation and test set for trees and for other methods
lf.write('\n\n')
lf.write(get_log_time()+': Selecting final dataset and casting into RDD')
data = df.select('meter_reading','features').rdd.map(tuple)
data = data.map(lambda x: LabeledPoint(x[0], as_mllib(x[1])))
data_trees = df.select('meter_reading','features_trees').rdd.map(tuple)
data_trees = data_trees.map(lambda x: LabeledPoint(x[0], as_mllib(x[1])))

#get mapping for interpreting trees
categ_mapping = dict()
for i in range(2,6):
    categ_mapping[i-2] = (categ_attrs[i-2], dict(enumerate(preprocessingModel.stages[i].labels)))

with open('categ_mapping', 'wb') as file:
	pickle.dump(categ_mapping, file)

#######################################################################################


#######################################################################################
########################## SPLIT TRAINING AND TEST SET ################################

lf.write('\n\n')
lf.write(get_log_time()+': Splitting data into training, validation and test sets')
full_train, test = data.randomSplit([0.8, 0.2], seed=4)
full_train_trees, test_trees = data_trees.randomSplit([0.8, 0.2], seed=4)

#save datasets for test phase
lf.write('\n\n')
lf.write(get_log_time()+': Saving full training and test datasets for the testing phase')
full_train = full_train.toDF()
file_name = os.path.join('data','full_train')
save_table(full_train, file_name, save_as='json')
test = test.toDF()
file_name = os.path.join('data','test')
save_table(test, file_name, save_as='json')
full_train_trees = full_train_trees.toDF()
file_name = os.path.join('data','full_train_trees')
save_table(full_train_trees, file_name, save_as='json')
test_trees = test_trees.toDF()
file_name = os.path.join('data','test_trees')
save_table(test_trees, file_name, save_as='json')

#######################################################################################

##### close log file
lf.write('\n\n')
lf.write(get_log_time()+': SUCCESS\n')
lf.close()

spark.stop()

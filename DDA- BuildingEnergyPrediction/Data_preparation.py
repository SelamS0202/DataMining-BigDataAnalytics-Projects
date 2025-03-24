from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql.window import Window
import pandas as pd
import os
import time
from my_functions import save_table, get_log_time

####################################################################################
########################  SPARK SESSION AND LOG FILE  ##############################

#create log file
current_time = time.localtime()
log_file_name = 'data_preparation_'+get_log_time()[:10].replace('/','')+'.log'
log_file_name = os.path.join('log', log_file_name)
lf = open(log_file_name, 'w')

#set spark session
spark = SparkSession.builder.appName("data_preparation").getOrCreate()

####################################################################################


####################################################################################
##############################  READING DATASETS ###################################

lf.write(get_log_time()+': Reading datasets')
#read the datasets
df_weather = spark.read.csv(os.path.join('data','weather.csv'), header=True)
df_buildings = spark.read.csv(os.path.join('data','building.csv'), header=True)
df_energy = spark.read.csv(os.path.join('data','energy.csv'), header=True)

##### cast columns
#weather
df_weather = df_weather.withColumn('site_id', df_weather['site_id'].cast(IntegerType()))
df_weather = df_weather.withColumn('timestamp', df_weather['timestamp'].cast(TimestampType()))
df_weather = df_weather.withColumn('wind_speed', df_weather['wind_speed'].cast(FloatType()))
df_weather = df_weather.withColumn('wind_direction', df_weather['wind_direction'].cast(FloatType()))
df_weather = df_weather.withColumn('sea_level_pressure', df_weather['sea_level_pressure'].cast(FloatType()))
df_weather = df_weather.withColumn('precip_depth_1_hr', df_weather['precip_depth_1_hr'].cast(FloatType()))
df_weather = df_weather.withColumn('cloud_coverage', df_weather['cloud_coverage'].cast(FloatType()))
df_weather = df_weather.withColumn('air_temperature', df_weather['air_temperature'].cast(FloatType()))
df_weather = df_weather.withColumn('dew_temperature', df_weather['dew_temperature'].cast(FloatType()))

#buildings
df_buildings = df_buildings.withColumn('site_id', df_buildings['site_id'].cast(IntegerType()))
df_buildings = df_buildings.withColumn('building_id', df_buildings['building_id'].cast(IntegerType()))
df_buildings = df_buildings.withColumn('square_feet', df_buildings['square_feet'].cast(FloatType()))
df_buildings = df_buildings.withColumn('year_built', df_buildings['year_built'].cast(FloatType()))
df_buildings = df_buildings.withColumn('floor_count', df_buildings['floor_count'].cast(FloatType()))

#train
df_energy = df_energy.withColumn('building_id', df_energy['building_id'].cast(IntegerType()))
df_energy = df_energy.withColumn('meter', df_energy['meter'].cast(IntegerType()))
df_energy = df_energy.withColumn('timestamp', df_energy['timestamp'].cast(TimestampType()))
df_energy = df_energy.withColumn('meter_reading', df_energy['meter_reading'].cast(FloatType()))

####################################################################################

####################################################################################
############################  WEATHER DATASET  #####################################

##### Drop the two columns with too many missing values
lf.write('\n\n')
lf.write(get_log_time()+': Dropping columns with many missing values: ')
df_weather = df_weather.drop('cloud_coverage','precip_depth_1_hr')

#create date and time attributes
df_weather = df_weather.withColumn('quarter', quarter(df_weather['timestamp']))
df_weather = df_weather.withColumn('day_of_week', date_format(df_weather['timestamp'], 'EEEE'))
df_weather = df_weather.withColumn('hour', hour(df_weather['timestamp']))
df_weather = df_weather.withColumn('week', date_format(df_weather['timestamp'], 'W'))
df_weather = df_weather.withColumn('week', df_weather['week'].cast(IntegerType()))

####replace missing values
lf.write('\n\n')
lf.write(get_log_time()+': Replacing missing values')

#define windows on wich computing the average value
week_site_window = Window.partitionBy(['site_id','week']).rangeBetween(Window.unboundedPreceding, Window.unboundedFollowing)
week_site_hour_window = Window.partitionBy(['site_id','week','hour']).rangeBetween(Window.unboundedPreceding, Window.unboundedFollowing)

cont_cols = ['air_temperature', 'dew_temperature', 'sea_level_pressure', 'wind_direction', 'wind_speed']
for col in cont_cols:
    if col in ['air_temperature', 'dew_temperature']:
        my_window = week_site_hour_window
    else:
        my_window = week_site_window
        
    df_weather = df_weather.withColumn(col, coalesce(col,avg(df_weather[col]).over(my_window)))

#replace the remaining missing values with the total average
avg_sea_level = df_weather.select('sea_level_pressure').groupBy().avg().collect()[0][0]
df_weather = df_weather.na.fill({'sea_level_pressure':avg_sea_level})

####################################################################################

####################################################################################
############################  BUILDINGS DATASET  #####################################

##### Drop the two columns with too many missing values
lf.write('\n\n')
lf.write(get_log_time()+': Dropping columns with many missing values: ')
df_buildings = df_buildings.drop('year_built','floor_count')

####################################################################################

####################################################################################
############################  FINAL DATASET  #######################################

lf.write('\n\n')
lf.write(get_log_time()+': Building the final dataset joining train table with weather and building tables')
df = df_energy.join(df_buildings, 'building_id')
df = df.join(df_weather, ['timestamp','site_id'])

#Create new attributes
lf.write('\n\n')
lf.write(get_log_time()+': Removing records with missing values on the target variable and create new attributes')
my_window = Window.partitionBy(['day_of_week','hour']).orderBy(['building_id', 'meter', 'timestamp'])
df = df.withColumn('prev1week_meter_reading', lag(df['meter_reading'],1).over(my_window))
df = df.withColumn('prev2week_meter_reading', lag(df['meter_reading'],2).over(my_window))

#fill NaN with zeros and remove all the records with zeros in the meter actual reading or previous meter reading
df = df.na.fill({'prev1week_meter_reading':0, 'prev2week_meter_reading':0})
df = df.filter('meter_reading > 0 and prev2week_meter_reading > 0 and prev1week_meter_reading > 0')

#Check that there are no missing values
lf.write('\n\n')
lf.write(get_log_time()+': Checking that there are no missing values... ')
if [df.filter(isnull(col)).count() for col in df.columns] == [0 for col in df.columns]:
    lf.write('No missing values found')
else:
    lf.write('There are some missing values. Check the code!')

##### save the cleaned dataset
lf.write('\n\n')
lf.write(get_log_time()+': Saving the final dataset')
file_name = os.path.join('data', 'cleaned_data')
save_table(df, file_name)

##### save small dataset for developing purposes
df_small, _ = df.randomSplit([10**-3, 1-10**-3])
lf.write('\n\n')
lf.write(get_log_time()+': Saving a small dataset')
file_name = os.path.join('data', 'cleaned_data_small')
save_table(df_small, file_name)

##### close log file
lf.write('\n\n')
lf.write(get_log_time()+': SUCCESS\n')
lf.close()

spark.stop()
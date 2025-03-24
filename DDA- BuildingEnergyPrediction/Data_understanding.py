from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.sql.functions import *
import matplotlib.pyplot as plt
from pyspark_dist_explore import distplot
from matplotlib.ticker import FormatStrFormatter
from statsmodels.graphics.correlation import plot_corr
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller
import numpy as np
import pandas as pd
import os
import time
from my_functions import save_table, get_log_time



####################################################################################
########################  SPARK SESSION AND LOG FILE  ##############################

#create log file
current_time = time.localtime()
log_file_name = 'data_understanding_'+get_log_time()[:10].replace('/','')+'.log'
log_file_name = os.path.join('log', log_file_name)
lf = open(log_file_name, 'w')

#set spark session
spark = SparkSession.builder.appName("data_understanding").getOrCreate()

####################################################################################


####################################################################################
##############################  READING DATASETS ###################################

lf.write(get_log_time()+': Reading datasets\n')
#read the datasets
df_weather = spark.read.csv(os.path.join('data','weather.csv'), header=True)
df_buildings = spark.read.csv(os.path.join('data','building.csv'), header=True)
df_energy = spark.read.csv(os.path.join('data','energy.csv'), header=True)

#attributes
lf.write('Attributes weather\n{}\nAttributes buildings\n{}\nAttributes train\n{}'.format(df_weather.dtypes, df_buildings.dtypes, df_energy.dtypes))

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

lf.write('\n\n')
lf.write(get_log_time()+': Weather dataset\n')
n = df_weather.count()
lf.write('Number of istances: {}'.format(n))

##### Checking missing values
lf.write('\n\n')
lf.write(get_log_time()+': Checking missing values\n')
for col in df_weather.columns:
    n_nan = df_weather.filter(isnull(col)).count()
    lf.write('Column: {0}\nNumber of missing values: {1}\nPercentage of missing values: {2:.3f}%\n'.format(col, n_nan, n_nan/n*100))

##### Drop the two columns with too many missing values
lf.write('\n\n')
lf.write(get_log_time()+': Dropping columns with many missing values: ')
df_weather = df_weather.drop('cloud_coverage','precip_depth_1_hr')

###### Variables distribution
lf.write('\n\n')
lf.write(get_log_time()+': Computing variables distributions')
plt.figure(figsize=(15,10))
cont_cols = ['air_temperature', 'dew_temperature', 'sea_level_pressure', 'wind_direction', 'wind_speed']

for i, col in enumerate(cont_cols):
    ax = plt.subplot(2,3,i+1)
    distplot(ax, df_weather.select(col).dropna(), bins=20)
    plt.title(col)
    ax.get_yaxis().set_visible(False)

lf.write('\n\n')
lf.write(get_log_time()+': Saving variables distributions')
file_name = os.path.join('plots', 'weather_variables_dist')
plt.savefig(file_name)


###### Correlation
lf.write('\n\n')
lf.write(get_log_time()+': Computing correlation between variables')
df_corr = pd.DataFrame(index=cont_cols, columns=cont_cols)

#compute the correlation between pairs of variables
for i in range(len(cont_cols)):
    for j in range(i+1, len(cont_cols)):
        corr = df_weather.corr(cont_cols[i], cont_cols[j])
        
        #assign the value to a dataframe in two simmetrics positions since the correlation matrix is simmetric
        df_corr.loc[cont_cols[i], cont_cols[j]] = corr
        df_corr.loc[cont_cols[j], cont_cols[i]] = corr
        
#set to one the diagonal. No need to compute
for col in cont_cols:
    df_corr.loc[col, col] = 1.0


#Save results
file_name = os.path.join('results', 'weather_correlation.csv')
df_corr.to_csv(file_name)

fig = plot_corr(df_corr.astype(float), xnames=df_corr.columns)
fig.set_size_inches(12, 9)
file_name = os.path.join('plots', 'weather_correlation')
plt.savefig(file_name)
    

#### Tiem series of weaher dataset
#split the dataset for site_id in order to analize time series of the different sites separatly
lf.write('\n\n')
lf.write(get_log_time()+': Analizing weather time series')

pd_df_weather = df_weather.toPandas()
pd_dfs = [pd_df_weather[pd_df_weather['site_id']==i] for i in set(pd_df_weather.site_id)]

#plot the time series
lf.write('\n\n')
lf.write(get_log_time()+': Plotting weather time series')

plt.figure(figsize=(20,30))

for i, pd_df in enumerate(pd_dfs):
    for j, col in enumerate(cont_cols):
        ax = plt.subplot(16,5,i*5+j+1)
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        plt.plot(pd_df[col].dropna())
        plt.title('Site id: {} Feature: {}'.format(pd_df.site_id.iloc[0], col))

file_name = os.path.join('plots', 'weather_time_series')
plt.savefig(file_name)

#boxplots
lf.write('\n\n')
lf.write(get_log_time()+': Plotting boxplot weather time series')
plt.figure(figsize=(20,50))
for i, pd_df in enumerate(pd_dfs):
    for j, col in enumerate(cont_cols):
        ax = plt.subplot(16,5,i*5+j+1)
        pd.DataFrame(pd_df_weather[col]).boxplot()
        ax.get_xaxis().set_visible(False)
        plt.title('Site id: {} Feature: {}'.format(pd_df.site_id.iloc[0], col))

file_name = os.path.join('plots', 'weather_time_series_boxplots')
plt.savefig(file_name)

#compute the linear trend of each time series
lf.write('\n\n')
lf.write(get_log_time()+': Computing linear trend of each time series')

for pd_df in pd_dfs:
    for col in cont_cols:
        if len(pd_df[col].dropna()) == 0:
            lf.write('Site id: {} Feature: {} No values found\n'.format(pd_df.site_id.iloc[0], col))
            continue
            
        trend_model = LinearRegression(normalize=True, fit_intercept=True)
        trend_model.fit(np.array(pd_df[col].dropna().index).reshape((-1,1)), pd_df[col].dropna())
        lf.write('Site id: {} Feature: {} Trend model coefficient={}\n'.format(pd_df.site_id.iloc[0], col, trend_model.coef_[0]))

####################################################################################


####################################################################################  
############################  BUILDINGS DATASET  ###################################
lf.write('\n\n')
lf.write(get_log_time()+': Buildings dataset\n')
n = df_buildings.count()
lf.write('Number of istances: {}'.format(n))

##### Checking missing values
lf.write('\n\n')
lf.write(get_log_time()+': Checking missing values\n')

for col in df_buildings.columns:
    n_nan = df_buildings.filter(isnull(col)).count()
    lf.write('Column: {0}\nNumber of missing values: {1}\nPercentage of missing values: {2:.3f}%\n'.format(col, n_nan, n_nan/n*100))
    
##### Drop the two columns with too many missing values
lf.write('\n\n')
lf.write(get_log_time()+': Dropping columns with many missing values: ')
df_buildings = df_buildings.drop('year_built','floor_count')

###### Variables distribution
lf.write('\n\n')
lf.write(get_log_time()+': Computing variables distributions')

plt.figure(figsize=(11,5))
ax = plt.subplot(1,2,1)
distplot(ax, df_buildings.select('square_feet').dropna(), bins=20)
plt.title('square_feet')
ax.get_yaxis().set_visible(False)
primary_use_counts = df_buildings.groupBy('primary_use').count().toPandas()
plt.subplot(1,2,2)
plt.bar(primary_use_counts['primary_use'], primary_use_counts['count'])
plt.title('primary_use')
plt.xticks(rotation=90)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
file_name = os.path.join('plots', 'buildings_variables_dist')
plt.savefig(file_name)

####################################################################################
############################  FINAL DATASET  #######################################

lf.write('\n\n')
lf.write(get_log_time()+': Building the final dataset joining train table with weather and building tables')
df = df_energy.join(df_buildings, 'building_id')
df = df.join(df_weather, ['timestamp','site_id'])

#Remove records with missing values on target variable
lf.write('\n\n')
lf.write(get_log_time()+': Removing records with missing values on the target variable and create new attributes')
df = df.filter(df['meter_reading'] > 0)
df = df.withColumn('quarter', quarter(df['timestamp']))
df = df.withColumn('day_of_week', date_format(df['timestamp'], 'EEEE'))
df = df.withColumn('hour', hour(df['timestamp']))


###### Variables distribution
lf.write('\n\n')
lf.write(get_log_time()+': Computing variables distributions')

numeric_cols = ['air_temperature', 'dew_temperature', 'sea_level_pressure', 'wind_direction', 'wind_speed','square_feet']
categ_cols = ['meter','quarter','day_of_week','hour','primary_use']
target_cols = ['meter_reading']

plt.figure(figsize=(15,20))

for i, col in enumerate(numeric_cols+target_cols):
    ax = plt.subplot(4,3,i+1)
    distplot(ax, df.select(col).dropna(), bins=20)
    plt.title(col)
    ax.get_yaxis().set_visible(False)
    
for (i, col) in enumerate(categ_cols):
    counts = df.groupBy(col).count().toPandas()
    plt.subplot(4,3,i+8)
    plt.bar(counts[col], counts['count'])
    plt.title(col)
    if col in ['day_of_week','primary_use']:
        plt.xticks(rotation=90)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%e'))

file_name = os.path.join('plots', 'train_variables_dist')
plt.savefig(file_name)


##### compute statistics and save table
lf.write('\n\n')
lf.write(get_log_time()+': Computing statistics\n')
statistics = df.describe()
lf.write('\n\n')
lf.write(get_log_time()+': Saving statistics\n')
file_name = os.path.join('results','statistics')
save_table(statistics, file_name)

###### Correlation
lf.write('\n\n')
lf.write(get_log_time()+': Computing correlation between variables')

cols = numeric_cols+target_cols
df_corr = pd.DataFrame(index=cols, columns=cols)

#compute the correlation between pairs of variables
for i in range(len(cols)):
    for j in range(i+1, len(cols)):
        corr = df.corr(cols[i], cols[j])
        
        #assign the value to a dataframe in two simmetrics positions since the correlation matrix is simmetric
        df_corr.loc[cols[i], cols[j]] = corr
        df_corr.loc[cols[j], cols[i]] = corr
        
#set to one the diagonal. No need to compute
for col in cols:
    df_corr.loc[col, col] = 1.0


#Save results
file_name = os.path.join('results', 'train_correlation.csv')
df_corr.to_csv(file_name)

fig = plot_corr(df_corr.astype(float), xnames=df_corr.columns)
fig.set_size_inches(12, 9)
file_name = os.path.join('plots', 'train_correlation')
plt.savefig(file_name)

####################################################################################


##### close log file
lf.write('\n\n')
lf.write(get_log_time()+': SUCCESS\n')
lf.close()

spark.stop()
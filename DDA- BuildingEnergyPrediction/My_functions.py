from pyspark.ml.linalg import Vector as MLVector, Vectors as MLVectors
from pyspark.mllib.linalg import Vector as MLLibVector, Vectors as MLLibVectors
from pyspark.ml import linalg as ml_linalg
from pyspark.mllib.regression import LabeledPoint
import os
import time

#for saving pyspark dataframe
def save_table(table, file_name, save_as='csv'):
    if save_as == 'csv':
        table.repartition(1).write.mode('overwrite').csv(file_name, header = 'true')
    elif save_as == 'json':
        table.coalesce(1).write.mode('overwrite').format('json').save(file_name)

    files = os.listdir(file_name)
    for file in files:
        if os.path.splitext(file)[1] == '.'+save_as:
            os.rename(os.path.join(file_name, file), file_name+'.'+save_as)
        else:
            os.remove(os.path.join(file_name, file))
            
    os.removedirs(file_name)

#for getting the current time for writing logs inside the log file
def get_log_time():
    current_time = time.localtime()
    log_time = str(current_time[0])
    for i in range(1,3):
        log_time += '/'
        if current_time[i] < 10:
            log_time += '0'+str(current_time[i])
        else:
            log_time += str(current_time[i])
            
    log_time += ' '
    for i in range(3,6):
        if i != 3:
            log_time += ':'
        if current_time[i] < 10:
            log_time += '0'+str(current_time[i])
        else:
            log_time += str(current_time[i])
            
    return log_time

#casting from MLVector to MLLIBVector
def as_mllib(v):
    if isinstance(v, ml_linalg.SparseVector):
        return MLLibVectors.sparse(v.size, v.indices, v.values)
    elif isinstance(v, ml_linalg.DenseVector):
        return MLLibVectors.dense(v.toArray())
    else:
        raise TypeError("Unsupported type: {0}".format(type(v)))


def from_json_to_labeled_rdd(spark, file_name, n_features, sparse=True):
    df = spark.read.json(file_name)
    data = df.rdd.map(tuple)
    if sparse:
        data = data.map(lambda x: (x[1], MLLibVectors.sparse(x[0].size, x[0].indices, x[0].values)))
    else:
        data = data.map(lambda x: (x[1], MLLibVectors.dense(x[0].values)))
    
    data = data.map(lambda x: LabeledPoint(x[0], x[1])).filter(lambda x: x.features.size==n_features)
    
    return data
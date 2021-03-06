'''
The ideas are from https://spark.apache.org/docs/latest/ml-classification-regression.html#regression
'''
import pyspark.sql
from pyspark.ml import Pipeline
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator
import sys

from pyspark.sql import SQLContext
from pyspark import SparkContext
import math
from scipy import stats
import numpy as np
from itertools import izip

# !!!!!!!!!!!!!!!!!
# You need to get best lambda before you use this transformation
# !!!!!!!!!!!!!!!!!

def collect_features(input):
	'''
	this function is used to collect delta errors from array
	'''
	output = []
	for i in input:
		output.append(i['label'])
	return output

def get_root_mean_squared_error(list1,list2):
	if len(list1) != len(list2):
		raise Exception('two lists have different lengths')
	list_len = len(list1)
	sum_diff = 0
	for count in range(list_len):
		sum_diff = sum_diff + (list1[count]-list2[count])**2
	avg_sum_diff = sum_diff/list_len
	return math.sqrt(avg_sum_diff)

def collect_id(input_row_list):
	'''
	this function gets the id from the input
	pyspark Row list and return id list and pyspark
	Row without id
	'''
	id_list = []
	row_list = []
	for i in input_row_list:
		# last item of the array is id
		id_list.append(i['features'].toArray()[-1])
		# features without id
		tmp_features = i['features'].toArray()[:-1]
		# create sparse dict
		tmp_dict = dict(izip(range(len(tmp_features)),tmp_features))
		new_row_feature = pyspark.ml.linalg.SparseVector(len(tmp_features),tmp_dict)
		row_list.append(pyspark.sql.types.Row(label=i['label'], features=new_row_feature))
	return id_list, row_list

def find_min_label(input_row_list):
	'''
	this function finds the min label value
	'''
	tmp_min = 100000
	# obtain min label value
	for i in input_row_list:
		if i['label'] < tmp_min:
			tmp_min = i['label'] 
	return tmp_min


def boxcox_transform(input_row_list, tmp_min):
	'''
	this function transform original y
	it returns transformed value and a (tmp_min)
	'''
	best_lambda = float(sys.argv[5])
	row_list = []
	for i in input_row_list:
		# transform here
		row_list.append(pyspark.sql.types.Row(label=((i['label']-tmp_min)**best_lambda-1)/best_lambda, features=i['features']))
	return row_list

def reverse_boxcox_transform(input_list, tmp_min):
	'''
	this function transform y back
	'''
	best_lambda = float(sys.argv[5])
	result_list = []
	for i in input_list:
		# transform here
		tmp = i*best_lambda+1
		result_list.append((tmp**(1.0/best_lambda))+tmp_min)
	return result_list

# percentage for perdictions interval
# this means 10% to 90% interval
per_PI = 1 - 0.1/2
# degree of freedom
dof = 0
# sample standard deviation
ssd = 0
# sample mean
smean = 0

sc = SparkContext()
sqlContext = SQLContext(sc)

# Load the data stored in LIBSVM format as a DataFrame.
# filename = 'static/data/test.libsvm'
filename = sys.argv[1]
data = sqlContext.read.format("libsvm").load(filename)

# Automatically identify categorical features, and index them.
# We specify maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer =\
    VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = data.randomSplit([float(sys.argv[3]), 1-float(sys.argv[3])])
# (trainingData, testData) = data.randomSplit([0.9,0.1])

# Train a DecisionTree model.
dt = DecisionTreeRegressor(featuresCol="indexedFeatures")

# Chain indexer and tree in a Pipeline
pipeline = Pipeline(stages=[featureIndexer, dt])



tmp_min = find_min_label(data.collect())
# convert dataframe into list
test = testData.collect()
test = boxcox_transform(test,tmp_min)
# test_id, test = collect_id(test)
train = trainingData.collect()
train = boxcox_transform(train,tmp_min)

# Train model.  This also runs the indexer.
model = pipeline.fit(sc.parallelize(train).toDF())


predictions = model.transform(sc.parallelize(test).toDF())
predictions_list = predictions.toPandas()['prediction'].tolist()
ground_truth_list = predictions.toPandas()['label'].tolist()

# reverse log sinh transform
predictions_list = reverse_boxcox_transform(predictions_list,tmp_min)
ground_truth_list = reverse_boxcox_transform(ground_truth_list,tmp_min)


fp = open(sys.argv[2],'a')
fp.write(str(get_root_mean_squared_error(predictions_list,ground_truth_list)/float(sys.argv[4]))+'\n')
# fp.write(str(get_root_mean_squared_error(predictions_list,ground_truth_list)/float('0.1')+'\n')
fp.close()


# str(get_root_mean_squared_error(predictions_list,ground_truth_list)/float(sys.argv[4]))
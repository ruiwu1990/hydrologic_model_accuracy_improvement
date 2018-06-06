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

def log_sinh_transform(input_row_list,tmp_min,epsilon = 0.01):
	'''
	this function transform original y
	it returns transformed value and a (tmp_min)
	'''
	row_list = []
	for i in input_row_list:
		# transform here
		row_list.append(pyspark.sql.types.Row(label=math.log10(math.sinh(i['label']*float(sys.argv[7])+float(sys.argv[6])-tmp_min+epsilon))/float(sys.argv[7]), features=i['features']))
	return row_list

def reverse_log_sinh_transform(input_list,tmp_min,epsilon = 0.01):
	'''
	this function transform y back
	'''
	result_list = []
	for i in input_list:
		# transform here
		# http://mathworld.wolfram.com/InverseHyperbolicSine.html
		sinh_part = 10**(i*float(sys.argv[7]))
		tmp_origin = (math.log(sinh_part+math.sqrt(1+sinh_part*sinh_part))-float(sys.argv[6])+tmp_min-epsilon)/float(sys.argv[7])
		result_list.append(tmp_origin)
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
train_file = sys.argv[1]
trainingData = sqlContext.read.format("libsvm").load(train_file)

test_file = sys.argv[8]
testData = sqlContext.read.format("libsvm").load(test_file)

# Automatically identify categorical features, and index them.
# We specify maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer =\
    VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(trainingData)

# Train a DecisionTree model.
dt = DecisionTreeRegressor(featuresCol="indexedFeatures")

# Chain indexer and tree in a Pipeline
pipeline = Pipeline(stages=[featureIndexer, dt])


'''
use union to add first col in testData
need to convert row into df
need to remove first row in testData, using original.subtract(firstRowDF)
'''
tmp_min1 = find_min_label(trainingData.collect())
tmp_min2 = find_min_label(testData.collect())
tmp_min = min(tmp_min1, tmp_min2)
# convert dataframe into list
test = testData.collect()
test = log_sinh_transform(test,tmp_min)
# test_id, test = collect_id(test)
train = trainingData.collect()
train = log_sinh_transform(train,tmp_min)

# Train model.  This also runs the indexer.
model = pipeline.fit(sc.parallelize(train).toDF())


# train_id, train = collect_id(train)
train_len = len(train)
dof = train_len - 1
# check URL https://en.wikipedia.org/wiki/Prediction_interval
# for Ta
ta = stats.t.ppf(per_PI,dof)
train_delta = collect_features(train)
ssd = np.std(np.asarray(train_delta),ddof=1)
smean = sum(train_delta)/len(train_delta)
surfix = math.sqrt(1+(1.0/train_len))
# sort by dtime
test = sorted(test, key = lambda x: (x['features'][0],x['features'][1],x['features'][2]))
new_train_data = sorted(train, key = lambda x: (x['features'][0],x['features'][1],x['features'][2]))
predictions_list = []
ground_truth_list = []
PI_upper = []
PI_lower = []
time_stamp = []
test_len = len(test)
for count in range(test_len):
	current_row = test[count]
	new_data = sc.parallelize([current_row]).toDF()
	# get current prediction
	predictions = model.transform(new_data)
	# collect predictions into result lists
	tmp_predict = predictions.toPandas()['prediction'].tolist()[0]
	predictions_list.append(tmp_predict)
	ground_truth_list.append(predictions.toPandas()['label'].tolist()[0])
	tmp = ta*ssd*surfix
	PI_upper.append(smean+tmp)
	PI_lower.append(smean-tmp)
	time_stamp.append(str(int(current_row['features'][0]))+"--"+str(int(current_row['features'][1]))+"--"+str(int(current_row['features'][2])))
	new_train_data = [current_row] + new_train_data
	# remove the oldest data
	new_train_data = sorted(new_train_data, key = lambda x: (x['features'][0],x['features'][1],x['features'][2]))[1:]
	train_delta = collect_features(new_train_data)
	ssd = np.std(np.asarray(train_delta),ddof=1)
	smean = sum(train_delta)/len(train_delta)
	new_train_data_df = sc.parallelize([current_row] + new_train_data).toDF()
	# train model again with the updated data
	model = pipeline.fit(new_train_data_df)
	# print out hint info
	print "current processing: "+str(count+1)+"; "+str(test_len-count-1)+" rows left."

# reverse log sinh transform
PI_lower = reverse_log_sinh_transform(PI_lower,tmp_min)
PI_upper = reverse_log_sinh_transform(PI_upper,tmp_min)
predictions_list = reverse_log_sinh_transform(predictions_list,tmp_min)
ground_truth_list = reverse_log_sinh_transform(ground_truth_list,tmp_min)


fp = open(sys.argv[5] + '/bound.csv','w')

fp.write("time,lower,upper,prediction,ground_truth\n")
for i in range(len(PI_upper)):
	fp.write(time_stamp[i]+","+str(PI_lower[i])+","+str(PI_upper[i])+","+str(predictions_list[i])+","+str(ground_truth_list[i]/float(sys.argv[4]))+"\n")

fp.close()

# predictions_list = [10,2,3,4,5]
# ground_truth_list = [2,2,4,4,5]
# print (str(get_root_mean_squared_error(predictions_list,ground_truth_list)/float('0.1'))+'\n')

fp = open(sys.argv[2],'a')
fp.write(str(get_root_mean_squared_error(predictions_list,ground_truth_list)/float(sys.argv[4]))+'\n')
# fp.write(str(get_root_mean_squared_error(predictions_list,ground_truth_list)/float('0.1')+'\n')
fp.close()


# str(get_root_mean_squared_error(predictions_list,ground_truth_list)/float(sys.argv[4]))
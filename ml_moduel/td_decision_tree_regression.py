'''
The ideas are from https://spark.apache.org/docs/latest/ml-classification-regression.html#regression
'''

from pyspark.ml import Pipeline
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator
import sys

from pyspark.sql import SQLContext
from pyspark import SparkContext
import math

# print "system 1: "+sys.argv[1]
# print "system 2: "+sys.argv[2]
# print "system 3: "+sys.argv[3]
# print "system 4: "+sys.argv[4]

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
# (trainingData, testData) = data.randomSplit([0.7,0.3])

# Train a DecisionTree model.
dt = DecisionTreeRegressor(featuresCol="indexedFeatures")

# Chain indexer and tree in a Pipeline
pipeline = Pipeline(stages=[featureIndexer, dt])

# Train model.  This also runs the indexer.
model = pipeline.fit(trainingData)

'''
use union to add first col in testData
need to convert row into df
need to remove first row in testData, using original.subtract(firstRowDF)
'''
# convert dataframe into list
test = testData.collect()
train = trainingData.collect()
# sort by dtime
test = sorted(test, key = lambda x: (x['features'][1],x['features'][2],x['features'][3]))
new_train_data = sorted(train, key = lambda x: (x['features'][1],x['features'][2],x['features'][3]))
predictions_list = []
ground_truth_list = []
test_len = len(test)
for count in range(test_len):
	current_row = test[count]
	new_data = sc.parallelize([current_row]).toDF()
	# get current prediction
	predictions = model.transform(new_data)
	# collect predictions into result lists
	predictions_list.append(predictions.toPandas()['prediction'].tolist()[0])
	ground_truth_list.append(predictions.toPandas()['label'].tolist()[0])
	new_train_data = [current_row] + new_train_data
	# remove the oldest data
	new_train_data = sorted(new_train_data, key = lambda x: (x['features'][1],x['features'][2],x['features'][3]))[1:]
	new_train_data_df = sc.parallelize([current_row] + new_train_data).toDF()
	# train model again with the updated data
	model = pipeline.fit(new_train_data_df)
	# print out hint info
	print "current processing: "+str(count+1)+"; "+str(test_len-count-1)+" rows left."



def get_root_mean_squared_error(list1,list2):
	if len(list1) != len(list2):
		raise Exception('two lists have different lengths')
	list_len = len(list1)
	sum_diff = 0
	for count in range(list_len):
		sum_diff = sum_diff + (list1[count]-list2[count])**2
	avg_sum_diff = sum_diff/list_len
	return math.sqrt(avg_sum_diff)


# predictions_list = [10,2,3,4,5]
# ground_truth_list = [2,2,4,4,5]
# print (str(get_root_mean_squared_error(predictions_list,ground_truth_list)/float('0.1'))+'\n')

fp = open(sys.argv[2],'a')
fp.write(str(get_root_mean_squared_error(predictions_list,ground_truth_list)/float(sys.argv[4]))+'\n')
# fp.write(str(get_root_mean_squared_error(predictions_list,ground_truth_list)/float('0.1')+'\n')
fp.close()


# str(get_root_mean_squared_error(predictions_list,ground_truth_list)/float(sys.argv[4]))
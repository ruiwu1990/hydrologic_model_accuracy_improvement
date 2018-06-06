from pyspark.ml import Pipeline
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SQLContext
from pyspark import SparkContext
import sys 

sc = SparkContext()
sqlContext = SQLContext(sc)
# Load and parse the data file, converting it to a DataFrame.
filename = sys.argv[1]
data = sqlContext.read.format("libsvm").load(filename)

# Automatically identify categorical features, and index them.
# Set maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer =\
    VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = data.randomSplit([0.7, 0.3])

# Train a GBT model.
gbt = GBTRegressor(featuresCol="indexedFeatures", maxIter=10)

# Chain indexer and GBT in a Pipeline
pipeline = Pipeline(stages=[featureIndexer, gbt])

# Train model.  This also runs the indexer.
model = pipeline.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)

# Select example rows to display.
# predictions.select("prediction", "label", "features").show(50)

# Select (prediction, true label) and compute test error
evaluator = RegressionEvaluator(
    labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)

fp = open(sys.argv[2],'w')
fp.write("Root Mean Squared Error (RMSE) on test data = %g" % rmse)
fp.write('\n')

# print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)
predictions = model.transform(data)
pd_df = predictions.toPandas()
fp.write(','.join([str(i) for i in pd_df['prediction'].tolist()]))


# gbtModel = model.stages[1]
# print(gbtModel)  # summary only
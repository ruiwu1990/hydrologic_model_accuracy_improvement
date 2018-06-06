'''
This file use machine learning to find the direction
and use modified TD to improve model accuracy
I may need to change the timeout configuration by using this:
/cse/home/rwu/Desktop/hadoop/spark_installation/spark-2.1.0/bin/pyspark --conf spark.executor.heartbeatInterval=10000000 --conf spark.network.timeout=10000000
'''
import pandas as pd
import math
import subprocess
import sys
import csv
import os
from collections import defaultdict
import json
import os

app_path = os.path.dirname(os.path.abspath('__file__'))
spark_submit_location = '/home/hos1/Desktop/hadoop/spark-2.1.0/bin/spark-submit'
spark_config1 = '--conf spark.executor.heartbeatInterval=10000000'
spark_config2 = '--conf spark.network.timeout=10000000'
# this will be '/cse/home/rwu/Desktop/machine_learning_prms_accuracy/tmp_test'
# /cse/home/rwu/Desktop/hadoop/spark_installation/spark-2.1.0/bin/pyspark

# the following construct_line and convert_csv_into_libsvm
# convert csv into libsvm
# the function is basically 
# from https://github.com/zygmuntz/phraug/blob/master/csv2libsvm.py
def construct_line( label, line ):
	new_line = []
	if float( label ) == 0.0:
		label = "0"
	new_line.append( label )

	for i, item in enumerate( line ):
		if item == '' or float( item ) == 0.0:
			continue
		new_item = "%s:%s" % ( i + 1, item )
		new_line.append( new_item )
	new_line = " ".join( new_line )
	new_line += "\n"
	return new_line

def convert_csv_into_libsvm(input_file,output_file,label_index=0,skip_headers=True):
	'''
	the function converts csv into libsvm
	'''
	i = open( input_file, 'rb' )
	o = open( output_file, 'wb' )
	reader = csv.reader( i )

	if skip_headers:
		headers = reader.next()

	for line in reader:
		if label_index == -1:
			label = '1'
		else:
			label = line.pop( label_index )

		new_line = construct_line( label, line )
		o.write( new_line )

def delta_error_file(filename, e_filename, alpha=1):
	'''
	this function replace the first column (observed)
	and second column (predicted values) with delta_e
	alpha is factor for final_delta = alpha*delta
	'''
	pd_df = pd.read_csv(filename)
	# add the delta error col
	# first col observed
	observed_name = list(pd_df.columns.values)[0]
	# second col predicted
	predicted_name = list(pd_df.columns.values)[1]
	pd_df.insert(0,'delta_e',alpha*pd_df[observed_name].sub(pd_df[predicted_name]))
	# remove observed and predicted col
	del pd_df[observed_name]
	del pd_df[predicted_name]
	pd_df.to_csv(e_filename,index=False)
	return observed_name, predicted_name

def get_avg(filename):
	'''
	this function get avg value of a single col file
	'''
	fp = open(filename, 'r')
	sum_result = 0
	count = 0
	for line in fp:
		count = count + 1
		sum_result = float(line) + sum_result
	fp.close()
	return sum_result/count

def exec_regression(filename, regression_technique):
	'''
	this function run decision tree regression
	, output the results in a log file, and return the 
	predicted delta error col
	'''
	if regression_technique =='rf':
		log_path = app_path + '/rf_log.txt'
		err_log_path = app_path + '/rf_err_log.txt'
		exec_file_loc = app_path + '/ml_moduel/random_forest_regression.py'
		result_file = app_path + '/rf_result.txt'

	elif regression_technique =='decision_tree':
		log_path = app_path + '/decision_tree_log.txt'
		err_log_path = app_path + '/decision_tree_err_log.txt'
		# change!!!!!!!!!!!!!!!
		# exec_file_loc = app_path + '/ml_moduel/td_decision_tree_regression.py'
		exec_file_loc = app_path + '/ml_moduel/td_decision_tree_regression_prediction_interval.py'
		# exec_file_loc = app_path + '/ml_moduel/decision_tree_regression.py'
		result_file = app_path + '/decision_tree_result.txt'

	elif regression_technique =='glr':
		log_path = app_path + '/glr_log.txt'
		err_log_path = app_path + '/glr_err_log.txt'
		exec_file_loc = app_path + '/ml_moduel/generalized_linear_regression.py'
		result_file = app_path + '/glr_result.txt'

	elif regression_technique =='gb_tree':
		log_path = app_path + '/gbt_log.txt'
		err_log_path = app_path + '/gbt_err_log.txt'
		exec_file_loc = app_path + '/ml_moduel/gradient_boosted_regression.py'
		result_file = app_path + '/gbt_result.txt'
	else:
		print 'Sorry, current system does not support the input regression technique'
	
	min_rmse = 10000
	best_alpha = -1
	best_window_per = -1
	fp1 = open('all_results.csv','w')
	# print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"+exec_file_loc
	# change!!!!!!!!!!!!!!!
	# for alpha_count in range(10):
	for alpha_count in range(1):
		# alpha = 0.1*(alpha_count+1)
		# change!!!!!!!!!!!!!!!
		alpha = 0.8
		# change!!!!!!!!!!!!!!!
		# for window_count in range(9):
		for window_count in range(1):
			# max window size is 95%
			# window_per = 0.1*(window_count+1)
			# change!!!!!!!!!!!!!!!
			window_per = 0.9
			# clean previous generated results
			if os.path.isfile(result_file):
				# if file exist
				os.remove(result_file)

			# get the libsvm file
			delta_error_csv = app_path + '/temp_delta_error.csv'
			delta_error_filename = app_path + '/delta_error.libsvm'
			# observed_name, predicted_name = delta_error_file(filename,delta_error_filename)
			
			delta_error_file(filename,delta_error_csv,alpha)
			convert_csv_into_libsvm(delta_error_csv,delta_error_filename)

			# this command will work if source the spark-submit correctly
			# command = ['spark-submit',exec_file_loc,output_file,result_file]
			# hard coded here because the pyspark random split is confusing... I need to mannually 
			# obtain the test data length
			# test_data_len = 142
			# for i in range(test_data_len):
			# change!!!!!!!!!!!!!!!
			# command = [spark_submit_location, exec_file_loc,delta_error_filename,result_file, str(window_per), spark_config1, spark_config1]
			command = [spark_submit_location, exec_file_loc,delta_error_filename,result_file, str(window_per), str(alpha), app_path, spark_config1, spark_config1]
			# command = [spark_submit_location,exec_file_loc,output_file,i]
			#  30 times crossover validation
			for i in range(30):
			# !!!!!!!!!!!!!!!!!!!change
			# for i in range(1):
			# execute the model
				with open(log_path, 'wb') as process_out, open(log_path, 'rb', 1) as reader, open(err_log_path, 'wb') as err_out:
					process = subprocess.Popen(
						command, stdout=process_out, stderr=err_out, cwd=app_path)

				# this waits the process finishes
				process.wait()
				print "current processing loop for alaph "+str(alpha)+", and window size "+str(window_per)+": "+str(i)+"//////////////////////////////"
				# sys.exit()

			cur_avg_rmse = get_avg(result_file)
			# need to times cur_avg_rmse back to real value
			# change!!!!!!!!!!!!!!!
			# cur_avg_rmse = cur_avg_rmse * (1/alpha)

			print "~~~~~current avg is rmse: "+str(cur_avg_rmse)
			fp1.write(str(alpha)+","+str(window_per)+","+str(cur_avg_rmse)+'\n')
			if cur_avg_rmse < min_rmse:
				min_rmse = cur_avg_rmse
				best_alpha = alpha
				best_window_per = window_per
	fp1.close()
	print "min rmse is: "+ str(min_rmse)+"; best alpha is: "+str(best_alpha)+"; best window size is: "+str(window_per)
	return True

# input file, first column is observation, second column is prediction
exec_regression('prms_input.csv','decision_tree')
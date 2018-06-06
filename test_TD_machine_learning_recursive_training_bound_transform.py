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
import shutil
import numpy as np
from util import construct_line, convert_csv_into_libsvm, delta_error_file, smooth_origin_input_cse, get_avg
from util import obtain_total_row_num, get_root_mean_squared_error, original_csv_rmse, split_csv_file, split_csv_file_loop
from util import collect_corresponding_obs_pred, merge_bound_file, exec_regression_by_name, exec_regression

app_path = os.path.dirname(os.path.abspath('__file__'))
spark_submit_location = '/home/rwu/Desktop/spark-2.3.0-bin-hadoop2.7/bin/spark-submit'
spark_config1 = '--conf spark.executor.heartbeatInterval=10000000'
spark_config2 = '--conf spark.network.timeout=10000000'


def real_crossover_exec_regression(filename, regression_technique, window_per=0.5, training_window_per = 0.9, transform_tech = 'logsinh'):
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
			result_file = app_path + '/decision_tree_result.txt'

			if transform_tech == 'logsinh':
				exec_no_recursive_file_loc = app_path + '/ml_moduel/decision_tree_regression_transform_no_recursive_logsinh.py'
				# exec_file_loc = app_path + '/ml_moduel/td_decision_tree_regression_prediction_interval_log_sinh.py'
				# exec_file_loc = app_path + '/ml_moduel/decision_tree_regression.py'
			elif transform_tech == 'boxcox':
				exec_no_recursive_file_loc = app_path + '/ml_moduel/decision_tree_regression_transform_no_recursive_boxcox.py'
			else:
				print 'Sorry, current system does not support the input transformation technique'
			

	elif regression_technique =='glr':
		log_path = app_path + '/glr_log.txt'
		err_log_path = app_path + '/glr_err_log.txt'
		result_file = app_path + '/glr_result.txt'
		if transform_tech == 'boxcox':
			exec_no_recursive_file_loc = app_path + '/ml_moduel/generalized_linear_regression_transform_no_recursive_boxcox.py'
		else:
			print 'Sorry, current system does not support the input transformation technique'

	elif regression_technique =='gb_tree':
		log_path = app_path + '/gbt_log.txt'
		err_log_path = app_path + '/gbt_err_log.txt'
		result_file = app_path + '/gbt_result.txt'

		if transform_tech == 'logsinh':
			exec_no_recursive_file_loc = app_path + '/ml_moduel/gb_tree_regression_transform_no_recursive_logsinh.py'
			# exec_file_loc = app_path + '/ml_moduel/td_gd_tree_regression_prediction_interval_log_sinh.py'
		else:
			print 'Sorry, current system does not support the input transformation technique'

	else:
		print 'Sorry, current system does not support the input regression technique'
	
	min_rmse = 10000
	best_alpha = -1
	fp1 = open('all_results.csv','w')

	train_file='prms_input1.csv'
	test_file='prms_input2.csv'
	split_csv_file(filename, window_per, train_file, test_file)

	best_a = -1
	best_b = -1
	best_lambda = -1
	# change!!!!!!!!!!!!!!!
	#for alpha_count in range(5):
	for alpha_count in range(2):
		alpha = 0.1*(alpha_count+2)
		#alpha = 0.1*(alpha_count+1)
		
		# clean previous generated results
		if os.path.isfile(result_file):
			# if file exist
			os.remove(result_file)

		# get the libsvm file
		delta_error_csv = app_path + '/temp_delta_error.csv'
		delta_error_filename = app_path + '/delta_error.libsvm'
		
		delta_error_file(train_file,delta_error_csv,alpha)
		convert_csv_into_libsvm(delta_error_csv,delta_error_filename)

		if transform_tech == 'logsinh':
			for a_count in range(10):
			# change!!!!!!!!!!!!!!!!!!!!
			# for a_count in range(4):
				# tmp_a = 0.01*(a_count+2)+0.0005
				tmp_a = 0.01*(a_count+1)+0.0005

				for b_count in range(10):
				# change!!!!!!!!!!!!!!!!!!!!11
				# for b_count in range(4):
					# tmp_b = 0.01*(b_count+2)+0.0005
					tmp_b = 0.01*(b_count+1)+0.0005

					# this command will work if source the spark-submit correctly
					# no recursive for crossover validation
					
					command = [spark_submit_location, exec_no_recursive_file_loc,delta_error_filename,result_file, str(training_window_per), str(alpha), str(tmp_a), str(tmp_b), spark_config1, spark_config2]

					#  30 times crossover validation
					# for i in range(30):
					# !!!!!!!!!!!!!!!!!!!change
					for i in range(10):
					# execute the model
						with open(log_path, 'wb') as process_out, open(log_path, 'rb', 1) as reader, open(err_log_path, 'wb') as err_out:
							process = subprocess.Popen(
								command, stdout=process_out, stderr=err_out, cwd=app_path)

						# this waits the process finishes
						process.wait()
						print "current processing loop for alaph "+str(alpha)+", a: "+str(tmp_a)+", b: "+str(tmp_b)+", and crossover time: "\
								+str(i)+"//////////////////////////////"
						# sys.exit()

					cur_avg_rmse = get_avg(result_file)
					os.remove(result_file)

					print "~~~~~current avg is rmse: "+str(cur_avg_rmse)
					fp1.write(str(alpha)+","+str(window_per)+","+str(cur_avg_rmse)+","+str(tmp_a)+","+str(tmp_b)+'\n')
					if cur_avg_rmse < min_rmse:
						min_rmse = cur_avg_rmse
						best_alpha = alpha
						best_a = tmp_a
						best_b = tmp_b

		elif transform_tech == 'boxcox':
			for lambda_count in range(20):
				tmp_lambda = 0.2*(lambda_count+1)+7

				# this command will work if source the spark-submit correctly
				# no recursive for crossover validation
				command = [spark_submit_location, exec_no_recursive_file_loc,delta_error_filename,result_file, str(training_window_per), str(alpha), str(tmp_lambda), spark_config1, spark_config2]

				#  30 times crossover validation
				# for i in range(30):
				# !!!!!!!!!!!!!!!!!!!change
				for i in range(10):
				# execute the model
					with open(log_path, 'wb') as process_out, open(log_path, 'rb', 1) as reader, open(err_log_path, 'wb') as err_out:
						process = subprocess.Popen(
							command, stdout=process_out, stderr=err_out, cwd=app_path)

					# this waits the process finishes
					process.wait()
					print "current processing loop for alaph "+str(alpha)+", lambda: "+str(tmp_lambda)+", and crossover time: "\
							+str(i)+"//////////////////////////////"
					# sys.exit()

				cur_avg_rmse = get_avg(result_file)
				os.remove(result_file)

				print "~~~~~current avg is rmse: "+str(cur_avg_rmse)
				fp1.write(str(alpha)+","+str(window_per)+","+str(cur_avg_rmse)+","+str(tmp_lambda)+'\n')
				if cur_avg_rmse < min_rmse:
					min_rmse = cur_avg_rmse
					best_alpha = alpha
					best_lambda = tmp_lambda

	fp1.close()
	if transform_tech == 'logsinh':
		print "min rmse is: "+ str(min_rmse)+"; best alpha is: "+str(best_alpha)+"; best a is: "+str(best_a)+"; best b is: "+str(best_b)+"; current window size is: "+str(window_per)
	elif transform_tech == 'boxcox':
		print "min rmse is: "+ str(min_rmse)+"; best alpha is: "+str(best_alpha)+"; best lambda is: "+str(tmp_lambda)+"; current window size is: "+str(window_per)

	# recursive test file, with best alpha, a and b
	if os.path.isfile(result_file):
		# if file exist
		os.remove(result_file)

	# change!!! need to recover
	exec_regression(filename, regression_technique, window_per, best_alpha,app_path, best_a, best_b, True, True, 500, transform_tech, best_lambda)

	return True

def convert_month_word_into_int(month_word):
	'''
	'''
	if month_word.lower() == 'jan':
		return 1
	elif month_word.lower() == 'feb':
		return 2
	elif month_word.lower() == 'mar':
		return 3
	elif month_word.lower() == 'apr':
		return 4
	elif month_word.lower() == 'may':
		return 5
	elif month_word.lower() == 'jun':
		return 6
	elif month_word.lower() == 'jul':
		return 7
	elif month_word.lower() == 'aug':
		return 8
	elif month_word.lower() == 'sep':
		return 9
	elif month_word.lower() == 'oct':
		return 10
	elif month_word.lower() == 'nov':
		return 11
	elif month_word.lower() == 'sep':
		return 12

def extract_feature_row(input_feature, tmp_input_feature):
	'''
	convert feature file 15mins into hourly
	'''
	fp = open(input_feature,'r')
	# skip header
	fp.readline()
	fp_out = open(tmp_input_feature,'w')
	fp_out.write('year,month,day,precip'+'\n')
	count = 0
	tmp_row2 = fp.readline()
	while tmp_row2 != '':
		row_list2 = []
		# remove ""
		tmp_row2 = tmp_row2.replace("\"","")
		# remove spaces
		tmp_row2 = tmp_row2.replace(" ","")
		tmp_list2 = tmp_row2.strip().split(',')
		# year
		year = int(tmp_list2[0][5:])
		row_list2.append(str(year))
		# month
		month = convert_month_word_into_int(tmp_list2[0][2:5])
		row_list2.append(str(month))
		# day
		day = float(tmp_list2[0][:2])
		if count == 0:
			# hour
			hour = float(tmp_list2[1].split(":")[0])
			row_list2.append(str(day+hour/24))
			# preciptation
			row_list2.append(tmp_list2[-1])
		else:
			# hour
			hour = float(tmp_list2[1].split(":")[0])+1
			row_list2.append(str(day+hour/24))
			precip1 = float(tmp_list2[-1])
			precip2 = float(fp.readline().strip().split(',')[-1])
			precip3 = float(fp.readline().strip().split(',')[-1])
			precip4 = float(fp.readline().strip().split(',')[-1])
			row_list2.append(str(precip1+precip2+precip3+precip4))

		count = count + 1
		fp_out.write(','.join(row_list2)+'\n')
		tmp_row2 = fp.readline()

	fp.close()
	fp_out.close()


def preprocess_input_csv(input_result, input_feature, cali_file='data/tmp_cali.csv',uncali_file='data/tmp_uncali.csv'):
	'''
	this function is used to convert two input files into
	two files (obs, with_cali_pre, features) and (obs, without_cali_pre, features)
	!!!!!be careful about these things:
	1) the input files' title may need to be changed
	2) start time may have a problem, coz features are recorded every 15 min
	   and results are recorded every one hour
	3) result file may have less records than feature file
	'''
	fp1 = open(input_result,'r')
	fp2 = open(input_feature,'r')
	fp_cali = open(cali_file,'w')
	fp_uncali = open(uncali_file,'w')
	# write header
	fp_cali.write('runoff_obs,basin_cfs_pred,year,month,day,precip\n')
	fp_uncali.write('runoff_obs,basin_cfs_pred,year,month,day,precip\n')

	# skip header
	tmp_row1 = fp1.readline()
	tmp_row2 = fp2.readline()
	# read first data line
	tmp_row1 = fp1.readline()
	tmp_row2 = fp2.readline()
	while tmp_row1 != '' and tmp_row2 != '':
		row1_list = tmp_row1.strip().split(',')
		fp_cali.write(str(row1_list[-1])+','+str(row1_list[-3])+','+tmp_row2)
		fp_uncali.write(str(row1_list[-1])+','+str(row1_list[-2])+','+tmp_row2)
		tmp_row1 = fp1.readline()
		tmp_row2 = fp2.readline()

	
	fp1.close()
	fp2.close()
	fp_cali.close()
	fp_uncali.close()


# input file, first column is observation, second column is prediction
# exec_regression('prms_input.csv','decision_tree')


# print 'original rmse is: '+str(original_csv_rmse('prms_input.csv',0.9))

# exec_regression('prms_input.csv', 'gb_tree', 0.9, 0.8,app_path, 0.0105, 0.0205, True, True, 100)

# exec_regression_by_name('sub_results/prms_train0.csv', 'sub_results/prms_test0.csv', 'gb_tree', 0.2, 0.4,app_path, 0.0405, 0.0505)
# print original_csv_rmse('prms_input.csv', window_per=0.4)

#merge_bound_file('smoothed_prms_input.csv', file_path,loop_time)

# --------------------------recursive starts-------------------------------------------
# create smooth version input file
smooth_origin_input_cse('data/prms_input.csv', 'data/smoothed_prms_input.csv', 30)
real_crossover_exec_regression('smoothed_prms_input.csv','gb_tree',0.5)
# exec_regression('data/smoothed_prms_input.csv', 'gb_tree',0.5, 0.6,app_path, 0.0305, 0.0105, True, True, 500)

#smooth_origin_input_cse('data/prms_input.csv', 'data/smoothed_prms_input.csv', 10)
#exec_regression('data/smoothed_prms_input.csv', 'gb_tree',0.5, 0.9,app_path, 0.1005, 0.0705, True, True, 500)

# train_file = 'data/window_train.csv'
# test_file = 'data/window_test.csv'
# filename = 'data/train_input.csv'
# exec_regression_by_name(filename, train_file, test_file, 'gb_tree', 0.5, 0.9,app_path, 0.1005, 0.0705, True, True, 100)
# --------------------------recursive ends-------------------------------------------

#smooth_origin_input_cse('prms_input_without_calibrate.csv', 'smoothed_prms_input.csv', 10)
# real_crossover_exec_regression('smoothed_prms_input.csv','gb_tree',0.5)
#exec_regression('smoothed_prms_input.csv', 'gb_tree',0.5, 0.9,app_path, 0.1005, 0.0705, True, True, 500)

# with cali
#extract_feature_row('data/2.csv','data/tmp_2.csv')
#preprocess_input_csv('data/1.csv','data/tmp_2.csv')
#print 'original rmse is: '+str(original_csv_rmse('data/tmp_cali.csv',0.55))
# smooth_origin_input_cse('data/tmp_cali.csv', 'data/smoothed_prms_input.csv', 100000000)
#real_crossover_exec_regression('data/smoothed_prms_input.csv','decision_tree',0.55, transform_tech = 'boxcox')
# exec_regression('data/smoothed_prms_input.csv', 'decision_tree',0.55, 0.3,app_path, 0.0405, 0.0305, True, True, 50, 'boxcox', 11.0)
# exec_regression_by_name('sub_results/prms_train0.csv', 'sub_results/prms_test0.csv', 'decision_tree', 0.2, 0.4,app_path, 0.0405, 0.0505, 'boxcox', 7.0)

# # # without cali
#extract_feature_row('data/2.csv','data/tmp_2.csv')
#preprocess_input_csv('data/1.csv','data/tmp_2.csv')
#print 'original rmse is: '+str(original_csv_rmse('data/tmp_uncali.csv',0.55))
#smooth_origin_input_cse('data/tmp_uncali.csv', 'data/smoothed_prms_input.csv', 1000000000)
#real_crossover_exec_regression('data/smoothed_prms_input.csv','decision_tree',0.55, transform_tech = 'boxcox')
# exec_regression('data/smoothed_prms_input.csv', 'decision_tree',0.55, 0.4,app_path, 0.0405, 0.0305, True, True, 50, 'boxcox', 11.0)

# --------------------------recursive starts-------------------------------------------
# create smooth version input file
# smooth_origin_input_cse('prms_input.csv', 'smoothed_prms_input.csv', 30)
#real_crossover_exec_regression('data/threshold_extreme_event.csv','gb_tree',0.5)
#real_crossover_exec_regression('data/threshold_normal_event.csv','gb_tree',0.5)
#real_crossover_exec_regression('data/label_extreme_event.csv','gb_tree',0.5)
# real_crossover_exec_regression('data/label_normal_event.csv','gb_tree',0.5)
# exec_regression('data/label_extreme_event.csv', 'gb_tree',0.5, 0.2,app_path, 0.0505, 0.0605, True, True, 500)
#exec_regression('data/threshold_extreme_event.csv', 'gb_tree',0.5, 0.2,app_path, 0.0605, 0.0605, True, True, 5)
# smooth_origin_input_cse('prms_input.csv', 'smoothed_prms_input.csv', 10)
# exec_regression('smoothed_prms_input.csv', 'gb_tree',0.5, 0.9,app_path, 0.1005, 0.0705, True, True, 500)

# train_file = 'data/threshold_extreme_event_train.csv'
# test_file = 'data/threshold_extreme_event_test.csv'
# filename = 'data/threshold_extreme_event.csv'
# exec_regression_by_name(filename, train_file, test_file, 'gb_tree', 0.5, 0.2,app_path, 0.0405, 0.0305, True, True, 100)
# --------------------------recursive ends-------------------------------------------
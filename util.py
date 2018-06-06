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
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.mlab as mlab

app_path = os.path.dirname(os.path.abspath('__file__'))
spark_submit_location = '/home/rwu/Desktop/spark-2.3.0-bin-hadoop2.7/bin/spark-submit'
spark_config1 = '--conf spark.executor.heartbeatInterval=10000000'
spark_config2 = '--conf spark.network.timeout=10000000'

# ------------------------
# test matrix functions

def get_root_mean_squared_error(list1,list2):
	if len(list1) != len(list2):
		raise Exception('two lists have different lengths')
	list_len = len(list1)
	sum_diff = 0
	for count in range(list_len):
		sum_diff = sum_diff + (list1[count]-list2[count])**2
	avg_sum_diff = sum_diff/list_len
	return math.sqrt(avg_sum_diff)

def get_pbias(list1, list2):
	'''
	percent bias
	list1 is model simulated value
	list2 is observed data
	'''
	if len(list1) != len(list2):
		raise Exception('two lists have different lengths')
	list_len = len(list1)
	sum_diff = 0
	sum_original = 0
	for count in range(list_len):
		sum_diff = sum_diff + (list1[count]-list2[count])
		sum_original = sum_original + list2[count]
	result = sum_diff/sum_original
	return result*100

def get_coeficient_determination(list1,list2):
	'''
	list1 is model simulated value
	list2 is observed data
	'''
	if len(list1) != len(list2):
		raise Exception('two lists have different lengths')
	list_len = len(list1)
	mean_list1 = reduce(lambda x, y: x + y, list1) / len(list1)
	mean_list2 = reduce(lambda x, y: x + y, list2) / len(list2)
	sum_diff = 0
	sum_diff_o_s = 0
	sum_diff_p_s = 0
	for count in range(list_len):
		sum_diff = sum_diff + (list1[count]-mean_list1)*(list2[count]-mean_list2)
		sum_diff_o_s = sum_diff_o_s + (list2[count]-mean_list2)**2
		sum_diff_p_s = sum_diff_p_s + (list1[count]-mean_list1)**2
	result = (sum_diff/(pow(sum_diff_o_s,0.5)*pow(sum_diff_p_s,0.5)))**2
	return result

def get_nse(list1,list2):
	'''
	Nash-Sutcliffe efficiency
	list1 is model simulated value
	list2 is observed data
	'''
	if len(list1) != len(list2):
		raise Exception('two lists have different lengths')
	list_len = len(list1)
	sum_diff_power = 0
	sum_diff_o_power = 0
	mean_list2 = reduce(lambda x, y: x + y, list2) / len(list2)
	for count in range(list_len):
		sum_diff_power = sum_diff_power + (list1[count]-list2[count])**2
		sum_diff_o_power = sum_diff_o_power + (list2[count]-mean_list2)**2
	result = sum_diff_power/sum_diff_o_power
	return 1 - result

# ------------------------

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

def smooth_origin_input_cse(input_file, output_file, threshold):
	'''
	this function smooths original function predictions
	if Pt - Pt-1 > threshold, then Pt <- Pt-1
	'''
	df_input = pd.read_csv(input_file)
	# second col predicted
	predicted_name = list(df_input.columns.values)[1]
	origin_predict = df_input[predicted_name].tolist()
	for i in range(1,len(origin_predict)-1):
		if abs(origin_predict[i] - origin_predict[i+1]) > threshold:
			origin_predict[i+1] = origin_predict[i]

	# replace original prediction with smoothed prediction
	df_input[predicted_name] = pd.Series(np.asarray(origin_predict))
	df_input.to_csv(output_file, mode = 'w', index=False)

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

def obtain_total_row_num(filename):
	'''
	get file total lines
	'''
	fp = open(filename,'r')
	result = sum(1 for row in fp)
	fp.close()
	return result


def original_csv_rmse(filename, window_per=0.9):
	'''
	this function finds original model (1-window_per%) rmse
	'''
	fir_output_file='prms_input1.csv'
	sec_output_file='prms_input2.csv'
	split_csv_file(filename, window_per, fir_output_file, sec_output_file)
	df = pd.read_csv(sec_output_file)
	return get_root_mean_squared_error(df['runoff_obs'].tolist(),df['basin_cfs_pred'].tolist())


def split_csv_file(input_file='prms_input.csv', n_per=0.9, fir_output_file='prms_input1.csv', sec_output_file='prms_input2.csv', padding_line_num = 0):
	'''
	fir_output_file n_per and sec_output_file 1-n_per
	'''
	# -1 coz title
	row_num = obtain_total_row_num(input_file)-1
	fir_row = int(row_num*n_per)
	fp = open(input_file,'r')
	fp1 = open(fir_output_file,'w')
	fp2 = open(sec_output_file,'w')
	# write title
	title = fp.readline()
	fp1.write(title)
	fp2.write(title)

	#  skip the padding lines
	for i in range(padding_line_num):
		fp.readline()

	cur_row_num = 0
	while cur_row_num < row_num:
		tmp_line = fp.readline()
		if cur_row_num < fir_row:
			fp1.write(tmp_line)
		else:
			fp2.write(tmp_line)
		cur_row_num = cur_row_num + 1

	fp.close()
	fp1.close()
	fp2.close()

def split_csv_file_loop(input_file, loop_count, train_file_len, max_test_file_len, train_file='sub_results/prms_train.csv', test_file='sub_results/prms_test.csv'):
	'''
	this function is used to split files into train file and test file
	train file and test file together do not equal input file
	'''
	# -1 coz title
	row_num = obtain_total_row_num(input_file)-1
	fp = open(input_file,'r')
	fp1 = open(train_file,'w')
	fp2 = open(test_file,'w')
	# write title
	title = fp.readline()
	fp1.write(title)
	fp2.write(title)

	#  skip the padding lines
	for i in range(max_test_file_len*loop_count):
		fp.readline()
		
	cur_row_num = 0
	# test
	# test_line_count = 0
	while cur_row_num < (train_file_len+max_test_file_len):
		tmp_line = fp.readline()
		if cur_row_num < train_file_len:
			fp1.write(tmp_line)
		else:
			fp2.write(tmp_line)
			# test_line_count = test_line_count +1
		cur_row_num = cur_row_num + 1

	fp.close()
	fp1.close()
	fp2.close()
	# print test_line_count
	print 'Split file done....'

def convert_str_into_time(input_list):
	result_list = []
	if int(float(input_list[0].split('--')[2])) == float(input_list[0].split('--')[2]):
		for i in input_list:
			result_list.append(datetime.strptime(i, '%Y--%m--%d'))
	else:
		for i in input_list:
			# turn day into day and hour
			tmp = i.split('--')
			tmp = [float(m) for m in tmp]
			# separate hour from day
			tmp.append(round(float(str(tmp[2]-int(tmp[2]))[1:])*24))
			tmp = [int(m) for m in tmp]
			tmp = [str(m) for m in tmp]
			tmp = '--'.join(tmp)
			try:
				result_list.append(datetime.strptime(tmp, '%Y--%m--%d--%H'))
			except Exception:
				print "Error with this day:"+tmp+";original time: "+i
				tmp = tmp.split('--')
				tmp = [int(m) for m in tmp]
				tmp[2] = tmp[2] - 1
				tmp[3] = 23
				tmp = [str(m) for m in tmp]
				tmp = '--'.join(tmp)
				result_list.append(datetime.strptime(tmp, '%Y--%m--%d--%H'))
	return result_list

def convert_year_month_day_into_time(year_list,month_list,day_list):
	'''
	'''
	result_list = []
	for counter in range(len(year_list)):
		tmp = str(year_list[counter])+'--'+str(month_list[counter])+'--'+str(day_list[counter])
		result_list.append(datetime.strptime(tmp, '%Y--%m--%d'))

	return result_list		



def nth_decimal(input_a, n):
	'''
	this function will not carry-over
	e.g. 1.55, 1 => 1.5
	'''
	input_str = str(input_a)
	result_list = input_str.split('.')
	result_list[1] = result_list[1][:n]
	return float(result_list[0]+'.'+result_list[1])

def collect_corresponding_obs_pred(input_df, time_list):
	'''
	this function collects corresponding values
	based on time info, and return obs and original pred
	'''
	obs_list = []
	original_pred_list = []
	for i in time_list:
		time_info = i.split('--')
		year = time_info[0]
		month = time_info[1]
		day = time_info[2]
		if int(float(day)) == float(day):
			aim_df = input_df.query('year=='+year+' & month=='+month+' & day=='+day)
		else:
			# if day is not int then only convert day into 8 decimals
			tmp_df = input_df.round({'day': 8})
			aim_df = tmp_df.query('year=='+year+' & month=='+month+' & day=='+'{0:.8f}'.format(float(day)))
			# aim_df = input_df.query('year=='+year+' & month=='+month+' & day=='+str(nth_decimal(day,6)))
		try:
			# get first search result
			obs_list.append(float(aim_df.iloc[0]['runoff_obs']))
			original_pred_list.append(float(aim_df.iloc[0]['basin_cfs_pred']))
		except Exception:
			print "errors with time: "+i
	return obs_list, original_pred_list


def merge_bound_file(original_file, file_path,loop_time):
	'''
	this function merge bound0.csv, bound1.csv, ..., boundn-1.csv 
	and return rmse
	'''
	bound_loc = file_path+'bound.csv'
	fp = open(bound_loc,'w')
	# write header
	fp_tmp = open(file_path+'bound0.csv','r')
	fp.write(fp_tmp.readline())
	fp_tmp.close()
	for i in range(loop_time):
		fp_tmp = open(file_path+'bound'+str(i)+'.csv','r')
		# skip header
		fp_tmp.readline()
		for line in fp_tmp:
			# skip line with empty space
			if line not in ['\n', '\r\n']:
				# print line
				fp.write(line)
		fp_tmp.close()
	fp.close()

	# make sure that all values above 0, coz physical meaning
	df_delta = pd.read_csv(bound_loc)
	df_origin = pd.read_csv(original_file)
	time_list = df_delta['time'].tolist()
	truth,origin_pred = collect_corresponding_obs_pred(df_origin,time_list)

	lower_error = df_delta['lower'].tolist()
	lower = [x + y for x, y in zip(lower_error,origin_pred)]
	df_delta['lower'] = pd.Series(np.asarray(lower))

	upper_error = df_delta['upper'].tolist()
	upper = [x + y for x, y in zip(upper_error,origin_pred)]
	df_delta['upper'] = pd.Series(np.asarray(upper))

	prediction_error = df_delta['prediction'].tolist()
	prediction = [x + y for x, y in zip(prediction_error,origin_pred)]
	# need series
	df_delta['prediction'] = pd.Series(np.asarray(prediction))
	df_delta['ground_truth'] = pd.Series(np.asarray(truth))

	# replace negative values with zeros
	num = df_delta._get_numeric_data()
	num[num<0] = 0

	df_delta.to_csv(bound_loc,index=False)
	# get rmse
	return get_root_mean_squared_error(df_delta['prediction'],truth)
	# return get_root_mean_squared_error(df_delta['prediction'],df_delta['ground_truth'].tolist())



def exec_regression(filename, regression_technique, window_per, best_alpha,app_path, best_a, best_b, recursive = True, transformation = True, max_row_num=500, transform_tech = 'logsinh', best_lambda = 1):
	'''
	!!!!!!!!!!!!!!!file should be ordered based on time, from oldest to latest
	this function run decision tree regression
	, output the results in a log file, and return the 
	predicted delta error col
	max_row_num means each spark program max handle row num
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
			if recursive == True and transformation == True:
				exec_file_loc = app_path + '/ml_moduel/td_decision_tree_regression_prediction_interval_log_sinh.py'
			elif recursive == False and transformation == True:
				exec_file_loc = app_path + '/ml_moduel/decision_tree_regression_transform_no_recursive_logsinh_final_test.py'
			elif recursive == True and transformation == False:
				exec_file_loc = app_path + '/ml_moduel/td_decision_tree_regression_prediction_no_transform_interval_log_sinh.py'
			elif recursive == False and transformation == False:
				exec_file_loc = app_path + '/ml_moduel/decision_tree_regression_no_transform_no_recursive_logsinh_final_test.py'
		elif transform_tech == 'boxcox':
			if recursive == True and transformation == True:
				exec_file_loc = app_path + '/ml_moduel/td_decision_tree_regression_prediction_interval_boxcox.py'
		else:
			print 'Sorry, current system does not support the input transformation technique'
		

	elif regression_technique =='glr':
		log_path = app_path + '/glr_log.txt'
		err_log_path = app_path + '/glr_err_log.txt'
		exec_file_loc = app_path + '/ml_moduel/generalized_linear_regression.py'
		result_file = app_path + '/glr_result.txt'

	elif regression_technique =='gb_tree':
		log_path = app_path + '/gbt_log.txt'
		err_log_path = app_path + '/gbt_err_log.txt'
		result_file = app_path + '/gbt_result.txt'

		if transform_tech == 'logsinh':
			if recursive == True and transformation == True:
				exec_file_loc = app_path + '/ml_moduel/td_gb_tree_regression_prediction_interval_log_sinh.py'
			elif recursive == False and transformation == True:
				exec_file_loc = app_path + '/ml_moduel/gb_tree_regression_transform_no_recursive_logsinh_final_test.py'
			elif recursive == True and transformation == False:
				exec_file_loc = app_path + '/ml_moduel/td_gb_tree_regression_prediction_no_transform_interval_log_sinh.py'
			elif recursive == False and transformation == False:
				exec_file_loc = app_path + '/ml_moduel/gb_tree_regression_no_transform_no_recursive_logsinh_final_test.py'
		else:
			print 'Sorry, current system does not support the input transformation technique'
		
	else:
		print 'Sorry, current system does not support the input regression technique'
	
	if os.path.isfile(result_file):
		# if file exist
		os.remove(result_file)

	train_file='prms_input1.csv'
	test_file='prms_input2.csv'
	split_csv_file(filename, window_per, train_file, test_file)

	# should not count header
	test_file_len = obtain_total_row_num(test_file) - 1
	if test_file_len < max_row_num:
		# training file
		delta_error_csv_train = app_path + '/temp_delta_error_train.csv'
		delta_error_filename_train = app_path + '/delta_error_train.libsvm'
		# observed_name, predicted_name = delta_error_file(filename,delta_error_filename)
		delta_error_file(train_file,delta_error_csv_train,best_alpha)
		convert_csv_into_libsvm(delta_error_csv_train,delta_error_filename_train)

		# test file
		delta_error_csv_test = app_path + '/temp_delta_error_test.csv'
		delta_error_filename_test = app_path + '/delta_error_test.libsvm'
		# observed_name, predicted_name = delta_error_file(filename,delta_error_filename)
		delta_error_file(test_file,delta_error_csv_test,best_alpha)
		convert_csv_into_libsvm(delta_error_csv_test,delta_error_filename_test)
		if recursive == True and transform_tech == 'logsinh':
			command = [spark_submit_location, exec_file_loc,delta_error_filename_train,result_file, str(window_per), str(best_alpha), app_path, str(best_a), str(best_b), delta_error_filename_test, spark_config1, spark_config2]
		elif recursive == True and transform_tech == 'boxcox':
			command = [spark_submit_location, exec_file_loc,delta_error_filename_train,result_file, str(window_per), str(best_alpha), app_path, str(best_lambda), delta_error_filename_test, spark_config1, spark_config2]
		else:
			command = [spark_submit_location, exec_file_loc,delta_error_filename_train,result_file, str(window_per), str(best_alpha), str(best_a), str(best_b), delta_error_filename_test, spark_config1, spark_config2]
		with open(log_path, 'wb') as process_out, open(log_path, 'rb', 1) as reader, open(err_log_path, 'wb') as err_out:
			process = subprocess.Popen(
				command, stdout=process_out, stderr=err_out, cwd=app_path)

		# this waits the process finishes
		process.wait()
		cur_avg_rmse = get_avg(result_file)
		print "final rmse is: "+str(cur_avg_rmse)
	else:
		tmp_dirt = 'sub_results/'
		loop_time = int(math.ceil(float(test_file_len)/max_row_num))
		# if folder does not exist create folder, if not delete
		if not os.path.exists(tmp_dirt):
			os.makedirs(tmp_dirt)
		else:
			shutil.rmtree(tmp_dirt)
			os.makedirs(tmp_dirt)

		train_file_len = obtain_total_row_num(train_file) - 1

		bound_loc = app_path+'/'+tmp_dirt+'bound.csv'
		# if os.path.isfile(bound_loc):
		# 	# if file exist
		# 	os.remove(bound_loc)

		for i in range(loop_time):
		# for i in range(1):
			tmp_test_file = tmp_dirt+'prms_test'+str(i)+'.csv'
			tmp_train_file= tmp_dirt+'prms_train'+str(i)+'.csv'
			# split files into train and test
			split_csv_file_loop(filename, i, train_file_len, max_row_num, tmp_train_file, tmp_test_file)
			# print 'current max_row_num: '+str(max_row_num)+'; current loop num: '+str(loop_time)
			# break
			delta_error_csv_train = app_path + '/temp_delta_error_train.csv'
			delta_error_filename_train = app_path + '/delta_error_train.libsvm'
			delta_error_file(tmp_train_file,delta_error_csv_train,best_alpha)
			convert_csv_into_libsvm(delta_error_csv_train,delta_error_filename_train)

			# test file
			delta_error_csv_test = app_path + '/temp_delta_error_test.csv'
			delta_error_filename_test = app_path + '/delta_error_test.libsvm'
			delta_error_file(tmp_test_file,delta_error_csv_test,best_alpha)
			convert_csv_into_libsvm(delta_error_csv_test,delta_error_filename_test)

			if recursive == True and transform_tech == 'logsinh':
				command = [spark_submit_location, exec_file_loc,delta_error_filename_train,result_file, str(window_per), str(best_alpha), app_path+'/'+tmp_dirt.replace('/',''), str(best_a), str(best_b), delta_error_filename_test, spark_config1, spark_config2]
				# sys.argv = ['ml_moduel/gb_tree_regression_no_transform_no_recursive_logsinh_final_test.py','delta_error_train.libsvm','gbt_result.txt','0.5','0.1','/home/host0/Desktop/machine_learning_prms_accuracy/tmp_test_prms','0.0405','0.0305','delta_error_test.libsvm']
			elif recursive == True and transform_tech == 'boxcox':
				command = [spark_submit_location, exec_file_loc,delta_error_filename_train,result_file, str(window_per), str(best_alpha), app_path+'/'+tmp_dirt.replace('/',''), str(best_lambda), delta_error_filename_test, spark_config1, spark_config2]
			else:
				command = [spark_submit_location, exec_file_loc,delta_error_filename_train,result_file, str(window_per), str(best_alpha), str(best_a), str(best_b), delta_error_filename_test, spark_config1, spark_config2]
			with open(log_path, 'wb') as process_out, open(log_path, 'rb', 1) as reader, open(err_log_path, 'wb') as err_out:
				process = subprocess.Popen(
					command, stdout=process_out, stderr=err_out, cwd=app_path)

			# this waits the process finishes
			process.wait()
			print str(i)+'th part of the file is processing'
			print str(loop_time-i-1)+'left for processed'

			if os.path.isfile(bound_loc):
				# if file exist
				shutil.copyfile(bound_loc,app_path+'/'+tmp_dirt+'bound'+str(i)+'.csv')

		print 'final rmse is: '+str(merge_bound_file(filename, app_path+'/'+tmp_dirt,loop_time))

	return True

def exec_regression_by_name(filename, train_file, test_file, regression_technique, window_per, best_alpha,app_path, best_a, best_b, recursive = True, transformation = True, max_row_num=500, transform_tech = 'logsinh', best_lambda = 1):
	'''
	!!!!!!!!!!!!!!!file should be ordered based on time, from oldest to latest
	this function run decision tree regression
	, output the results in a log file, and return the 
	predicted delta error col
	max_row_num means each spark program max handle row num
	window_per no use for this function.
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
			if recursive == True and transformation == True:
				exec_file_loc = app_path + '/ml_moduel/td_decision_tree_regression_prediction_interval_log_sinh.py'
			elif recursive == False and transformation == True:
				exec_file_loc = app_path + '/ml_moduel/decision_tree_regression_transform_no_recursive_logsinh_final_test.py'
			elif recursive == True and transformation == False:
				exec_file_loc = app_path + '/ml_moduel/td_decision_tree_regression_prediction_no_transform_interval_log_sinh.py'
			elif recursive == False and transformation == False:
				exec_file_loc = app_path + '/ml_moduel/decision_tree_regression_no_transform_no_recursive_logsinh_final_test.py'
		elif transform_tech == 'boxcox':
			if recursive == True and transformation == True:
				exec_file_loc = app_path + '/ml_moduel/td_decision_tree_regression_prediction_interval_boxcox.py'
		else:
			print 'Sorry, current system does not support the input transformation technique'
		

	elif regression_technique =='glr':
		log_path = app_path + '/glr_log.txt'
		err_log_path = app_path + '/glr_err_log.txt'
		exec_file_loc = app_path + '/ml_moduel/generalized_linear_regression.py'
		result_file = app_path + '/glr_result.txt'

	elif regression_technique =='gb_tree':
		log_path = app_path + '/gbt_log.txt'
		err_log_path = app_path + '/gbt_err_log.txt'
		result_file = app_path + '/gbt_result.txt'

		if transform_tech == 'logsinh':
			if recursive == True and transformation == True:
				exec_file_loc = app_path + '/ml_moduel/td_gb_tree_regression_prediction_interval_log_sinh.py'
			elif recursive == False and transformation == True:
				exec_file_loc = app_path + '/ml_moduel/gb_tree_regression_transform_no_recursive_logsinh_final_test.py'
			elif recursive == True and transformation == False:
				exec_file_loc = app_path + '/ml_moduel/td_gb_tree_regression_prediction_no_transform_interval_log_sinh.py'
			elif recursive == False and transformation == False:
				exec_file_loc = app_path + '/ml_moduel/gb_tree_regression_no_transform_no_recursive_logsinh_final_test.py'
		else:
			print 'Sorry, current system does not support the input transformation technique'
		
	else:
		print 'Sorry, current system does not support the input regression technique'
	
	if os.path.isfile(result_file):
		# if file exist
		os.remove(result_file)

	# train_file='prms_input1.csv'
	# test_file='prms_input2.csv'
	# split_csv_file(filename, window_per, train_file, test_file)

	# should not count header
	test_file_len = obtain_total_row_num(test_file) - 1
	if test_file_len < max_row_num:
		# training file
		delta_error_csv_train = app_path + '/temp_delta_error_train.csv'
		delta_error_filename_train = app_path + '/delta_error_train.libsvm'
		# observed_name, predicted_name = delta_error_file(filename,delta_error_filename)
		delta_error_file(train_file,delta_error_csv_train,best_alpha)
		convert_csv_into_libsvm(delta_error_csv_train,delta_error_filename_train)

		# test file
		delta_error_csv_test = app_path + '/temp_delta_error_test.csv'
		delta_error_filename_test = app_path + '/delta_error_test.libsvm'
		# observed_name, predicted_name = delta_error_file(filename,delta_error_filename)
		delta_error_file(test_file,delta_error_csv_test,best_alpha)
		convert_csv_into_libsvm(delta_error_csv_test,delta_error_filename_test)
		if recursive == True and transform_tech == 'logsinh':
			command = [spark_submit_location, exec_file_loc,delta_error_filename_train,result_file, str(window_per), str(best_alpha), app_path, str(best_a), str(best_b), delta_error_filename_test, spark_config1, spark_config2]
		elif recursive == True and transform_tech == 'boxcox':
			command = [spark_submit_location, exec_file_loc,delta_error_filename_train,result_file, str(window_per), str(best_alpha), app_path, str(best_lambda), delta_error_filename_test, spark_config1, spark_config2]
		else:
			command = [spark_submit_location, exec_file_loc,delta_error_filename_train,result_file, str(window_per), str(best_alpha), str(best_a), str(best_b), delta_error_filename_test, spark_config1, spark_config2]
		with open(log_path, 'wb') as process_out, open(log_path, 'rb', 1) as reader, open(err_log_path, 'wb') as err_out:
			process = subprocess.Popen(
				command, stdout=process_out, stderr=err_out, cwd=app_path)

		# this waits the process finishes
		process.wait()
		cur_avg_rmse = get_avg(result_file)
		print "final rmse is: "+str(cur_avg_rmse)
	else:
		tmp_dirt = 'sub_results/'
		loop_time = int(math.ceil(float(test_file_len)/max_row_num))
		# if folder does not exist create folder, if not delete
		if not os.path.exists(tmp_dirt):
			os.makedirs(tmp_dirt)
		else:
			shutil.rmtree(tmp_dirt)
			os.makedirs(tmp_dirt)

		train_file_len = obtain_total_row_num(train_file) - 1

		bound_loc = app_path+'/'+tmp_dirt+'bound.csv'
		# if os.path.isfile(bound_loc):
		# 	# if file exist
		# 	os.remove(bound_loc)

		for i in range(loop_time):
		# for i in range(1):
			tmp_test_file = tmp_dirt+'prms_test'+str(i)+'.csv'
			tmp_train_file= tmp_dirt+'prms_train'+str(i)+'.csv'
			# split files into train and test
			split_csv_file_loop(filename, i, train_file_len, max_row_num, tmp_train_file, tmp_test_file)
			# print 'current max_row_num: '+str(max_row_num)+'; current loop num: '+str(loop_time)
			# break
			delta_error_csv_train = app_path + '/temp_delta_error_train.csv'
			delta_error_filename_train = app_path + '/delta_error_train.libsvm'
			delta_error_file(tmp_train_file,delta_error_csv_train,best_alpha)
			convert_csv_into_libsvm(delta_error_csv_train,delta_error_filename_train)

			# test file
			delta_error_csv_test = app_path + '/temp_delta_error_test.csv'
			delta_error_filename_test = app_path + '/delta_error_test.libsvm'
			delta_error_file(tmp_test_file,delta_error_csv_test,best_alpha)
			convert_csv_into_libsvm(delta_error_csv_test,delta_error_filename_test)

			if recursive == True and transform_tech == 'logsinh':
				command = [spark_submit_location, exec_file_loc,delta_error_filename_train,result_file, str(window_per), str(best_alpha), app_path+'/'+tmp_dirt.replace('/',''), str(best_a), str(best_b), delta_error_filename_test, spark_config1, spark_config2]
				# sys.argv = ['ml_moduel/gb_tree_regression_no_transform_no_recursive_logsinh_final_test.py','delta_error_train.libsvm','gbt_result.txt','0.5','0.1','/home/host0/Desktop/machine_learning_prms_accuracy/tmp_test_prms','0.0405','0.0305','delta_error_test.libsvm']
			elif recursive == True and transform_tech == 'boxcox':
				command = [spark_submit_location, exec_file_loc,delta_error_filename_train,result_file, str(window_per), str(best_alpha), app_path+'/'+tmp_dirt.replace('/',''), str(best_lambda), delta_error_filename_test, spark_config1, spark_config2]
			else:
				command = [spark_submit_location, exec_file_loc,delta_error_filename_train,result_file, str(window_per), str(best_alpha), str(best_a), str(best_b), delta_error_filename_test, spark_config1, spark_config2]
			with open(log_path, 'wb') as process_out, open(log_path, 'rb', 1) as reader, open(err_log_path, 'wb') as err_out:
				process = subprocess.Popen(
					command, stdout=process_out, stderr=err_out, cwd=app_path)

			# this waits the process finishes
			process.wait()
			print str(i)+'th part of the file is processing'
			print str(loop_time-i-1)+'left for processed'

			if os.path.isfile(bound_loc):
				# if file exist
				shutil.copyfile(bound_loc,app_path+'/'+tmp_dirt+'bound'+str(i)+'.csv')

		print 'final rmse is: '+str(merge_bound_file(filename, app_path+'/'+tmp_dirt,loop_time))

	return True

def calculate_cor(filename, err_col='error', init_window_per=0.5):
	'''
	moving window based on cor values
	err_col: should be the error col name
	init_window_per: inital window size percentage
	'''
	window_size_list = []
	new_data_num_list = []
	df = pd.read_csv(filename)
	total_len = df.shape[0]
	# start is the lower bound of training dataset
	start = 0
	window_size = int(total_len*init_window_per)
	left = total_len - window_size
	cur_df = df[:window_size]
	# replace nan with 0 and convert all negative with abs value
	cur_list = [0 if math.isnan(x) else abs(x) for x in cur_df.corr()[err_col]]
	# abs value sum
	cur_result = sum([i for i in cur_list])
	# new_data_index is upper bound of training dataset
	new_data_index = start + window_size - 1
	for i in range(left):
		new_data_index = new_data_index + 1
		flag_inloop = False
		while tmp_result <= cur_result and start < new_data_index:
			flag_inloop = True
			start = start + 1
			cur_df = df[start:new_data_index]
			# replace nan with 0 and convert all negative with abs value
			cur_list = [0 if math.isnan(x) else abs(x) for x in cur_df.corr()[err_col]]
			# abs value sum
			cur_result = sum([m for m in cur_list])
			if tmp_result <= cur_result:
				tmp_result = cur_result
			print "current lower bound is: "+str(start)+"; current upper bound is: "+str(new_data_index)+"; total len is: "+str(new_data_index-start)
		#
		if flag_inloop: 
			start = start - 1
		window_size = new_data_index-start
		print "current lower bound is: "+str(start)+"; current upper bound is: "+str(new_data_index)+"; total len is: "+str(window_size)
		# draw graph
		window_size_list.append(window_size)
		new_data_num_list.append(i)

	print window_size_list
	fig, ax = plt.subplots()
	ax.plot(new_data_num_list,window_size_list, '-',linewidth=2, label='new_data_vs_window_size')

	plt.xlabel('new_data_id')
	plt.ylabel('window_size')
	plt.title('new_data_vs_window_size')
	plt.show()


def calculate_window_size(filename, err_col='error', init_window=30, training_per=0.5, peak_buffer=60):
	'''
	moving window based on cor values
	err_col: should be the error col name
	init_window: inital window size
	training_per: trainig data percentage
	peak_buffer: if correlation value decrease, keep search for next peak_buffer point
	'''
	# use training dataset to find best window size
	df = pd.read_csv(filename)
	total_len = df.shape[0]
	# start is the lower bound of training dataset
	# start = 0
	window_size = init_window
	cur_df = df[:window_size]
	# replace nan with 0 and convert all negative with abs value
	cur_list = [0 if math.isnan(x) else abs(x) for x in cur_df.corr()[err_col]]
	# abs value sum
	cur_result = sum([i for i in cur_list])
	# new_data_index is upper bound of training dataset
	# new_data_index = start + window_size - 1
	new_data_index = window_size - 1
	tmp_max_result = 0
	tmp_new_data_index = 0

	while new_data_index < total_len*training_per:
		flag_out = True
		new_data_index = new_data_index + 1
		cur_df = df[:new_data_index+1]
		# replace nan with 0 and convert all negative with abs value
		cur_list = [0 if math.isnan(x) else abs(x) for x in cur_df.corr()[err_col]]
		# abs value sum
		cur_result = sum([m for m in cur_list])
		if tmp_max_result <= cur_result:
			tmp_max_result = cur_result
			flag_out = False
			# print "new_data_index: "+str(new_data_index)
		else:
			for ii in range(peak_buffer):
				tmp_new_data_index = new_data_index + ii
				cur_df = df[:tmp_new_data_index]
				# replace nan with 0 and convert all negative with abs value
				cur_list = [0 if math.isnan(x) else abs(x) for x in cur_df.corr()[err_col]]
				# abs value sum
				# print "inner: "+str(ii)
				cur_result = sum([m for m in cur_list])
				if tmp_max_result <= cur_result:
					tmp_max_result = cur_result
					new_data_index = tmp_new_data_index
					flag_out = False
		if flag_out:
			break

	window_size = new_data_index + 1
	print "total len is: "+str(window_size)+" for init window size: "+str(init_window)
	return window_size

def time_vs_window_size():
	'''
	this function draws initial window size vs final window size
	'''
	filename = 'delta_prms.csv'
	err_col='error'
	# one month
	init_window=30
	training_per = 0.5
	peak_buffer = 30
	window_size_list = []
	month_list = []
	window_size_list.append(calculate_window_size(filename, err_col, init_window, training_per, peak_buffer))
	df = pd.read_csv(filename)
	total_len = int(df.shape[0] * training_per)
	i = 1
	month_list.append(i)
	while True:
		i = i + 1
		if init_window < total_len:
			# init_window = init_window * i
			init_window = init_window + 30
			month_list.append(i)
			window_size_list.append(calculate_window_size(filename, err_col, init_window, training_per, peak_buffer))
		else:
			break

	time_list = []
	start_year = 2002
	# start_month is one less than real month
	start_month = 9
	for i in month_list:
		tmp = str(start_year)+'--'+str(start_month+1)
		time_list.append(datetime.strptime(tmp, '%Y--%m'))
		start_year = start_year + (start_month+1)/12
		start_month = (start_month + 1)%12
		print str(start_year)+"--"+str(start_month+1)
	fig, ax = plt.subplots()
	ax.plot(time_list,window_size_list, '-',linewidth=2, label='new_data_vs_window_size')

	plt.xlabel('start_window_size')
	plt.ylabel('window_size')
	plt.title('time_vs_window_size')
	plt.show()


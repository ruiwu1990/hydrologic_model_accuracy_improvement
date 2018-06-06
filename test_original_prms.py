import pandas as pd
import math
from random import randint
import random
import copy


def original_model_rmse(filename, train_per, test_per, crossover_times=30):
	'''
	'''
	df = pd.read_csv(filename)
	row_len = len(df)
	train_row = int(row_len*train_per)
	test_row = row_len - train_row
	rmse = 0
	new_pred = []
	ground_truth = []
	# crossover
	for c_count in range(crossover_times):
		chosen_list = random.sample(range(row_len),test_row)
		# w = copy.deepcopy(init_w)

		training_list = [x for x in range(row_len) if x not in chosen_list]

		new_pred = []
		ground_truth = []
		# testing phase
		for r in range(test_row):

			new_pred.append(df['basin_cfs_pred'][chosen_list[r]])
			ground_truth.append(df['runoff_obs'][chosen_list[r]])
			

		rmse = rmse + get_root_mean_squared_error(new_pred,ground_truth)

	rmse = rmse/crossover_times
	print "rmse is: "+str(rmse)
	return rmse, new_pred, ground_truth



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






min_rmse,a,b = original_model_rmse('prms_input.csv',0.7,0.3)



print "pbias: "+str(get_pbias(a,b))+"; cd is: "+str(get_coeficient_determination(a,b))+"; nse is: "+str(get_nse(a,b))

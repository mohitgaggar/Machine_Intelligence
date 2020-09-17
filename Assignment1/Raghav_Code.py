import numpy as np
import pandas as pd
import random

def log_2(number):
	if(number == 0):
		return 0
	else:
		return np.log2(number)

def get_entropy_of_dataset(df):
	num_columns = len(df.columns)
	output = df[df.columns[-1]]
	num_rows = len(df)
	out = {}

	for i in df[df.columns[-1]]:
		if j not in out:
			out[j] = 1
		else:
			out[j] += 1
		

	entropy = 0

	for i in out:
		pi = (out[i] / num_rows)
		entropy += -pi*log_2(pi)

	return entropy



'''Return entropy of the attribute provided as parameter'''
	#input:pandas_dataframe,str   {i.e the column name ,ex: Temperature in the Play tennis dataset}
	#output:int/float/double/large

def get_entropy_of_attribute(df,attribute):
	num_columns = len(df.columns)
	col_= df[attribute]
	output = df[df.columns[-1]]
	num_rows = len(df)
	out = {}

	for i in range(num_rows):
		if col_[i] not in out:
			out[col_[i]] = {output[i]:1}
		else:
			if output[i] not in out[col_[i]]:
				out[col_[i]][output[i]] = 1
			else:
				out[col_[i]][output[i]] += 1

	entropy_of_attribute = 0

	for i in out:
		tot_attr = 0
		for attr in out[i]:
			tot_attr += out[i][attr]

		for attr in out[i]:
			pi = out[i][attr]/tot_attr
			entropy_of_attribute += tot_attr/num_rows * (-pi*log_2(pi))

	return entropy_of_attribute



'''Return Information Gain of the attribute provided as parameter'''
	#input:int/float/double/large,int/float/double/large
	#output:int/float/double/large

def get_information_gain(df,attribute):
	information_gain = get_entropy_of_dataset(df) - get_entropy_of_attribute(df,attribute)
	
	return information_gain



''' Returns Attribute with highest info gain'''  
	#input: pandas_dataframe
	#output: ({dict},'str')   

def get_selected_attribute(df):
	attributes = list(df.columns[:-1])
	information_gains = {}
	maxInfoGain = 0
	
	for attribute in attributes:
		information_gains[attribute] = get_information_gain(df,attribute)
		if(information_gains[attribute] > maxInfoGain):
			selected_column = attribute
			maxInfoGain = information_gains[attribute]	

	return (information_gains,selected_column)
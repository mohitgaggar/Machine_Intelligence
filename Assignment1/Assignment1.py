import numpy as np
import pandas as pd
import random

def log_2(number):
	if(number == 0):
		return 0
	else:
		return np.log2(number)


def get_entropy_of_dataset(df):
	entropy = 0
	num_columns = len(df.columns)
	num_rows = len(df)
	p = 0
	n = 0

	positive_responses = ["yes", "true", "1", "valid", "positive"]
	negative_responses = ["no", "false", "0", "invalid", "negative"]

	output = df[df.columns[-1]]   #getting the output column
	for i in output:
		out = i.lower()           # taking care of different cases in the string
		if(out in positive_responses):
			p += 1
		elif(out in negative_responses):
			n += 1
		else:
			continue             #invalid output such as missing value is discarded

	p_ratio = (p/(p+n))
	n_ratio = 1-p_ratio
	entropy = -(p_ratio)*log_2(p_ratio)-(n_ratio)*log_2(n_ratio)

	return entropy



'''Return entropy of the attribute provided as parameter'''
	#input:pandas_dataframe,str   {i.e the column name ,ex: Temperature in the Play tennis dataset}
	#output:int/float/double/large

def get_entropy_of_attribute(df,attribute):
	entropy_of_attribute = 0
	col_= df[attribute]
	output = df[df.columns[-1]]
	num_rows = len(df)
	unique_vals_in_col = list(set(col_))
	p = {}
	n = {}

	for i in unique_vals_in_col:
		p[i] = 0
		n[i] = 0
		
	positive_responses = ["yes", "true", "1", "valid", "positive"]
	negative_responses = ["no", "false", "0", "invalid", "negative"]

	for i in range(num_rows):
		out = output[i].lower()      
		if(out in positive_responses):
			p[col_[i]] += 1 
		elif(out in negative_responses):
			n[col_[i]] += 1
		else:
			continue
	
	for i in unique_vals_in_col:
		p_ratio = (p[i] / (p[i]+n[i]))
		n_ratio = 1-p_ratio
		entropy = -(p_ratio)*log_2(p_ratio) - (n_ratio)*log_2(n_ratio)
		entropy_of_attribute += ((p[i]+n[i])/(num_rows))*entropy
	
	return abs(entropy_of_attribute)



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
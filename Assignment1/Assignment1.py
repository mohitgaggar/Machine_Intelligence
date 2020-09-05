import numpy as np
import pandas as pd
import random

def log_2(number):
	if(number == 0):
		return 0
	else:
		return np.log2(number)

def get_ouput_attribute(df):
	last=df.columns[-1]
	vals={}
	for i in set(df[last]):
		vals[i]=0
	return vals

def get_entropy_of_dataset(df):
	entropy = 0
	num_columns = len(df.columns)
	num_rows = len(df)
	p = 0
	n = 0
	responses=get_ouput_attribute(df)

	output = df[df.columns[-1]]   #getting the output column
	for i in output:
		responses[i]+=1
	entropy=0
	for i in responses.keys():
		ratio=responses[i]/num_rows
		entropy-=ratio*log_2(ratio)

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
	multiple_responses={}
	responses=get_ouput_attribute(df) # returns the output values types
	for i in unique_vals_in_col:
		multiple_responses[i]=responses.copy()  # create a copy of responses dict for each attribute value

	for i in range(num_rows):
		multiple_responses[col_[i]][output[i]]+=1  # find out the number of reponses for each kind for each attribute value

	for j in multiple_responses.keys():
		entropy=0
		num_of_values_in_attribute=sum(list(multiple_responses[j].values()))
		for i in responses.keys():
			ratio=multiple_responses[j][i]/num_of_values_in_attribute
			entropy-=ratio*log_2(ratio)  # calculate entropy of individual attribute value
		entropy_of_attribute += (num_of_values_in_attribute/num_rows)*entropy # calculate entoropy of whole attribute
	
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
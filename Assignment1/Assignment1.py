'''
Assume df is a pandas dataframe object of the dataset given
'''
import numpy as np
import pandas as pd
import random

'''Calculate the entropy of the enitre dataset'''
	#input:pandas_dataframe
	#output:int/float/double/large

# function to return the log based 2 of a number ,workaround for log(0) is returning 0

def log_2(number):
	if(number==0):
		return 0
	else:
		return np.log2(number)


def get_entropy_of_dataset(df):
	entropy = 0
	num_columns=len(df.columns)
	num_rows=len(df)
	p=0
	n=0

	output = df[df.columns[-1]]
	for i in output:
		out = i.lower()
		if(out == "yes" or out == "true" or out == "1"):
			p+=1
		elif(out == "no" or out == "false" or out == "0"):
			n+=1
		else:
			continue

	p_ratio=(p/(p+n))
	n_ratio=1-p_ratio
	entropy=-(p_ratio)*log_2(p_ratio)-(n_ratio)*log_2(n_ratio)
	return entropy



'''Return entropy of the attribute provided as parameter'''
	#input:pandas_dataframe,str   {i.e the column name ,ex: Temperature in the Play tennis dataset}
	#output:int/float/double/large


def get_entropy_of_attribute(df,attribute):
	entropy_of_attribute = 0
	col_=df[attribute]
	output = df[df.columns[-1]]
	num_rows=len(df)
	unique_vals_in_col=list(set(col_))
	p={}
	n={}

	for i in unique_vals_in_col:
		p[i]=0
		n[i]=0
	
	for i in range(num_rows):
		out = output[i].lower()
		if(out == "yes" or out == "true" or out == "1"):
			p[col_[i]]+=1 
		elif(out == "no" or out == "false" or out == "0"):
			n[col_[i]]+=1
		else:
			continue
	
	for i in unique_vals_in_col:
		p_ratio=(p[i]/(p[i]+n[i]))
		n_ratio=1-p_ratio
		entropy=-(p_ratio)*log_2(p_ratio)-(n_ratio)*log_2(n_ratio)
		entropy_of_attribute+=((p[i]+n[i])/(num_rows))*entropy
	
	return abs(entropy_of_attribute)



'''Return Information Gain of the attribute provided as parameter'''
	#input:int/float/double/large,int/float/double/large
	#output:int/float/double/large
def get_information_gain(df,attribute):
	information_gain = get_entropy_of_dataset(df)-get_entropy_of_attribute(df,attribute)
	
	return information_gain



''' Returns Attribute with highest info gain'''  
	#input: pandas_dataframe
	#output: ({dict},'str')     
def get_selected_attribute(df):
	attributes=list(df.columns[:-1])
	information_gains={}
	maxInfoGain=0
	
	for attribute in attributes:
		information_gains[attribute]=get_information_gain(df,attribute)
		if(information_gains[attribute] > maxInfoGain):
			selected_column=attribute
			maxInfoGain=information_gains[attribute]	

	'''
	Return a tuple with the first element as a dictionary which has IG of all columns 
	and the second element as a string with the name of the column selected

	example : ({'A':0.123,'B':0.768,'C':1.23} , 'C')
	'''
	# print(information_gains,selected_column)

	return (information_gains,selected_column)



'''
------- TEST CASES --------
How to run sample test cases ?

Simply run the file DT_SampleTestCase.py
Follow convention and do not change any file / function names

'''
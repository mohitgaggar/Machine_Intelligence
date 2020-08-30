'''
Assume df is a pandas dataframe object of the dataset given
'''
import numpy as np
import pandas as pd
import random

'''Calculate the entropy of the enitre dataset'''
	#input:pandas_dataframe
	#output:int/float/double/large

def get_entropy(p, n):
	
	ent = ((-p/(p+n)*np.log2(p/(p+n))) if p != 0 else 0) \
			- ((n/(n+p)*np.log2(n/(n+p))) if n != 0 else 0)

	return ent

def get_entropy_of_dataset(df):

	p = 0
	n = 0

	for i in df.iloc[:, -1]:
		if(i == "yes" or i == 1):
			p+=1
		else:
			n+=1

	entropy = get_entropy(p,n)

	return entropy



'''Return entropy of the attribute provided as parameter'''
	#input:pandas_dataframe,str   {i.e the column name ,ex: Temperature in the Play tennis dataset}
	#output:int/float/double/large
def get_entropy_of_attribute(df,attribute):
	
	num_rows = len(df)
	vals = {}
	for i in range(num_rows):
		if(df[attribute][i] not in vals):
			vals[df[attribute][i]] = [0,0]
			if(df.iloc[:, -1][i] == "yes" or df.iloc[:, -1][i] == 1):
				vals[df[attribute][i]][0]+=1
			else:
				vals[df[attribute][i]][1]+=1
		else:
			if(df.iloc[:, -1][i] == "yes" or df.iloc[:, -1][i] == 1):
				vals[df[attribute][i]][0]+=1
			else:
				vals[df[attribute][i]][1]+=1

	entropy_of_attribute = 0

	for val in vals:
		p = vals[val][0]
		n = vals[val][1]
		entropy_of_attribute += ((p+n) * get_entropy(p,n)) / num_rows

	return abs(entropy_of_attribute)



'''Return Information Gain of the attribute provided as parameter'''
	#input:int/float/double/large,int/float/double/large
	#output:int/float/double/large
def get_information_gain(df,attribute):
	
	information_gain = get_entropy_of_dataset(df) - get_entropy_of_attribute(df, attribute)

	return information_gain



''' Returns Attribute with highest info gain'''  
	#input: pandas_dataframe
	#output: ({dict},'str')     
def get_selected_attribute(df):
   
	information_gains={}
	selected_column=''

	for column in df.iloc[:,:-1].columns:
		information_gains[column] = get_information_gain(df, column)


	maxInfGain = 0
	for col in information_gains:
		if(information_gains[col] > maxInfGain):
			maxInfGain = information_gains[col]
			selected_column = col
	'''
	Return a tuple with the first element as a dictionary which has IG of all columns 
	and the second element as a string with the name of the column selected

	example : ({'A':0.123,'B':0.768,'C':1.23} , 'C')
	'''

	return (information_gains,selected_column)



'''
------- TEST CASES --------
How to run sample test cases ?

Simply run the file DT_SampleTestCase.py
Follow convention and do not change any file / function names

'''

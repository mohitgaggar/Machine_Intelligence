#!/usr/bin/python3

'''
Design of a Neural Network from scratch

*************<IMP>*************
Mention hyperparameters used and describe functionality in detail in this space
- carries 1 mark
'''

import pandas as pd
import numpy as np

class NN:

	''' X and Y are dataframes '''
	def __init__(self):
		pass

	def fit(self,X,Y):
		pass
		'''
		Function that trains the neural network by taking x_train and y_train samples as input
		'''
	
	def predict(self,X):
		pass
		"""
		The predict function performs a simple feed forward of weights
		and outputs yhat values 

		yhat is a list of the predicted value for df X
		"""
		
		return yhat

	def CM(y_test,y_test_obs):
		'''
		Prints confusion matrix 
		y_test is list of y values in the test dataset
		y_test_obs is list of y values predicted by the model

		'''

		for i in range(len(y_test_obs)):
			if(y_test_obs[i]>0.6):
				y_test_obs[i]=1
			else:
				y_test_obs[i]=0
		
		cm=[[0,0],[0,0]]
		fp=0
		fn=0
		tp=0
		tn=0
		
		for i in range(len(y_test)):
			if(y_test[i]==1 and y_test_obs[i]==1):
				tp=tp+1
			if(y_test[i]==0 and y_test_obs[i]==0):
				tn=tn+1
			if(y_test[i]==1 and y_test_obs[i]==0):
				fp=fp+1
			if(y_test[i]==0 and y_test_obs[i]==1):
				fn=fn+1
		cm[0][0]=tn
		cm[0][1]=fp
		cm[1][0]=fn
		cm[1][1]=tp

		p= tp/(tp+fp)
		r=tp/(tp+fn)
		f1=(2*p*r)/(p+r)
		
		print("Confusion Matrix : ")
		print(cm)
		print("\n")
		print(f"Precision : {p}")
		print(f"Recall : {r}")
		print(f"F1 SCORE : {f1}")



def clean_data(data):
	replace_mean = ['Age','Weight','HB','BP']
	for col in replace_mean:
	    data[col].fillna(data[col].mean(),inplace = True)

	# Delivery Phase,IFA,Residence with forward fill 
	forward_fill = ['Delivery phase','IFA' ,'Residence']
	for col in forward_fill:
	    data[col].ffill(axis = 0,inplace=True)

	# Community,Education with mode
	replace_mode = ['Community','Education','Result']
	mode=[data.mode()[col][0] for col in replace_mode]
	for i in range(len(replace_mode)):
	    data[replace_mode[i]].fillna(mode[i],inplace = True)

	# Normalising the data 
	normal_col = ["Age", "Weight", "HB", "BP"]
	for column in normal_col:
	    max_val = max(data[column])
	    min_val = min(data[column])

	    data[column] = (data[column]-min_val)/(max_val-min_val)

	return data


def sigmoid(x):
	return 1/(1+np.exp(-x))

data = pd.read_csv('LBW_Dataset.csv')
data = clean_data(data)
data = data.to_numpy()
# print(type(data))
# 96, 10

weight1 = np.random.uniform(-1.5, 1.5, 10)
data2 = np.dot(data,weight1)
data2 = sigmoid(data2)
# print(data2)

weight2 = np.random.uniform(-1.5, 1.5, 96)
yhat = np.dot(weight2, data2)
yhat = sigmoid(yhat)
print(yhat)


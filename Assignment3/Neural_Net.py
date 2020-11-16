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
	# def main(self):
	# 	data=

	def __init__(self):
		self.epochs=100
		self.learning_rate=0.01
		self.number_of_hidden_layers=3
		self.number_of_input_features=9
		self.y=np.zeros(((self.number_of_hidden_layers+1),self.number_of_input_features))
		#Activation function applied 
		self.oy=np.zeros(((self.number_of_hidden_layers+1),self.number_of_input_features))
		self.activation_function=['relu']*self.number_of_hidden_layers + ['sigmoid']
		self.w=np.zeros((self.number_of_hidden_layers+1,self.number_of_input_features,self.number_of_input_features))
		self.b=np.zeros((self.number_of_hidden_layers+1,self.number_of_input_features))
		self.partial_der_intermediate_outputs=np.zeros((self.number_of_hidden_layers + 1, self.number_of_input_features))

			
	def derivative_relu(self,x):
		return 1 if x > 0 else 0


	def sigmoid(self,x):
		return 1/(1+np.exp(-x))

	def activate(self,func_index,number):
		if(self.activation_function[func_index]=='sigmoid'):
			f=self.sigmoid
		elif(self.activation_function[func_index]=='relu'):
			f=self.relu
		return f(number)

	def derivative_sigmoid(self,x):
		return x*(1-x)

	def relu(self,x):
		return np.maximum(0,x)

	def derivative_function(self,func_index,out):
		if(self.activation_function[func_index]=='sigmoid'):
			f=self.derivative_sigmoid
		elif(self.activation_function[func_index]=='relu'):
			f=self.derivative_relu
		return f(out)


	def forward_prop(self,inp):
		# print(inp)
		number_of_input_features=self.number_of_input_features
		number_of_hidden_layers=self.number_of_hidden_layers
		#First hidden layer
		for j in range(self.number_of_input_features):
			for k in range(self.number_of_input_features):
				self.y[0][k]+=inp[j]*self.w[0][j][k]

		
		for j in range(self.number_of_input_features):
			self.y[0][j]+=self.b[0][j]
			self.oy[0][j]=self.activate(0,self.y[0][j])

		for i in range(1,self.number_of_hidden_layers):
			for j in range(self.number_of_input_features):
				for k in range(self.number_of_input_features):
					self.y[i][k]+=self.oy[i-1][j]*self.w[i][j][k]

			for j in range(self.number_of_input_features):
				self.y[i][j]+=self.b[i][j]
				self.oy[i][j]=self.activate(i,self.y[i][j])
		
		for j in range(self.number_of_input_features):
			self.y[number_of_hidden_layers][0]+=self.oy[number_of_hidden_layers - 1][j] * self.w[number_of_hidden_layers][j][0]

		self.oy[number_of_hidden_layers][0]=self.activate(self.number_of_hidden_layers , self.y[self.number_of_hidden_layers][0])

		# print(self.y)


	def Partial_der(self, actual_output):
		# der_loss_wrt_output=oy[number_of_hidden_layers][0] - actual_output    # using 1/2(y^  -  y)**2
		number_of_hidden_layers=self.number_of_hidden_layers
		number_of_input_features=self.number_of_input_features

		calculated_output=self.oy[number_of_hidden_layers][0]
		der_loss_wrt_output= ((1-actual_output)/(1-calculated_output)) - (actual_output/calculated_output)      # using binary cross entropy
		
		self.partial_der_intermediate_outputs[number_of_hidden_layers][0]=der_loss_wrt_output * self.derivative_function(number_of_hidden_layers,self.oy[number_of_hidden_layers][0])
		
		for j in range(number_of_input_features):
			self.partial_der_intermediate_outputs[number_of_hidden_layers - 1][j]=self.partial_der_intermediate_outputs[number_of_hidden_layers][0] * self.w[number_of_hidden_layers-1][j][0]* self.derivative_function(number_of_hidden_layers-1,self.oy[number_of_hidden_layers - 1][j])    
		
		for i in range(number_of_hidden_layers-2,-1,-1):
			for j in range(number_of_input_features):
				for k in range(number_of_input_features):
					self.partial_der_intermediate_outputs[i][j]+= self.partial_der_intermediate_outputs[i+1][k] * self.w[i][j][k] * self.derivative_function(i,self.oy[i][j])

		# print(partial_der_intermediate_outputs)



	def back_prop(self,actual_output, inp):
		
		number_of_hidden_layers=self.number_of_hidden_layers
		number_of_input_features=self.number_of_input_features

		nw=np.zeros((number_of_hidden_layers+1,number_of_input_features,number_of_input_features))
		nb=np.zeros((number_of_hidden_layers+1,number_of_input_features))

		
		self.Partial_der(actual_output)
		
		for j in range(number_of_input_features):
			for k in range(number_of_input_features):
				nw[0][j][k]=self.w[0][j][k]-self.learning_rate*self.partial_der_intermediate_outputs[0][k]*inp[j]
					


		for i in range(1,number_of_hidden_layers):
			for j in range(number_of_input_features):
				for k in range(number_of_input_features):
					nw[i][j][k]=self.w[i][j][k]-self.learning_rate*self.partial_der_intermediate_outputs[i][k]*self.oy[i-1][j]

		for j in range(number_of_input_features):
			nw[number_of_hidden_layers][j][0]=self.w[number_of_hidden_layers][j][0]-self.learning_rate*self.partial_der_intermediate_outputs[number_of_hidden_layers][0]*self.oy[number_of_hidden_layers-1][j]
		
		for i in range(number_of_hidden_layers):
			for j in range(number_of_input_features):
				nb[i][j]=self.b[i][j] - self.learning_rate * self.partial_der_intermediate_outputs[i][j]
					
		for j in range(number_of_input_features):
			nb[number_of_hidden_layers][j]=self.b[number_of_hidden_layers][j] - self.learning_rate * self.partial_der_intermediate_outputs[number_of_hidden_layers][0]

		self.b=nb
		self.w=nw

		
	def fit(self,X,Y):
		'''
		Function that trains the neural network by taking x_train and y_train samples as input
		'''
		
		train_data=X.to_numpy()
		y_train=Y.to_numpy()

		np.random.shuffle(train_data)

		for _ in range(self.epochs):
			for i in range(len(train_data)):
				actual_output=y_train[i]

				self.forward_prop(train_data[i])    
				self.back_prop(y_train[i],train_data[i])   

	
	def predict(self,X):

		"""
		The predict function performs a simple feed forward of weights
		and outputs yhat values 

		yhat is a list of the predicted value for df X
		
		"""
		x_test=X.to_numpy()
		yhat=[]
		number_of_hidden_layers=self.number_of_hidden_layers
		number_of_input_features=self.number_of_input_features
		for i in range(len(x_test)):
			inp=x_test[i]

			y=np.zeros(((self.number_of_hidden_layers+1),self.number_of_input_features))

			oy=np.zeros(((self.number_of_hidden_layers+1),self.number_of_input_features))
			
			for j in range(self.number_of_input_features):
				for k in range(self.number_of_input_features):
					y[0][k]+=inp[j]*self.w[0][j][k]

			
			for j in range(self.number_of_input_features):
				y[0][j]+=self.b[0][j]
				oy[0][j]=self.activate(0,y[0][j])

			for i in range(1,self.number_of_hidden_layers):
				for j in range(self.number_of_input_features):
					for k in range(self.number_of_input_features):
						y[i][k]+=oy[i-1][j]*self.w[i][j][k]

				for j in range(self.number_of_input_features):
					y[i][j]+=self.b[i][j]
					oy[i][j]=self.activate(i,y[i][j])
			
			for j in range(self.number_of_input_features):
				y[number_of_hidden_layers][0]+=oy[number_of_hidden_layers - 1][j] * self.w[number_of_hidden_layers][j][0]

			oy[number_of_hidden_layers][0]=self.activate(self.number_of_hidden_layers , y[self.number_of_hidden_layers][0])
			
			yhat.append(oy[number_of_hidden_layers][0])

			# print(oy)
		# print(yhat)
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
			

	def main(self):
		data=pd.read_csv(r'LBW_Dataset.csv')

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

		# normalizing
		cols=list(data.columns)
		cols.remove('Education')
		for column in cols:
			max_val = max(data[column])
			min_val = min(data[column])
			data[column] = (data[column]-min_val)/(max_val-min_val)
		data['Education']=1.0
		# print(data.head())
		factor=0.7
		index_split=int(factor * len(data))

		x_train=data.iloc[:index_split,:-1] 
		y_train=data.iloc[:index_split,-1:] 
		# test=data.iloc[index_split:,:]
		x_test=data.iloc[index_split:,:-1] 
		y_test=data.iloc[index_split:,-1:] 

		self.fit(x_train,y_train)
		# print(self.w)
		yhat=self.predict(x_test)

		print(yhat)
		# print()
		
nn=NN()
nn.main()


	



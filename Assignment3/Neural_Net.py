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

	# def __init__(self):
	# 	self.epochs=100
	# 	self.learning_rate=0.01
	# 	self.number_of_hidden_layers=3
	# 	self.number_of_input_features=9
	# 	self.y=np.zeros(((self.number_of_hidden_layers+1),self.number_of_input_features))
	# 	#Activation function applied 
	# 	self.oy=np.zeros(((self.number_of_hidden_layers+1),self.number_of_input_features))
	# 	self.activation_function=['relu']*self.number_of_hidden_layers + ['sigmoid']
	# 	self.w=np.zeros((self.number_of_hidden_layers+1,self.number_of_input_features,self.number_of_input_features))
	# 	self.b=np.zeros((self.number_of_hidden_layers+1,self.number_of_input_features))
	# 	self.partial_der_intermediate_outputs=np.zeros((self.number_of_hidden_layers + 1, self.number_of_input_features))

			
	
		
	def fit(self,X,Y):
		'''
		Function that trains the neural network by taking x_train and y_train samples as input
		'''
				
		def derivative_relu(x):
			return 1 if x > 0 else 0


		def sigmoid(x):
			return 1/(1+np.exp(-x))

		def activate(func,number):
			if(func=='sigmoid'):
				f=sigmoid
			elif(func=='relu'):
				f=relu
			return f(number)

		def derivative_sigmoid(x):
			return x*(1-x)

		def relu(x):
			return np.maximum(0,x)

		def derivative_function(func,out):
			if(func=='sigmoid'):
				f=derivative_sigmoid
			elif(func=='relu'):
				f=derivative_relu
			return f(out)

		def forward_prop(inp,activation_function,w,b):

			#Summed output at each layer
			y=np.zeros(((number_of_hidden_layers+1),number_of_input_features))
			#Activation function applied 
			oy=np.zeros(((number_of_hidden_layers+1),number_of_input_features))

			#First hidden layer
			for j in range(number_of_input_features):
				for k in range(number_of_input_features):
					y[0][k]+=inp[j]*w[0][j][k]

			
			for j in range(number_of_input_features):
				y[0][j]+=b[0][j]
				oy[0][j]=activate(activation_function[0],y[0][j])

			for i in range(1,number_of_hidden_layers):
				for j in range(number_of_input_features):
					for k in range(number_of_input_features):
						y[i][k]+=oy[i-1][j]*w[i][j][k]

				for j in range(number_of_input_features):
					y[i][j]+=b[i][j]
					oy[i][j]=activate(activation_function[i],y[i][j])
			
			for j in range(number_of_input_features):
				y[number_of_hidden_layers][0]+=oy[number_of_hidden_layers - 1][j] * w[number_of_hidden_layers][j][0]

			oy[number_of_hidden_layers][0]=activate(activation_function[number_of_hidden_layers],y[number_of_hidden_layers][0])

			return y,oy

		# print(partial_der_intermediate_outputs)

		def Partial_der(partial_der_intermediate_outputs , actual_output ,activation_functions , oy ,w):
			# der_loss_wrt_output=oy[number_of_hidden_layers][0] - actual_output    # using 1/2(y^  -  y)**2

			calculated_output=oy[number_of_hidden_layers][0]
			der_loss_wrt_output= ((1-actual_output)/(1-calculated_output)) - (actual_output/calculated_output)      # using binary cross entropy
			
			partial_der_intermediate_outputs[number_of_hidden_layers][0]=der_loss_wrt_output * derivative_function(activation_functions[number_of_hidden_layers],oy[number_of_hidden_layers][0])
			
			for j in range(number_of_input_features):
				partial_der_intermediate_outputs[number_of_hidden_layers - 1][j]=partial_der_intermediate_outputs[number_of_hidden_layers][0] * w[number_of_hidden_layers-1][j][0]* derivative_function(activation_functions[number_of_hidden_layers-1],oy[number_of_hidden_layers - 1][j])    
			
			for i in range(number_of_hidden_layers-2,-1,-1):
				for j in range(number_of_input_features):
					for k in range(number_of_input_features):
						partial_der_intermediate_outputs[i][j]+= partial_der_intermediate_outputs[i+1][k] * w[i][j][k] * derivative_function(activation_function[i],oy[i][j])

			# print(partial_der_intermediate_outputs)



		def back_prop(w , b , actual_output , activation_functions ,oy , learning_rate , inp):
			# print("HI")
			nw=np.zeros((number_of_hidden_layers+1,number_of_input_features,number_of_input_features))
			nb=np.zeros((number_of_hidden_layers+1,number_of_input_features))

			partial_der_intermediate_outputs=np.zeros((number_of_hidden_layers + 1, number_of_input_features))
			Partial_der(partial_der_intermediate_outputs , actual_output , activation_functions ,oy ,w)
			
			
			for j in range(number_of_input_features):
				for k in range(number_of_input_features):
					nw[0][j][k]=w[0][j][k]-learning_rate*partial_der_intermediate_outputs[0][k]*inp[j]
						


			for i in range(1,number_of_hidden_layers):
				for j in range(number_of_input_features):
					for k in range(number_of_input_features):
						nw[i][j][k]=w[i][j][k]-learning_rate*partial_der_intermediate_outputs[i][k]*oy[i-1][j]

			for j in range(number_of_input_features):
				nw[number_of_hidden_layers][j][0]=w[number_of_hidden_layers][j][0]-learning_rate*partial_der_intermediate_outputs[number_of_hidden_layers][0]*oy[number_of_hidden_layers-1][j]
			
			for i in range(number_of_hidden_layers):
				for j in range(number_of_input_features):
					nb[i][j]=b[i][j] - learning_rate * partial_der_intermediate_outputs[i][j]
						
			for j in range(number_of_input_features):
				nb[number_of_hidden_layers][j]=b[number_of_hidden_layers][j] - learning_rate * partial_der_intermediate_outputs[number_of_hidden_layers][0]


			# print(b,nb)
			# print("HIHIHIH",nw[-1][0])

			# print(w[-1][1],nw[-1][1] ,sep="\n")
			return nw,nb
			# b=nb.copy()
			# w=nw.copy()

		# print(weight)

		# print(bias)

		# weight matrix dimensions (Number_of_hidden_layers+1) * 9 * 9
		# bias matrix dimansions (Number_of_hidden_layers+1) * 1
		# expected output is the given output for the row
		def loss_function(yhat,y):
			return -1 * (y*np.log(yhat) + (1-y)*np.log(1-yhat))




		def train(inp,weight,bias ,all_errors):
			# print(weight[-1][0])
			actual_output=inp[-1]
			inp=inp[:-1]
			# print(len(inp))

			y,oy=forward_prop(inp , activation_function , weight , bias )    
			# forward_prop is a function that takes in the input to the layer, activation function , entire weight, bias 
			# corresponding weight and bias for the layer can be calculated by using the layer_index 
			# forward_prop should call itself recursively and calculate final output and return the final output along with all intermediate outputs(oy) and intermediate summer outputs(y) 
			# dimensions of oy and y are (number_of_hidden_layers+1) X (number_of_input_features)
			# print(weight[-1][0])

			# y=np.random.rand((number_of_hidden_layers+1),number_of_input_features)
			# oy=np.random.rand((number_of_hidden_layers+1),number_of_input_features)
		
			# error=loss_function(oy[-1][0],actual_output)
			# all_errors.append(error)
			# # print(error,"is error")

			# # print("POP    ",oy[-1][0]," ",np.round(oy[-1][0])," ",actual_output)
			# if(actual_output==1.0):
			# 	# print("HI")
			# 	if(np.round(oy[-1][0]) == actual_output):
			# 		metric.tp+=1
			# 		# print("1")
			# 	else:
			# 		metric.fn+=1
			# else:
			# 	# print("YO")
			# 	if(np.round(oy[-1][0]) == actual_output):
			# 		metric.tn+=1
			# 		# print("1")
			# 	else:
			# 		metric.fp+=1
					# print("2")

			learning_rate=0.005
			weight,bias=back_prop(weight , bias , actual_output , activation_function, oy ,learning_rate , inp)   
			
			# back_prop is a function that takes in the weight and the bias matrices , the error in output , activation_func , layer_index
			# starts from the last layer and recursively calls itself till the input layer (corresponding activation_function to be passed)
			# print(weight[-1][0],bias[0])
			return weight,bias
				
		number_of_hidden_layers=5
		number_of_input_features=9

		# activation_function=['relu'] * (number_of_hidden_layers) + ['sigmoid']
		activation_function=['sigmoid'] * (number_of_hidden_layers+1) 

		# weight=np.random.rand(number_of_hidden_layers + 1,number_of_input_features,number_of_input_features)
		weight=np.random.rand(number_of_hidden_layers + 1,number_of_input_features,number_of_input_features)
		bias=np.random.rand(number_of_hidden_layers + 1 , number_of_input_features)
		# len(tonp)
		# np.random.shuffle(tonp)
		# split_factor=0.7
		# number_in_train=int(len(tonp)*split_factor)
		# train_data=tonp[:number_in_train]
		# test_data=tonp[number_in_train:]
		# trainmetric=metrics()
		# testmetric=metrics()
		# print(tonp[0])
		# print(trainmetric)

		epochs=500
		tr=pd.concat([X, Y], axis=1)
		train_data=tr.to_numpy()

		for _ in range(epochs):
			all_errors=[]
			for i in range(len(train_data)):
				weight,bias=train(train_data[i],weight,bias,all_errors)
		self.w=weight
		self.b=bias
		self.activation_function=activation_function
		self.number_of_hidden_layers=number_of_hidden_layers
		self.number_of_input_features=number_of_input_features

	
	def predict(self,X):

		"""
		The predict function performs a simple feed forward of weights
		and outputs yhat values 

		yhat is a list of the predicted value for df X
		
		"""
		# activation_function=[]
		test_data=X.to_numpy()
		number_of_hidden_layers=self.number_of_hidden_layers
		number_of_input_features=self.number_of_input_features
				
		def derivative_relu(x):
			return 1 if x > 0 else 0


		def sigmoid(x):
			return 1/(1+np.exp(-x))

		def activate(func,number):
			if(func=='sigmoid'):
				f=sigmoid
			elif(func=='relu'):
				f=relu
			return f(number)

		def derivative_sigmoid(x):
			return x*(1-x)

		def relu(x):
			return np.maximum(0,x)

		def derivative_function(func,out):
			if(func=='sigmoid'):
				f=derivative_sigmoid
			elif(func=='relu'):
				f=derivative_relu
			return f(out)

		
		def forward_prop(inp,activation_function,w,b):

			#Summed output at each layer
			y=np.random.rand((number_of_hidden_layers+1),number_of_input_features)
			#Activation function applied 
			oy=np.random.rand((number_of_hidden_layers+1),number_of_input_features)

			#First hidden layer
			for j in range(number_of_input_features):
				for k in range(number_of_input_features):
					y[0][k]+=inp[j]*w[0][j][k]

			
			for j in range(number_of_input_features):
				y[0][j]+=b[0][j]
				oy[0][j]=activate(activation_function[0],y[0][j])

			for i in range(1,number_of_hidden_layers):
				for j in range(number_of_input_features):
					for k in range(number_of_input_features):
						y[i][k]+=oy[i-1][j]*w[i][j][k]

				for j in range(number_of_input_features):
					y[i][j]+=b[i][j]
					oy[i][j]=activate(activation_function[i],y[i][j])
			
			for j in range(number_of_input_features):
				y[number_of_hidden_layers][0]+=oy[number_of_hidden_layers - 1][j] * w[number_of_hidden_layers][j][0]

			oy[number_of_hidden_layers][0]=activate(activation_function[number_of_hidden_layers],y[number_of_hidden_layers][0])

			return y,oy

		

		def test(inp,activation_function , w , b,arr):
			y,oy=forward_prop(inp, activation_function , w , b )  
			arr.append(oy[-1][0])
		
		arr=[]
		for i in range(len(test_data)):
			test(test_data[i],self.activation_function,self.w,self.b,arr)
		
		return arr


	def CM(self,y_test,y_test_obs):
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
		a=(tn+tp)/(tp+tn+fp+fn)
		print("Confusion Matrix : ")
		print(cm)
		print("\n")
		print(f"Precision : {p}")
		print(f"Recall : {r}")
		print(f"F1 SCORE : {f1}")
		print(f"Accuracy {a}")
			

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
		# print(data.columns)
		data=data.sample(frac=1)
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
		# yTest=y_test.values.tolist()
		yTest=y_test[y_test.columns[-1]].tolist()
		# print(len(y_test),len(yhat))
		# print(yTest,"\n",yhat)
		self.CM(yTest,yhat)
		# print(self.w)
		# print(yhat)
		# print(self.activate(self.number_of_hidden_layers,12))
		# print()
		
nn=NN()
nn.main()


	



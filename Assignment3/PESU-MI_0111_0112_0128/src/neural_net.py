'''
Design of a Neural Network from scratch

*************<IMP>*************
Mention hyperparameters used and describe functionality in detail in this space
#Implementation
The first step in predicting values for the dataset is to pass the neurons through a feedforward function.
Here, matrices 'y' computes the weighted inputs of each neuron added to bias. We perform matrix multiplication between 'input' 
and weight matrix 'w', after which we perform the addition of the bias matrix 'b' to this product to give us 'y'.
'oy' calculates the value of the neuron once the activation function is applied to each neuron in the layer. 
For each cell in the matrix 'y', we execute the corresponding activation function to give us the final matrix 'oy'.

Backpropagation is performed to help us tune the weights and bias to get a better fit.  
We have matrices 'nw' and 'nb' that get updated. We compute the partial derivatives of the outputs from each layer 
by using chain rule. We first calculate the output layer error and pass the result to the hidden layer before it. 
After calculating the hidden layer error, we pass its error value back to the previously hidden layer before it.
This means that we keep applying the derivative of activation function on each intermediate output, and multiply 
with the corresponding weight of the layer, and the prior calculated partial derivative.

Training of the neural network takes place by applying a feedforward on the input neurons and improving the model 
with backpropagation by tuning the weights.

#Hyperparameters
We initialize the weight matrix as a three-dimensional matrix, of zeroes whose dimensions are 
(number of hidden layers + 1) * (number of inputs * number of inputs). 
The bias matrix is initialized with ones. Its dimensions are (number of hidden layers + 1) * (number of inputs).
The number of hidden layers is 4.
We chose a learning rate of 0.01.
We compute over 100 epochs. Relu activation function is applied to all the hidden layers, 
and sigmoid function is applied to the final layer for classification.

'''

import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

class NN:

	#Class variables
	num_hidden_layers=4
	num_input_features=9

	# activation_function is a dictionary specifying which function is applied to the given layer number
	# we apply sigmoid to final layer before prediction
	activation_function=['relu'] * (num_hidden_layers) + ['sigmoid']

	#Class Methods
	def activate(self,func,number):
		if(func=='sigmoid'):
				return 1/(1+np.exp(-number))

		elif(func=='relu'):
				return max(0,number)

	def derivative_function(self,func,out):
			if(func=='sigmoid'):
				return out*(1-out)
			elif(func=='relu'):
				return 1 if out>0 else 0

		
	'''
	forward_prop is a function that takes in the input to the layer, activation function , entire weight, bias 
	corresponding weight and bias for the layer can be calculated by using the layer_index 
	It does forward propogation and calculates final output and return the final output 
	along with all intermediate outputs(oy) and intermediate summer outputs(y)
	''' 
	def forward_prop(self,inp,activation_function,w,b): 		#Passing a single row at a time

		num_hidden_layers = self.num_hidden_layers
		num_input_features = self.num_input_features
		activation_function = self.activation_function

		y=np.zeros(((num_hidden_layers+1),num_input_features))
		oy=np.zeros(((num_hidden_layers+1),num_input_features))
	
		# We do matrix multiplication where each cell in y is computed
		# by summing product of the input neuron and weight of the corresponding cell in weight matrix
		# And then adding the bias
		
		#For first layer
		for j in range(num_input_features):
			for k in range(num_input_features):  #Multiplying with weight matrix
				y[0][k]+=inp[j]*w[0][j][k]

		for j in range(num_input_features):		 #Adding the bias
			y[0][j]+=b[0][j]
			oy[0][j]=self.activate(activation_function[0],y[0][j])


		#For all hidden layers
		for i in range(1,num_hidden_layers):
			for j in range(num_input_features):
				for k in range(num_input_features):
					y[i][k]+=oy[i-1][j]*w[i][j][k]
			for j in range(num_input_features):
				y[i][j]+=b[i][j]
				oy[i][j]=self.activate(activation_function[i],y[i][j])
		

		#For output layer
		for j in range(num_input_features):
			y[num_hidden_layers][0]+=oy[num_hidden_layers - 1][j] * w[num_hidden_layers][j][0]

		#Final output Oy
		oy[num_hidden_layers][0]=self.activate(activation_function[num_hidden_layers],y[num_hidden_layers][0])
		return y,oy

	

	def Partial_Derivative(self,actual_output ,activation_functions , oy ,w):
		
		num_hidden_layers = self.num_hidden_layers
		num_input_features = self.num_input_features
		activation_function = self.activation_function


		partial_derivative_intermediate_outputs=np.zeros((num_hidden_layers + 1, num_input_features))
		# numpy array to store partial derivatives of loss wrt summed input to neurons (dE/dy)

		der_loss_wrt_output=oy[num_hidden_layers][0] - actual_output
		# derivative of loss (using 1/2(y^  -  y)**2 as cost) wrt calculated output 
		# final calculated output of the neural net is the output of the single neuron in the final layer ie oy[num_hidden_layer][0]
		
					
		partial_derivative_intermediate_outputs[num_hidden_layers][0]=der_loss_wrt_output * self.derivative_function(activation_functions[num_hidden_layers],oy[num_hidden_layers][0])
		
		for j in range(num_input_features):
			partial_derivative_intermediate_outputs[num_hidden_layers - 1][j]=partial_derivative_intermediate_outputs[num_hidden_layers][0] * w[num_hidden_layers-1][j][0]* self.derivative_function(activation_functions[num_hidden_layers-1],oy[num_hidden_layers - 1][j])    
	
		#Calculating subsequent derivatives using chain rule
		for i in range(num_hidden_layers-2,-1,-1):
			for j in range(num_input_features):
				for k in range(num_input_features):
					partial_derivative_intermediate_outputs[i][j]+= partial_derivative_intermediate_outputs[i+1][k] * w[i][j][k] * self.derivative_function(activation_function[i],oy[i][j])


		return partial_derivative_intermediate_outputs

	

	'''		
	back_prop is a function that takes in the weight and the bias matrices , actual output , activation_func , all intermediate outputs given by forward_prop,learning rate and the input row
	starts from the last layer and propogates backwards till the input layer (corresponding activation_function to be passed)
	'''
	def back_prop(self,w , b , actual_output , activation_functions ,oy , learning_rate , inp):

		num_hidden_layers = self.num_hidden_layers
		num_input_features = self.num_input_features
		activation_function = self.activation_function

		# updated weight and bias matrices
		nw=np.zeros((num_hidden_layers+1,num_input_features,num_input_features))
		nb=np.zeros((num_hidden_layers+1,num_input_features))
		
		partial_derivative_intermediate_outputs=self.Partial_Derivative(actual_output , activation_functions ,oy ,w)
		# function to calculated partial derivatives at the summer at each neuron

		# updating weights for output layer
		for j in range(num_input_features):
			for k in range(num_input_features):
				nw[0][j][k]=w[0][j][k]-learning_rate*partial_derivative_intermediate_outputs[0][k]*inp[j]
					
		# updating weights for hidden layers
		for i in range(1,num_hidden_layers):
			for j in range(num_input_features):
				for k in range(num_input_features):
					nw[i][j][k]=w[i][j][k]-learning_rate*partial_derivative_intermediate_outputs[i][k]*oy[i-1][j]

		for j in range(num_input_features):
			nw[num_hidden_layers][j][0]=w[num_hidden_layers][j][0]-learning_rate*partial_derivative_intermediate_outputs[num_hidden_layers][0]*oy[num_hidden_layers-1][j]
		
		# updating bias for output layer
		for i in range(num_hidden_layers):
			for j in range(num_input_features):
				nb[i][j]=b[i][j] - learning_rate * partial_derivative_intermediate_outputs[i][j]
		
		# updating bias for hidden layers
		for j in range(num_input_features):
			nb[num_hidden_layers][j]=b[num_hidden_layers][j] - learning_rate * partial_derivative_intermediate_outputs[num_hidden_layers][0]

		return nw,nb


	'''
	Function that trains the neural network by taking x_train and y_train samples as input
	#Training the nn involves forward propagation to estimate values
	#and backward propgation to tune the weights and biases
	'''

	def fit(self,X,Y):

		num_hidden_layers = self.num_hidden_layers
		num_input_features = self.num_input_features
		activation_function = self.activation_function


		def train(inp,weight,bias):
			
			actual_output=inp[-1];	inp=inp[:-1]
				
			y,oy=self.forward_prop(inp , activation_function , weight , bias )  	
			learning_rate=0.01
			weight,bias=self.back_prop(weight , bias , actual_output , activation_function, oy ,learning_rate , inp)   

			return weight,bias


		weight=np.zeros((num_hidden_layers+1,num_input_features,num_input_features))
		weight.fill(2/num_input_features) 
		# initialising the weights  with value 2/num_input_features to prevent saturation of activation function value
		
		bias=np.random.uniform(-0.15, 0.15, (num_hidden_layers+1,num_input_features))
		# intializing random bias
		 
		epochs=100
		
		tr=pd.concat([X, Y], axis=1)
		# concatenating X and Y so that shuffling of training data can be done before training

		train_data=tr.to_numpy()

		# feeding in training data
		for _ in range(epochs):
			for i in range(len(train_data)):
				weight,bias=train(train_data[i],weight,bias)


		# storing weights, bias and other variables for further use 
		self.w=weight
		self.b=bias
		self.activation_function=activation_function
		self.num_hidden_layers=num_hidden_layers
		self.num_input_features=num_input_features


	'''
	The predict function performs a simple feed forward of weights
	and outputs yhat values 
	yhat is a list of the predicted value for df'''		
	
	def predict(self,X):

		test_data=X.to_numpy()
		
		# feeding in the test data and getting the output
		yhat=[]
		for i in range(len(test_data)):
			y, oy = self.forward_prop(test_data[i],self.activation_function,self.w, self.b)
			yhat.append(oy[-1][0])

		return yhat

	def CM(self, y_test,y_test_obs):
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
		a = (tp+tn)/(tp+tn+fp+fn)
		
		print("Confusion Matrix : ")
		print(cm)
		print("\n")
		print(f"Precision : {p}")
		print(f"Recall : {r}")
		print(f"F1 SCORE : {f1}")
		print(f"Accuray : {a}")


	def main(self):
		# reading in the cleaned data
		data=pd.read_csv('Cleaned_LBW_Dataset.csv')

		# splitting the data into test and train sets
		x_train, x_test, y_train, y_test = train_test_split(data.iloc[:, :data.shape[1]-1], data.iloc[:,-1], test_size = 0.3, random_state = 44)

		# fitting the model to the data
		self.fit(x_train,y_train)

		# predicitig values using trained model
		yhat=self.predict(x_test)

		# getting the metrics
		yTest=y_test.tolist()
		self.CM(yTest, yhat)


# creating the neural network 
nn=NN()
nn.main()

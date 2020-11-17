#Implementation
The first step in predicting values for the dataset is to pass the neurons through a feedforward function.
Here, matrices 'y' computes the weighted inputs of each neuron added to bias. We perform matrix multiplication between 'input' and weight matrix 'w', after which we perform the addition of the bias matrix 'b' to this product to give us 'y'.
'oy' calculates the value of the neuron once the activation function is applied to each neuron in the layer. 
For each cell in the matrix 'y', we execute the corresponding activation function to give us the final matrix 'oy'.

Backpropagation is performed to help us tune the weights and bias to get a better fit.  
We have matrices 'nw' and 'nb' that get updated. We compute the partial derivatives of the outputs from each layer by using chain rule. We first calculate the output layer error and pass the result to the hidden layer before it. 
After calculating the hidden layer error, we pass its error value back to the previously hidden layer before it.
This means that we keep applying the derivative of activation function on each intermediate output, and multiply with the corresponding weight of the layer, and the prior calculated partial derivative.
Training of the neural network takes place by applying a feedforward on the input neurons and improving the model with backpropagation by tuning the weights.

#hyperparameters
We initialize the weight matrix as a three-dimensional matrix, of zeroes whose dimensions are 
(number of hidden layers + 1) * (number of inputs * number of inputs). 
The bias matrix is initialized with ones. Its dimensions are (number of hidden layers + 1) * (number of inputs).
The number of hidden layers is 4.
We chose a learning rate of 0.01.
We compute over 100 epochs. Relu activation function is applied to all the hidden layers, and sigmoid function is applied to the final layer for classification.

#What is the key feature of your design that makes it stand out
Our design of the neural network is done with minimal dependance on libraries like numpy. For instance, we have worked on a step by step implementation of the matrix multiplication, during forward propagation.
We initialized a three-dimensional weight matrix according to which one dimension determines the layer number, to allow for multiplication of weights with the intermediate outputs.
Similarly in backward propagation, we have implemented the chain rule of derivation in this manner. We iterate over all layers and pass computed derivatives to the next layer.

#Detailed steps to run your files 
1] python data_preprocessing.py
2] python Neural_Net.py


data_preprocessing.py performs data cleaning on the original dataset.
The model Neural_Net.py works with the updated csv file.
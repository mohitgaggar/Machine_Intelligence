#Implementation
The first step in predicting values for the dataset is to pass the neurons through a feedforward function.
Here, matrices 'y' computes the weighted inputs of each neuron added to bias. We replicate matrix multiplication between 'input' and weight matrix 'w', after which we perform the addition of the bias matrix 'b' to this product to give us 'y'.
'oy' is the value of the summed output passed through the activation function.
For each cell in the matrix 'y', we execute the corresponding activation function to give us the final matrix 'oy'.

To tune weights and biases to get a overall fit of data in out neural network we use gradient descent by descending along the gradient of the error function to reach a minima.
To calculate the gradient by use backpropagation and performing chain rule repeatedly.
For out model we have a function that calculates the partial derivative of the loss function wrt the 'oy' (the outputs of the neurons when summed and passed through the activation at each stage)
This intermediate derivative is stored so when we can calculate the partial derivative of the loss function wrt to the weights and the biases.
Finally the back propagation function has 2 matrices 'nw' and 'nb' which stores the new weights and biases respectively using the formuala (nw = w - learning_rate * dw (just a representation))
these new weights and biases replace our original ones in the model.
This means that we keep applying the derivative of activation function on each intermediate output, and multiply with the corresponding weight of the layer, and the prior calculated partial derivative.
Training of the neural network takes place by applying a feedforward on the input neurons and improving the model with backpropagation by tuning the weights.


#Hyperparameters
We initialize the weight matrix as a three-dimensional matrix, of value 2/number_of_input_featiures to prevent saturation of the activation function whose dimensions are 
(number of hidden layers + 1) * (number of inputs) *() number of inputs).  
number oh hidden layer + 1 (+1 for output)
which is (5 X 9 X 9)


The bias matrix is initialized with ones. Its dimensions are (number of hidden layers + 1) * (number of inputs). which is (5 X 9)

The number of hidden layers is 4.

We chose a learning rate of 0.01.

We compute over 100 epochs. 
Relu activation function is applied to all the hidden layers, and sigmoid function is applied to the final layer for classification.

#What is the key feature of your design that makes it stand out
Our design of the neural network is done with minimal dependance on libraries like numpy. For instance, we have worked on a step by step implementation of the matrix multiplication, during forward propagation.
We initialized a three-dimensional weight matrix according to which one dimension determines the layer number, to allow for multiplication of weights with the intermediate outputs.
Similarly in backward propagation, we have implemented the chain rule of derivation in this manner. We iterate over all layers and pass computed derivatives to the next layer.

#Detailed steps to run your files 
1] python data_preprocessing.py
2] python Neural_Net.py


data_preprocessing.py performs data cleaning on the original dataset.
The model Neural_Net.py works with the updated csv file.
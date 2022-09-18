import numpy as np
import random
import pickle as cPickle
import gzip

## Learning rate of neural network
LEARNING_RATE = 0.5


class NeuralNet:

    def __init__(self, layers):
               
        '''
        initalize the weights and baises to random values from 0 to 1 on a normal distribution

        the biases are only added for all layers except the first input layer

        the weights connecting two layers are added as a 2-d array such that weight matrix is 
        layers[n+1] by layers[n] where n is current layer 
        '''

        self.layers = layers
 
        self.biases = [np.random.randn(layers[n+1],1) for n in range(len(layers)-1)]
        self.weights = [np.random.randn(layers[n+1], layers[n]) for n in range(len(layers)-1)]
        
    def feedForward(self, trainingData, batchSize, epoch):

        '''
        feed all the training data through the neural network and adjust the weights
        and biases according to the gradient descent
        '''

        # compute gradient descent for the number of epochs
        for e in range(epoch):
            
            print("starting epoch:%i" % (e+1))

            # shuffle the data set
            random.shuffle(trainingData)

            # creates batches from the trainingData
            batches = [trainingData[i:i+batchSize] for i in range(0, len(trainingData), batchSize)]
           
            for batch in batches:
                
                # initalize empty bias and weight adjustment matrices
                biasAdj = [np.zeros(i.shape) for i in self.biases]
                weightAdj = [np.zeros(i.shape) for i in self.weights]

                # find the weight and bias adjustments for each input in the batch
                for input, output in batch:

                    # list of weighted sums (not including input and output layers)
                    zs = []

                    # list of neuron activations at each layer inluding input and output
                    activation = input
                    activations = [input]
                    
                    # feed forward through the network
                    for weight, bias in zip(self.weights, self.biases):
                        
                        # compute weighted sums
                        z = np.dot(weight, activation) + bias
                        
                        zs.append(z)
                        
                        # compute activations
                        activation = sigmoid(z)
                        
                        activations.append(activation)
                 
                    # retrieve weight and bias adjustment from the given input
                    biasAdj, weightAdj = self.backProp(output, activations, biasAdj, weightAdj, zs)
                
                # adjust weights and biases with the average gradient descent from batch
                # bias = old_bias - (η/m) ∑∂C/∂b
                # weight = old_weight - (η/m) ∑∂C/∂w
                for i in range(len(biasAdj)):
                    biasAdj[i] *= LEARNING_RATE / len(batch)
                
                for i in range(len(weightAdj)):
                    weightAdj[i] *= LEARNING_RATE / len(batch)
                
                self.biases = [bias - bAdj for bias,bAdj in zip(self.biases, biasAdj)]
                self.weights = [weight - wAdj for weight,wAdj in zip(self.weights, weightAdj)]

    def backProp(self, expectedOut, activations, biasAdj, weightAdj, zs):
        
        # compute derivative of the squared residuals of cost function
        # ∂C/∂b = ∑-2(y-predicted)
        # bias adjustment of output layer is ∂C/∂b
        '''
        error = activations[-1] - expectedOut  
     
        delta =  np.array(error) * sigmoidDeriv(sigmoid(zs[-1]))
        '''
        delta = CostFunctions.crossEntropyDeriv(zs[-1], expectedOut)

     

        biasAdj[-1] += delta

        # compute weight adjustment of weights going to ouptut layer
        # ∂C/∂b = ∑-2(y-predicted)*prev_layer_activations
        weightAdj[-1] += np.dot(delta , activations[-2].T)


        for i in range(2, len(activations)):
            
            sigmoidDerivative = sigmoidDeriv(sigmoid(zs[-i]))
            
            delta = np.dot(self.weights[-i+1].transpose(), delta) * sigmoidDerivative
         
            biasAdj[-i] += delta
            
            weightAdj[-i] += np.dot(delta , activations[-i-1].T)

        return biasAdj, weightAdj

    def compute(self, input):

        activation = input
        
        for weight, bias in zip(self.weights, self.biases):

            activation = sigmoid(np.dot(weight, activation) + bias)

        return activation

    def saveWandB(self, fileName):
        
        np.savez(fileName, biases=self.biases, weights=self.weights)

    def loadWandB(self, fileName):

        data = np.load(fileName, allow_pickle=True)
        
        self.biases = data["biases"]
        self.weights = data["weights"]

    def train(self, fileName):
        self.feedForward(trainingData, 10, 30)

        self.saveWandB(fileName)

    def evaluateSingleData(self, fileName, trainingData):
        self.loadWandB(fileName)

        input = trainingData[0]
        output = trainingData[1]

        netOutput = self.compute(input)

        maxIndex = np.argmax(netOutput)

        output = np.zeros((len(netOutput),1))
        output[maxIndex] = 1

        print("NN output:")
        print("\t" + str(netOutput.T))
        print("\t" + str(output.T))
        print("\tdigit: " + str(maxIndex) + "\n")

        print("Actual Output:")
        print("\t" + str(output.T))
        print("\tdigit: " + str(np.argmax(output)))

    def testData(self, fileName, testData):
        
        self.loadWandB(fileName)

        totalNumData = len(testData)
        correct = 0

        for (x, y) in testData:
            
            netOutput = np.argmax(self.compute(x))
           
            if (netOutput == y):
                correct += 1
            
        print("Correctly Identified: [%i/%i]" % (correct, totalNumData))
        print("Accuracy: %.3f%%" % (correct/totalNumData*100.0))

  


class CostFunctions:

    def crossEntropyDeriv(self, a, y):
        return sigmoid(a) - y
    
    def quadraticCostDeriv(self, a, y):
        return sigmoidDeriv(sigmoid(a)) - y








def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoidDeriv(c):
    return c * (1 - c)

def loadMNISTData(file):
    '''
    load the trainging, validation, and testing data from MNIST data set

    trainingData contains 60,000 inputs and outputs

    validationData & testData contains 10,000 inputs
    '''
    f = gzip.open(file, 'rb')
    trainingData, validationData, testData = cPickle.load(f, encoding="latin1")
    f.close()

    '''
    convert the 28x28 input matix into a vector

    convert the output from 0-9 to a zero vector, where the index of 1 is the output number
    '''
    trainingInputs = [np.reshape(x, (784, 1)) for x in trainingData[0]]
    trainingResults = [vectorizedResult(y) for y in trainingData[1]]
    trainingData = zip(trainingInputs, trainingResults)
    
    validationInputs = [np.reshape(x, (784, 1)) for x in validationData[0]]
    validationData = zip(validationInputs, validationData[1])

    testInputs = [np.reshape(x, (784, 1)) for x in testData[0]]
    testData = zip(testInputs, testData[1])

    return trainingData, validationData, testData

def vectorizedResult(output):
    '''
    convert digit into a zero vector, where the index of 1 is the output number
    '''
    e = np.zeros((10, 1))
    e[output] = 1.0
    return e



if __name__ == "__main__":

    datas = loadMNISTData("mnist.pkl.gz")
    trainingData = list(datas[0])
    validationData = list(datas[1])
    testData = list(datas[2])

    net = NeuralNet([784,30,10])

    net.train("Trained_Data/784_30'_10.npz")
    
        
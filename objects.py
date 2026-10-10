import numpy as np
import math
from dataset import *

# np.random.seed(1)

X = [[0.2, 0.3, 0.4, 0.5],
     [0.6, 0.7, 0.8, 0.9],
     [0.2, 0.35, 0.17, 0.89],
     [0.01, 0.2, 0.44, 0.67]]



X, y = createData(100, 3)

class layerDense:
    def __init__(self, n_inputs, n_neurons):
        # shape declares inputs as columns so transpositions arent needed
        self.weights = 0.15 * np.random.randn(n_inputs, n_neurons)

        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class activationReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


# softmax normalises all outputs whilst skewing negative numbers to minimal positive numbers
# raises output to the e before dividing by sum of all exponentiated values
# -> subtracts the largest input from all to avoid raising e to the power of really large number
# --> thus inputs in the range -infinity to 0, and exponetiated range between 0 and 1

# basically creates a probability distribution of inputs
class activationSoftMax:
    def forward(self, inputs): 
        expOutput = np.exp(inputs - np.max(inputs, axis = 1, keepdims = True))
        probabilitites = expOutput / np.sum(expOutput, axis = 1, keepdims = True)

        self.output = probabilitites

class Loss:
    def calculate(self, output, y):
        sampleLosses = self.forward(output, y)
        dataLoss = np.mean(sampleLosses)

        return dataLoss
    
class LossCategoricalCrossEntropy(Loss):
    def forward(self, predictedY, actualY):
        samples = len(predictedY)

        # clips predicted Y to make sure it is not equal to 0
        # this is because the loss function uses ln(predictedY)
        # and means all the values in the batch, mean including infinity is infinity
        clippedPredictedY = np.clip(predictedY, 1e-7, 1-1e-7)

        # handling whether actualY is scalar values or one hot encoded vectors
        if len(actualY.shape) == 1:
            # 2d index where for each sample, the actualY is the actual index
            # correctConfidences is a list of the networks confidence in the right answer
            correctConfidences = clippedPredictedY[range(samples), actualY]

        elif len(actualY.shape) == 2:
            # multiplies each sample output by the one hot encoded actual y
            correctConfidences = np.sum(clippedPredictedY*actualY, axis = 1)

        loss = -np.log(correctConfidences)
        return loss


layer1 = layerDense(2, 5)
layer2 = layerDense(5, 5)

firstActivation = activationReLU()
lastActivation = activationSoftMax()

layer1.forward(X)
firstActivation.forward(layer1.output)

layer2.forward(firstActivation.output)
lastActivation.forward(layer2.output)

print(lastActivation.output)

lossFunction = LossCategoricalCrossEntropy()
loss = lossFunction.calculate(lastActivation.output, y)

print(loss)

#layer2.forward(layer1.output)

#print(layer2.output)

# exponentiation for softmax


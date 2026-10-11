from keras.api.datasets import mnist
import numpy as np

# gets training and test data from mnist
# -> the test images are kept separate, to measure how well the network does on images it hasn't trained on
data = mnist.load_data()

(train_X, train_y), (test_X, test_y) = data
train_X = train_X.reshape(train_X.shape[0], 784).T / 255
test_X = test_X.reshape(test_X.shape[0], 784).T / 255

train_y = train_y.T
test_y = test_y.T


# random starting weights and biases between -0.5 and 0.5
def initParams():
    # 10 nodes in layer1, each with 784 weights from previous layer
    weights1 = np.random.rand(10, 784) - 0.5

    # biases for layer 1 neurons
    biases1 = np.random.rand(10, 1) - 0.5

    # 10 nodes in layer2, each with 10 weights from previous layer
    weights2 = np.random.rand(10, 10) - 0.5

    # biases for layer 2 neurons
    biases2 = np.random.rand(10, 1) - 0.5

    return weights1, biases1, weights2, biases2


# sets every negative value to 0
def ReLU(Z):
    return np.maximum(Z, 0)


# raises e to the power of each output, then divides by each column's total
# -> each column (one image) becomes a probability distribution over the 10 classes
# -> subtracting each column's largest value first stops np.exp overflowing, and doesn't change the result
def softMax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    A =  expZ / expZ.sum(axis = 0)
    return A


# Z stands for a layer of neurons and A stands for the activated layer of these neurons
def forwardPropagation(weights1, biases1, weights2, biases2, X):
    Z1 = np.dot(weights1, X) + biases1
    A1 = ReLU(Z1)
    Z2 = np.dot(weights2, A1) + biases2
    A2 = softMax(Z2)

    return Z1, A1, Z2, A2


# creates a matrix of the actual expected values that should come from the neural network
def oneHotEncoding(Y):
    oneHotEncodedY = np.zeros((Y.max() + 1, Y.size))
    oneHotEncodedY[Y, np.arange(Y.size)] = 1

    return oneHotEncodedY


# gets the derivative of the ReLU function which is 1 when greater than 0
# boolean gets converted into 1 or 0 when going into calculations
def ReLUDerivative(Z):
    return Z > 0


# finds the gradient of all weights, biases, and layers
def backwardsPropogation(Z1, A1, Z2, A2, W2, X, Y):
    m = Y.size
    oneHotY = oneHotEncoding(Y)

    # gradient of the cross-entropy loss with respect to Z2
    # -> for softmax followed by cross-entropy this simplifies to the predicted probabilities minus the one-hot labels
    dZ2 = A2 - oneHotY

    # gradients of layer 2's weights and biases, averaged over the batch
    # -> each bias sums its own row over the images (axis=1), so every bias gets its own gradient
    dW2 = (1 / m) * dZ2.dot(A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    # can only apply a change if neuron was activated (relu derivative)
    dZ1 = W2.T.dot(dZ2) * ReLUDerivative(Z1)

    dW1 = (1 / m) * dZ1.dot(X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2


# moves each variable a small step against its derivative, towards a minimum of the loss
def updateParams(W1, b1, W2, b2, dW1, db1, dW2, db2, learningRate):
    W1 = W1 - learningRate * dW1
    W2 = W2 - learningRate * dW2
    b1 = b1 - learningRate * db1
    b2 = b2 - learningRate * db2

    return W1, b1, W2, b2


# returns the index of the value with biggest probability
def getPredictions(A2):
    return np.argmax(A2, 0)


# fraction of predictions that match the labels
def getAccuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size


# full-batch gradient descent: every iteration uses all of the training images at once
def gradientDescent(X, Y, iterations, learningRate):
    weights1, biases1, weights2, biases2 = initParams()

    for i in range(iterations):
        Z1, A1, Z2, A2 = forwardPropagation(weights1, biases1, weights2, biases2, X)
        dW1, db1, dW2, db2 = backwardsPropogation(Z1, A1, Z2, A2, weights2, X, Y)
        weights1, biases1, weights2, biases2 = updateParams(weights1, biases1, weights2, biases2, dW1, db1, dW2, db2, learningRate)

        if i % 10 == 0:
            print(f"Iteration: {i}")
            print(f"Accuracy: {getAccuracy(getPredictions(A2), Y)}")
            print()
    
    return weights1, biases1, weights2, biases2


if __name__ == "__main__":
    np.random.seed(0)
    weights1, biases1, weights2, biases2 = gradientDescent(train_X, train_y, 500, 0.1)

    _, _, _, A2 = forwardPropagation(weights1, biases1, weights2, biases2, test_X)
    print(f"Test accuracy: {getAccuracy(getPredictions(A2), test_y):.2%}")

from keras.api.datasets import fashion_mnist
from matplotlib import pyplot
import numpy as np
import json

# gets training and test data from fashion-mnist
# -> the test images are kept separate, to measure how well the network does on images it hasn't trained on
data = fashion_mnist.load_data()

(train_X, train_y), (test_X, test_y) = data
train_X = train_X.reshape(train_X.shape[0], 784).T / 255
test_X = test_X.reshape(test_X.shape[0], 784).T / 255

train_y = train_y.T
test_y = test_y.T


# random starting weights and biases between -0.5 and 0.5
def initParams():
    # 20 nodes in layer1, each with 784 weights from previous layer
    weights1 = np.random.rand(20, 784) - 0.5

    # biases for layer 1 neurons
    biases1 = np.random.rand(20, 1) - 0.5

    # 20 nodes in layer2, each with 20 weights from previous layer
    weights2 = np.random.rand(20, 20) - 0.5

    # biases for layer 2 neurons
    biases2 = np.random.rand(20, 1) - 0.5

    # 10 nodes in layer3 (one per class), each with 20 weights from previous layer
    weights3 = np.random.rand(10, 20) - 0.5

    # biases for layer 3 neurons
    biases3 = np.random.rand(10, 1) - 0.5

    return weights1, biases1, weights2, biases2, weights3, biases3


# sets every negative value to 0
def ReLU(Z):
    return np.maximum(Z, 0)


# raises e to the power of each output, then divides by each column's total
# -> each column (one image) becomes a probability distribution over the 10 classes
# -> subtracting each column's largest value first stops np.exp overflowing, and doesn't change the result
def softMax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    A = expZ / expZ.sum(axis = 0)
    return A


# Z stands for a layer of neurons and A stands for the activated layer of these neurons
def forwardPropagation(weights1, biases1, weights2, biases2, weights3, biases3, X):
    Z1 = np.dot(weights1, X) + biases1
    A1 = ReLU(Z1)
    Z2 = np.dot(weights2, A1) + biases2
    A2 = ReLU(Z2)
    Z3 = np.dot(weights3, A2) + biases3
    A3 = softMax(Z3)

    return Z1, A1, Z2, A2, Z3, A3


# creates a matrix of the actual expected values that should come from the neural network
def oneHotEncoding(Y):
    oneHotEncodedY = np.zeros((10, Y.size))
    oneHotEncodedY[Y, np.arange(Y.size)] = 1

    return oneHotEncodedY


# gets the derivative of the ReLU function which is 1 when greater than 0
# boolean gets converted into 1 or 0 when going into calculations
def ReLUDerivative(Z):
    return Z > 0


# finds the gradient of all weights, biases, and layers
def backwardsPropogation(Z1, A1, Z2, A2, W2, Z3, A3, W3, X, Y):
    m = Y.size
    oneHotY = oneHotEncoding(Y)

    # gradient of the cross-entropy loss with respect to Z3
    # -> for softmax followed by cross-entropy this simplifies to the predicted probabilities minus the one-hot labels
    # -> each bias sums its own row over the images (axis=1), so every bias gets its own gradient
    dZ3 = A3 - oneHotY
    dW3 = (1 / m) * dZ3.dot(A2.T)
    db3 = (1 / m) * np.sum(dZ3, axis=1, keepdims=True)

    # send the gradient back through W3; it only reaches neurons that were active (relu derivative)
    dZ2 = W3.T.dot(dZ3) * ReLUDerivative(Z2)
    dW2 = (1 / m) * dZ2.dot(A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    # the same again, back through W2
    dZ1 = W2.T.dot(dZ2) * ReLUDerivative(Z1)
    dW1 = (1 / m) * dZ1.dot(X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2, dW3, db3


# moves each variable a small step against its derivative, towards a minimum of the loss
def updateParams(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, learningRate):
    W1 = W1 - learningRate * dW1
    W2 = W2 - learningRate * dW2
    W3 = W3 - learningRate * dW3
    b1 = b1 - learningRate * db1
    b2 = b2 - learningRate * db2
    b3 = b3 - learningRate * db3

    return W1, b1, W2, b2, W3, b3


# returns the index of the value with biggest probability
def getPredictions(A3):
    return np.argmax(A3, 0)


# fraction of predictions that match the labels
def getAccuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

# average probability the network gave to the class it predicted
def getCertainty(predictions, A3):

    return np.mean(A3[predictions, np.arange(len(predictions))])


def gradientDescent(X, Y, epochs, learningRate, batchSize, updateFrequency, seed=0):
    np.random.seed(seed)

    weights1, biases1, weights2, biases2, weights3, biases3 = initParams()
    m = X.shape[1]

    for i in range(epochs):
        # shuffle data
        indices = np.arange(m)
        np.random.shuffle(indices)
        X_shuffled = X[:, indices]
        Y_shuffled = Y[indices]

        correct = 0

        # process a mini-batch
        for startIndex in range(0, m, batchSize):
            # the last batch can be smaller than batchSize
            endIndex = min(startIndex + batchSize, m)
            X_batch = X_shuffled[:, startIndex:endIndex]
            Y_batch = Y_shuffled[startIndex:endIndex]

            Z1, A1, Z2, A2, Z3, A3 = forwardPropagation(weights1, biases1, weights2, biases2, weights3, biases3, X_batch)
            correct += np.sum(getPredictions(A3) == Y_batch)
            dW1, db1, dW2, db2, dW3, db3 = backwardsPropogation(Z1, A1, Z2, A2, weights2, Z3, A3, weights3, X_batch, Y_batch)
            weights1, biases1, weights2, biases2, weights3, biases3 = updateParams(weights1, biases1, weights2, biases2, weights3, biases3, dW1, db1, dW2, db2, dW3, db3, learningRate)

        if i % updateFrequency == 0 or i == epochs - 1:
            print(f"Epoch: {i}")
            print(f"Accuracy: {correct / m:.2%}")
            print()
    
    return weights1, biases1, weights2, biases2, weights3, biases3


def saveVariables(W1, W2, W3, b1, b2, b3, filename):
    data = {
        "W1": W1.tolist(),
        "W2": W2.tolist(),
        "W3": W3.tolist(),
        "b1": b1.tolist(),
        "b2": b2.tolist(),
        "b3": b3.tolist()
    }

    with open(filename, "w") as f:
        json.dump(data, f, indent = 4)


def predict(X, W1, W2, W3, b1, b2, b3):
    _, _, _, _, _, A3 = forwardPropagation(W1, b1, W2, b2, W3, b3, X)
    predictions = getPredictions(A3)

    certainty = A3[predictions]
    return predictions, certainty


def testPredictionSingle(index, W1, W2, W3, b1, b2, b3):

    currentImage = test_X[:, index, None]
    prediction, certainty = predict(currentImage, W1, W2, W3, b1, b2, b3)

    label = test_y[index]
    label = convertToTextLabel(label)

    print(f"Prediction: {prediction}")
    print(f"Certainty: {certainty}")
    print(f"Label: {label}")

    currentImage = currentImage.reshape((28, 28)) * 255
    pyplot.gray()
    pyplot.imshow(currentImage, interpolation='nearest')
    pyplot.show()


def loadVariables(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    
    # Convert lists back to NumPy arrays
    weights1 = np.array(data["W1"])
    biases1 = np.array(data["b1"])
    weights2 = np.array(data["W2"])
    biases2 = np.array(data["b2"])
    weights3 = np.array(data["W3"])
    biases3 = np.array(data["b3"])
    
    return weights1, biases1, weights2, biases2, weights3, biases3


def fullTest(W1, W2, W3, b1, b2, b3):
    _, _, _, _, _, A3 = forwardPropagation(W1, b1, W2, b2, W3, b3, test_X)
    accuracy = getAccuracy(getPredictions(A3), test_y)
    certainty = getCertainty(getPredictions(A3), A3)

    return accuracy, certainty


def convertToTextLabel(label):
    type = ["T-Shirt", "Trousers", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]

    return type[int(label)]


# <------ MAIN -------->
if __name__ == "__main__":
    active = True

    while active:
        print("Options:\n1. Retrain neural network \n2. Single Test \n3. Full Test\n4. Exit \n")
        option = int(input())
        print()

        if option == 1:
            epochs = int(input("How many epochs would you like?\n\n"))
            print()

            learningRate = float(input("What would you like the learning rate to be (recommended between 0-0.2)?\n\n"))
            print()

            batchSize = int(input("What would you like the batch size to be (recommended between 64-128)?\n\n"))
            print()

            updates = int(input("How often would you like updates (in epochs)?\n\n"))
            print()

            seed = int(input("Enter an integer seed for rng:\n\n"))
            print()

            weights1, biases1, weights2, biases2, weights3, biases3 = gradientDescent(train_X, train_y, epochs, learningRate, batchSize, updates, seed)
            saveVariables(weights1, weights2, weights3, biases1, biases2, biases3, "threeVariableRetrained.json")
            print()

        elif option == 4:
            active = False

        else:
            filename = input("What file's weights would you like to load? (threeVariable.json is from the original training)\n\n")

            W1, b1, W2, b2, W3, b3 = loadVariables(filename)

            if option == 2:
                index = int(input("What index would you like to test?\n\n"))

                testPredictionSingle(index, W1, W2, W3, b1, b2, b3)
                print()

            else:
                accuracy, certainty = fullTest(W1, W2, W3, b1, b2, b3)
                print(f"Neural network has an accuracy of {accuracy * 100}%")
                print(f"Neural network has an average certainty of {certainty * 100}%")
                print()

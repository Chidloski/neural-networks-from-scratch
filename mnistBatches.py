from keras.api.datasets import fashion_mnist
from matplotlib import pyplot
import numpy as np
import json

# gets training and test data from mnist
# -> test data is needed to ensure model isn't overfitted to dataset
data = fashion_mnist.load_data()

(train_X, train_y), (test_X, test_y) = data
train_X = train_X.reshape(train_X.shape[0], 784).T / 255
test_X = test_X.reshape(test_X.shape[0], 784).T / 255

train_y = train_y.T
test_y = test_y.T


m, n = train_X.shape


'''print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))

for i in range(9):  
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(train_X[i], cmap=pyplot.get_cmap('gray'))


pyplot.show()'''


# assigns a value between -0.5 and 0.5 for the shape
def initParams():
    # 10 nodes in layer1, each with 784 weights from previous layer
    weights1 = np.random.rand(10, 784) - 0.5

    # biases for lyaer 1 neurons
    biases1 = np.random.rand(10, 1) - 0.5

    # 10 nodes in layer2, each with 10 weights from previous layer
    weights2 = np.random.rand(10, 10) - 0.5

    # biases for lyaer 2 neurons
    biases2 = np.random.rand(10, 1) - 0.5

    return weights1, biases1, weights2, biases2


# clips all neuron outputs within the layer below 0
def ReLU(Z):
    return np.maximum(Z, 0)


# raises all neuron outputs to the power of e before taking a mean of this
# -> creates a probability distribution
def softMax(Z):
    expZ = np.exp(Z - np.max(Z))
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
    oneHotEncodedY = np.zeros((10, Y.size))
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

    # think this calculates the loss as the difference between layer and actual
    dZ2 = A2 - oneHotY

    # the derivative for every variable for every neuron
    dW2 = (1 / m) * dZ2.dot(A1.T)
    db2 = (1 / m) * np.sum(dZ2)

    # can only apply a change if neuron was activated (relu derivative)
    dZ1 = W2.T.dot(dZ2) * ReLUDerivative(Z1)

    dW1 = (1 / m) * dZ1.dot(X.T)
    db1 = (1 / m) * np.sum(dZ1)

    return dW1, db1, dW2, db2


# alters the variables with their deriviatives to get them closer to minimum
def updateParams(W1, b1, W2, b2, dW1, db1, dW2, db2, learningRate):
    W1 = W1 - learningRate * dW1
    W2 = W2 - learningRate * dW2
    b1 = b1 - learningRate * db1
    b2 = b2 - learningRate * db2

    return W1, b1, W2, b2


# returns the index of the value with biggest probability
def getPredictions(A2):
    return np.argmax(A2, 0)


# percentage accuracy of network
def getAccuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def getCertainty(predictions, A2):

    return np.mean(A2[predictions, np.arange(len(predictions))])


def gradientDescent(X, Y, epochs, learningRate, batchSize, updateFrequency):
    weights1, biases1, weights2, biases2 = initParams()
    m = X.shape[1]

    for i in range(epochs):
        # shuffle data
        indices = np.arange(m)
        np.random.shuffle(indices)
        X_shuffled = X[:, indices]
        Y_shuffled = Y[indices]

        # process a mini-batch
        for startIndex in range(0, m, batchSize):
            # one index before the next loop
            endIndex = min(startIndex + batchSize, m)
            X_batch = X_shuffled[:, startIndex:endIndex]
            Y_batch = Y_shuffled[startIndex:endIndex]

            Z1, A1, Z2, A2 = forwardPropagation(weights1, biases1, weights2, biases2, X_batch)
            dW1, db1, dW2, db2 = backwardsPropogation(Z1, A1, Z2, A2, weights2, X_batch, Y_batch)
            weights1, biases1, weights2, biases2 = updateParams(weights1, biases1, weights2, biases2, dW1, db1, dW2, db2, learningRate)

        if i % updateFrequency == 0:
            print(f"Iteration: {i}")
            print(f"Accuracy: {getAccuracy(getPredictions(A2), Y_batch)}")
            print()
    
    return weights1, biases1, weights2, biases2


def saveVariables(W1, W2, b1, b2):
    data = {
        "W1": W1.tolist(),
        "W2": W2.tolist(),
        "b1": b1.tolist(),
        "b2": b2.tolist()
    }

    with open("variables.json", "w") as f:
        json.dump(data, f, indent = 4)


'''weights1, biases1, weights2, biases2 = gradientDescent(train_X, train_y, 100, 0.1, 64)
saveVariables(weights1, weights2, biases1, biases2)'''


def predict(X, W1, W2, b1, b2):
    _, _, _, A2 = forwardPropagation(W1, b1, W2, b2, X)
    predictions = getPredictions(A2)

    certainty = A2[predictions]
    return predictions, certainty


def testPredictionSingle(index, W1, W2, b1, b2):

    currentImage = test_X[:, index, None]
    prediction, certainty = predict(currentImage, W1, W2, b1, b2)

    label = test_y[index]
    label = convertToTextLabel(label)

    print(f"Prediction: {prediction}")
    print(f"Certainty: {certainty}")
    print(f"Label: {label}")

    currentImage = currentImage.reshape((28, 28)) * 255
    pyplot.gray()
    pyplot.imshow(currentImage, interpolation='nearest')
    pyplot.show()


def loadVariables():
    with open("variables.json", "r") as f:
        data = json.load(f)
    
    # Convert lists back to NumPy arrays
    weights1 = np.array(data["W1"])
    biases1 = np.array(data["b1"])
    weights2 = np.array(data["W2"])
    biases2 = np.array(data["b2"])
    
    return weights1, biases1, weights2, biases2

'''W1, b1, W2, b2 = loadVariables()

testPredictionSingle(190, W1, W2, b1, b2)'''


def fullTest(W1, W2, b1, b2):
    _, _, _, A2 = forwardPropagation(W1, b1, W2, b2, test_X)
    accuracy = getAccuracy(getPredictions(A2), test_y)
    certainty = getCertainty(getPredictions(A2), A2)

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

            weights1, biases1, weights2, biases2 = gradientDescent(train_X, train_y, epochs, learningRate, batchSize, updates)
            saveVariables(weights1, weights2, biases1, biases2)
            print()

        elif option == 4:
            active = False

        else:
            W1, b1, W2, b2 = loadVariables()

            if option == 2:
                index = int(input("What index would you like to test?\n\n"))

                testPredictionSingle(index, W1, W2, b1, b2)
                print()

            else:
                accuracy, certainty = fullTest(W1, W2, b1, b2)
                print(f"Neural network has an accuracy of {accuracy * 100}%")
                print(f"Neural network has an average certainty of {certainty * 100}%")
                print()

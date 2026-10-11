from keras.api.datasets import fashion_mnist
from matplotlib import pyplot
import numpy as np
import json
from scipy.signal import correlate, correlate2d

# gets training and test data from fashion-mnist
# -> the test images are kept separate, to measure how well the network does on images it hasn't trained on
# -> the images stay float64: as float32, scipy's fft convolution would run in single precision
data = fashion_mnist.load_data()

(train_X, train_y), (test_X, test_y) = data
train_X = train_X / 255

test_X = test_X / 255

train_X = np.expand_dims(train_X, axis=-1)  # Shape becomes (num_images, 28, 28, 1)
test_X = np.expand_dims(test_X, axis=-1)


# random starting weights, scaled by each layer's number of inputs and outputs (glorot initialisation)
# -> biases start at 0
def initParams():
    # kWeights has the weights for each position of the first layer's kernels
    # the dimension 9x9x1x2 represents a 9x9 kernel on an input of depth 1, with two separate kernels
    kWeights1 = np.random.randn(9, 9, 1, 2) * np.sqrt(2 / (9 * 9 * 1 + 2))
    # a single bias is needed for each kernel
    kBias1 = np.zeros(2)

    # this follows the same language as the first layer of kernels although each of the input has depth two
    # this comes from each filter in the first layer
    kWeights2 = np.random.randn(3, 3, 2, 4) * np.sqrt(2 / (3 * 3 * 2 + 4))
    kBias2 = np.zeros(4)

    # for the fully connected layer 64 weights are needed for each node
    # this is because the four 4x4 matrices made by the kernels are flattened to a 1d vector
    fcWeights = np.random.randn(10, 64) * np.sqrt(2 / (64 + 10))
    fcBias = np.zeros(10)

    return kWeights1, kBias1, kWeights2, kBias2, fcWeights, fcBias


# sets every negative value to 0
def ReLU(Z):
    return np.maximum(Z, 0)

# passes values into a sigmoid function with range 0-1
def sigmoid(Z):
    return 1 / (1 + np.exp(-np.clip(Z, -500, 500)))

# keeps the largest value in each 2x2 block, halving the height and width
# -> also returns a mask with a 1 where each block's largest value was, for the backward pass
def max_pooling(feature_maps, pool_size=2, stride=2):
    # Get the dimensions of the input feature maps
    height, width, num_channels = feature_maps.shape
    
    # Calculate the dimensions of the output feature maps
    out_height = (height - pool_size) // stride + 1
    out_width = (width - pool_size) // stride + 1
    
    # Initialize the output feature maps
    pooled_feature_maps = np.zeros((out_height, out_width, num_channels))
    mask = np.zeros_like(feature_maps)
    
    # Perform max pooling for each channel
    for c in range(num_channels):
        for i in range(0, height - pool_size + 1, stride):
            for j in range(0, width - pool_size + 1, stride):
                # Extract the pooling window
                pooling_window = feature_maps[i:i+pool_size, j:j+pool_size, c]
                
                # Take the maximum value in the pooling window
                pooled_feature_maps[i // stride, j // stride, c] = np.max(pooling_window)
                # record where the max was, so the backward pass can send the gradient only there
                # -> argmax picks exactly one cell even when values tie
                row, col = np.unravel_index(np.argmax(pooling_window), pooling_window.shape)
                mask[i + row, j + col, c] = 1
    
    return pooled_feature_maps, mask

# raises e to the power of each output, then divides by the total
# -> gives a probability for each of the 10 classes
# -> subtracting the largest output first stops np.exp overflowing, and doesn't change the result
def softMax(Z):
    expZ = np.exp(Z - np.max(Z))
    A =  expZ / expZ.sum(axis = 0)
    return A


# runs one image through both convolution layers and the fully connected layer
# -> 'Maps' are values before the activation function and 'activated' ones are after it
# -> the masks record where each max-pooling maximum was
def forwardPropagation(kWeights1, kBias1, kWeights2, kBias2, fcWeights, fcBias, X):
    layer1Maps = []
    layer1Mask = []
    activatedLayer1Maps = []

    for i in range(kWeights1.shape[3]):
        kernel = kWeights1[:, :, :, i]

        map = correlate(X, kernel, mode='valid')

        map = map + kBias1[i]

        activatedMap = ReLU(map)

        activatedMap, _ = max_pooling(activatedMap, pool_size = 2, stride = 2)
        map, mapMask = max_pooling(map, pool_size=2, stride=2)

        layer1Mask.append(mapMask)
        layer1Maps.append(map)
        activatedLayer1Maps.append(activatedMap)

    layer1Maps = np.stack(layer1Maps, axis = 2)
    layer1Mask = np.stack(layer1Mask, axis=2)
    activatedLayer1Maps = np.stack(activatedLayer1Maps, axis = 2)

    layer2Maps = []
    activatedLayer2Maps = []

    for i in range(kWeights2.shape[3]):
        kernel2 = kWeights2[:, :, :, i]

        map = np.zeros((8, 8))

        for j in range(activatedLayer1Maps.shape[2]):
            temp = correlate2d(activatedLayer1Maps[:, :, j, 0], kernel2[:, :, j], mode="valid")
            map += temp

        map = map + kBias2[i]

        activatedMap = sigmoid(map)

        layer2Maps.append(map)
        activatedLayer2Maps.append(activatedMap)

    layer2Maps = np.stack(layer2Maps, axis = -1)
    activatedLayer2Maps = np.stack(activatedLayer2Maps, axis = -1)

    layer2Maps, layer2Mask = max_pooling(layer2Maps, pool_size=2, stride = 2)
    activatedLayer2Maps, _ = max_pooling(activatedLayer2Maps, pool_size = 2, stride = 2)

    flattenedLayer2 = activatedLayer2Maps.reshape(-1)

    fcLayer = np.dot(fcWeights, flattenedLayer2) + fcBias

    activatedFcLayer = softMax(fcLayer)

    return layer1Maps, layer1Mask, activatedLayer1Maps, layer2Maps, layer2Mask, activatedLayer2Maps, fcLayer, activatedFcLayer


# works backwards through the network for one image
# -> names ending in Loss hold the gradient of the loss with respect to that layer, not the loss itself
def backPropagation(kWeights1, kBias1, kWeights2, kBias2, fcWeights, fcBias, 
                    layer1Maps, layer1Mask, activatedLayer1Maps, layer2Maps, layer2Mask, activatedLayer2Maps, 
                    fcLayer, activatedFcLayer, X, Y):
    
    oneHotY = oneHotEncoding(Y).reshape(-1)

    # populate with zeros
    dKWeights1, dKBias1 = np.zeros_like(kWeights1), np.zeros_like(kBias1)
    dKWeights2, dKBias2 = np.zeros_like(kWeights2), np.zeros_like(kBias2)
    dFcWeights, dFcBias = np.zeros_like(fcWeights), np.zeros_like(fcBias)

    # gradient of the cross-entropy loss with respect to the fully connected layer
    # -> for softmax followed by cross-entropy this simplifies to the output minus the one-hot label
    loss = activatedFcLayer - oneHotY

    dFcWeights = np.outer(loss, activatedLayer2Maps.reshape(-1))
    dFcBias = loss
    dFlattenedLayer = np.dot(fcWeights.T, loss)

    # reforms layer 2 back to unflattened state
    layer2loss = dFlattenedLayer.reshape(activatedLayer2Maps.shape)

    # the sigmoid derivative can be applied to the pooled values: sigmoid always increases, so the
    # largest value before it is in the same cell as the largest value after it
    activatedLayer2Loss = layer2loss * sigmoidDerivative(layer2Maps)

    # spread each pooled gradient over its 2x2 block, then keep it only in the cell that held the maximum
    resizedActivatedLayer2Loss = np.repeat(np.repeat(activatedLayer2Loss, 2, axis=0), 2, axis=1) * layer2Mask

    # calculating gradients for layer 2 kernels
    for i in range(kWeights2.shape[3]):
        kernelGradient = np.zeros_like(kWeights2[:, :, :, i])

        for j in range(activatedLayer1Maps.shape[2]):
            kernelGradient[:, :, j] = correlate2d(activatedLayer1Maps[:, :, j, 0], resizedActivatedLayer2Loss[:, :, i], mode = 'valid')

        dKWeights2[:, :, :, i] = kernelGradient
        # each block passes on exactly one gradient, so the pooled gradients sum to the bias gradient
        dKBias2[i] = np.sum(activatedLayer2Loss[:, :, i])

    # calculate gradients and derivatives of layer 1
    layer1Loss = np.zeros((10, 10, 2))

    # send the gradient back through the layer 2 kernels: a full correlation with each kernel rotated 180 degrees
    for i in range(kWeights2.shape[3]):
        for j in range(kWeights2.shape[2]):

            layer1Loss[:, :, j] += correlate2d(resizedActivatedLayer2Loss[:, :, i], kWeights2[::-1, ::-1, j, i], mode = "full")

    # multiply layer 1 by relu derivative to stop applying gradient to unactivated positions
    # -> as with sigmoid, this works on the pooled values because relu never decreases
    activatedLayer1Loss = layer1Loss * ReLUDerivative(layer1Maps[:, :, :, 0])
    # as for layer 2, only the cell that held each maximum gets the gradient
    resizedActivatedLayer1Loss = np.repeat(np.repeat(activatedLayer1Loss, 2, axis=0), 2, axis=1) * layer1Mask[:, :, :, 0]

    for i in range(kWeights1.shape[3]):
        kernelGradient = correlate2d(X[:, :, 0], resizedActivatedLayer1Loss[:, :, i], mode="valid")

        dKWeights1[:, :, 0, i] = kernelGradient
        dKBias1[i] = np.sum(activatedLayer1Loss[:, :, i])

    return dKWeights1, dKBias1, dKWeights2, dKBias2, dFcWeights, dFcBias


# creates a matrix of the actual expected values that should come from the neural network
def oneHotEncoding(Y):
    oneHotEncodedY = np.zeros((10, Y.size))
    oneHotEncodedY[Y, np.arange(Y.size)] = 1

    return oneHotEncodedY


# gets the derivative of the ReLU function which is 1 when greater than 0
# boolean gets converted into 1 or 0 when going into calculations
def ReLUDerivative(Z):
    return Z > 0


# derivative of the sigmoid: s(1 - s)
def sigmoidDerivative(Z):
    sig = sigmoid(Z)
    return sig * (1 - sig)


# moves each variable a small step against its derivative, towards a minimum of the loss
def updateParams(kWeights1, kBias1, kWeights2, kBias2, fcWeights, fcBias, 
                 dkWeights1, dkBias1, dkWeights2, dkBias2, dfcWeights, dfcBias, learningRate):
    
    
    kWeights1 -= dkWeights1 * learningRate
    kWeights2 -= dkWeights2 * learningRate
    kBias1 -= dkBias1 * learningRate
    kBias2 -= dkBias2 * learningRate
    fcWeights -= dfcWeights * learningRate
    fcBias -= dfcBias * learningRate

    return kWeights1, kBias1, kWeights2, kBias2, fcWeights, fcBias


# returns the index of the value with biggest probability
def getPredictions(A3):
    return np.argmax(A3, 0)


# trains on one image at a time; each round uses imagesPerRound randomly chosen training images
# -> with imagesPerRound = 60000, every round is one full pass (an epoch) over the training set
# -> the learning rate is multiplied by 0.97 after each round
def gradientDescent(X, Y, imagesPerRound, learningRate, rounds, updateFrequency, seed):
    np.random.seed(seed)

    kweights1, kbias1, kweights2, kbias2, fcweights, fcbias = initParams()

    for j in range(rounds):
        correct = 0

        sampleIndices = np.random.choice(X.shape[0], size=imagesPerRound, replace = False)
        Xsample = X[sampleIndices, :, :, :]
        Ysample = Y[sampleIndices]

        for i in range(Xsample.shape[0]):
            layer1Maps, layer1Mask, activatedLayer1Maps, layer2Maps, layer2Mask, activatedLayer2Maps, fcLayer, activatedFcLayer = forwardPropagation(kweights1, kbias1, kweights2, kbias2, fcweights, fcbias, Xsample[i, :, :, :])
            correct += int(getPredictions(activatedFcLayer) == Ysample[i])
            dKWeights1, dKBias1, dKWeights2, dKBias2, dFcWeights, dFcBias = backPropagation(kweights1, kbias1, kweights2, kbias2, fcweights, fcbias, layer1Maps, layer1Mask, activatedLayer1Maps, layer2Maps, layer2Mask, activatedLayer2Maps, fcLayer, activatedFcLayer, Xsample[i, :, :, :], Ysample[i])
            kweights1, kbias1, kweights2, kbias2, fcweights, fcbias = updateParams(kweights1, kbias1, kweights2, kbias2, fcweights, fcbias, dKWeights1, dKBias1, dKWeights2, dKBias2, dFcWeights, dFcBias, learningRate)

        if j % updateFrequency == 0 or j == rounds - 1:
            print(f"Round: {j}")
            print(f"Accuracy: {correct / imagesPerRound:.2%}")
            print()

        learningRate = learningRate * 0.97

    return kweights1, kbias1, kweights2, kbias2, fcweights, fcbias


# accuracy on the given images, and the average certainty on the ones it got right
def test(kweights1, kbias1, kweights2, kbias2, fcweights, fcbias, X, y):
    count = 0
    certainty = 0

    for i in range(0, X.shape[0]):
        sampleX = X[i, :, :, :]
        sampleY = y[i]
        _, _, _, _, _, _, _, output = forwardPropagation(kweights1, kbias1, kweights2, kbias2, fcweights, fcbias, sampleX)

        prediction = getPredictions(output)

        if prediction == sampleY:
            count += 1
            certainty += output[prediction]

    return count / X.shape[0], certainty / count


def saveVariables(kweights1, kbias1, kweights2, kbias2, fcweights, fcbias, filename="cnnVariables.json"):
    data = {
        "kWeights1": kweights1.tolist(),
        "kWeights2": kweights2.tolist(),
        "fcWeights": fcweights.tolist(),
        "kBias1": kbias1.tolist(),
        "kBias2": kbias2.tolist(),
        "fcBias": fcbias.tolist()
    }

    with open(filename, "w") as f:
        json.dump(data, f, indent = 4)


def predict(kWeights1, kBias1, kWeights2, kBias2, fcWeights, fcBias, X):
    _, _, _, _, _, _, _, AFcLayer = forwardPropagation(kWeights1, kBias1, kWeights2, kBias2, fcWeights, fcBias, X)
    predictions = getPredictions(AFcLayer)

    certainty = AFcLayer[predictions]
    return predictions, certainty


def testPredictionSingle(index, kWeights1, kBias1, kWeights2, kBias2, fcWeights, fcBias):

    currentImage = test_X[index]
    prediction, certainty = predict(kWeights1, kBias1, kWeights2, kBias2, fcWeights, fcBias, currentImage)

    label = test_y[index]
    label = convertToTextLabel(label)

    print(f"Prediction: {convertToTextLabel(prediction)}")
    print(f"Certainty: {certainty:.2%}")
    print(f"Label: {label}")

    currentImage = currentImage.reshape((28, 28)) * 255
    pyplot.gray()
    pyplot.imshow(currentImage, interpolation='nearest')
    pyplot.show()


def loadVariables(filename="cnnVariables.json"):
    with open(filename, "r") as f:
        data = json.load(f)
    
    # Convert lists back to NumPy arrays
    weights1 = np.array(data["kWeights1"])
    biases1 = np.array(data["kBias1"])
    weights2 = np.array(data["kWeights2"])
    biases2 = np.array(data["kBias2"])
    weights3 = np.array(data["fcWeights"])
    biases3 = np.array(data["fcBias"])
    
    return weights1, biases1, weights2, biases2, weights3, biases3


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
            rounds = int(input("How many rounds would you like?\n\n"))
            print()

            learningRate = float(input("What would you like the learning rate to be (recommended 0.01)?\n\n"))
            print()

            imagesPerRound = int(input("How many randomly chosen images per round (recommended 3000)?\n\n"))
            print()

            updates = int(input("How often would you like updates (in rounds)?\n\n"))
            print()

            seed = int(input("Please enter a seed for rng:\n\n"))
            print()

            kweights1, kbias1, kweights2, kbias2, fcweights, fcbias = gradientDescent(train_X, train_y, imagesPerRound=imagesPerRound, rounds=rounds, learningRate=learningRate, updateFrequency=updates, seed=seed)
            saveVariables(kweights1, kbias1, kweights2, kbias2, fcweights, fcbias)
            print()

        elif option == 4:
            active = False

        else:
            kweights1, kbias1, kweights2, kbias2, fcweights, fcbias = loadVariables()

            if option == 2:
                index = int(input("What index would you like to test?\n\n"))

                testPredictionSingle(index, kweights1, kbias1, kweights2, kbias2, fcweights, fcbias)
                print()

            else:
                accuracy, certainty = test(kweights1, kbias1, kweights2, kbias2, fcweights, fcbias, test_X, test_y)
                print(f"Neural network has an accuracy of {accuracy:.2%}")
                print(f"Neural network has an average certainty (when correct) of {certainty:.2%}")
                print()

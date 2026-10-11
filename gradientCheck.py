import numpy as np
import threeLayer
import mnistBatches

# file checks that backprop is working properly by checking the gradients against numerical gradients worked out by nudging parameters

# calculate the loss
def crossEntropyLoss(A, Y):
    confidences = A[Y, np.arange(Y.size)]
    return -np.mean(np.log(confidences))

# loss given params for threeLayer
def threeLayerLoss(w1, b1, w2, b2, w3, b3, X, Y):
    _, _, _, _, _, A3 = threeLayer.forwardPropagation(w1, b1, w2, b2, w3, b3, X)
    return crossEntropyLoss(A3, Y)

# grads given params for threeLayer
def threeLayerGradients(w1, b1, w2, b2, w3, b3, X, Y):
    Z1, A1, Z2, A2, Z3, A3 = threeLayer.forwardPropagation(w1, b1, w2, b2, w3, b3, X)
    return threeLayer.backwardsPropogation(Z1, A1, Z2, A2, w2, Z3, A3, w3, X, Y)

# loss for mnistBatches
def mnistBatchesLoss(w1, b1, w2, b2, X, Y):
    _, _, _, A2 = mnistBatches.forwardPropagation(w1, b1, w2, b2, X)
    return crossEntropyLoss(A2, Y)

def mnistBatchesGradients(w1, b1, w2, b2, X, Y):
    Z1, A1, Z2, A2 = mnistBatches.forwardPropagation(w1, b1, w2, b2, X)
    return mnistBatches.backwardsPropogation(Z1, A1, Z2, A2, w2, X, Y)

# numerical gradients made by nudging each parameter
def numericalGradients(lossFunction, params, X, Y):
    eps = 1e-5
    results = []
    for parameter in params:
        numerical = np.zeros_like(parameter)
        for index in np.ndindex(parameter.shape):
            parameter[index] += eps
            lossPlus = lossFunction(*params, X=X, Y=Y)
            parameter[index] -= 2*eps
            lossMinus = lossFunction(*params, X=X, Y=Y)
            parameter[index] += eps
            numerical[index] = (lossPlus - lossMinus) / (2*eps)
        results.append(numerical)
    return results

# computes the relative error between numerial and analytical grads
def relativeError(analytical, numerical):
    analytical = np.broadcast_to(analytical, numerical.shape)
    return np.linalg.norm(analytical - numerical) / (np.linalg.norm(analytical) + np.linalg.norm(numerical))


if __name__ == "__main__":
    np.random.seed(0)
    params = list(threeLayer.initParams())
    X = threeLayer.train_X[:, :10]
    Y = threeLayer.train_y[:10]
    analytical = threeLayerGradients(*params, X=X, Y=Y)
    numerical = numericalGradients(threeLayerLoss, params, X, Y)

    print("------ Three layer ------")
    for name, a, n in zip(["W1", "b1", "W2", "b2", "W3", "b3"], analytical, numerical):
        print(f"{name}: relative error = {relativeError(a, n):.2e}")

    print("\n\n ------ mnistBatches ------")
    params = list(mnistBatches.initParams())
    X = mnistBatches.train_X[:, :10]
    Y = mnistBatches.train_y[:10]
    analytical = mnistBatchesGradients(*params, X=X, Y=Y)
    numerical = numericalGradients(mnistBatchesLoss, params, X, Y)

    for name, a, n in zip(["W1", "b1", "W2", "b2"], analytical, numerical):
        print(f"{name}: relative error = {relativeError(a, n):.2e}")
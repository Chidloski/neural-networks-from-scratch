import numpy as np

np.random.seed(0)

def createData(datapoints, classes):
    X = np.zeros((datapoints * classes, 2))
    y = np.zeros(datapoints * classes, dtype='uint8')

    for classNum in range(classes):
        ix = range(datapoints * classNum, datapoints * (classNum + 1))
        r = np.linspace(0.0, 1, datapoints)
        t = np.linspace(classNum * 4, (classNum + 1) * 4, datapoints) + np.random.randn(datapoints) * 0.2
        X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        y[ix] = classNum

    return X, y
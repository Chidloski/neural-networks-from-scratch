import numpy as np

inputs = [[0.9, 2.3, 4.7, 4],
          [1, 2, 3, 4],
          [5, 6, 7, 8],
          [2, 7.6, 3, 4.1]]

 

weights = [[0.2, 1.4, 2.9, 30],
           [3, 1.7, 5.9, 5],
           [2, 14, 4.9, 2],
           [34, 1, 4, 5.7]]

# flips weights to make dot product possible
weights = np.array(weights).T

biases = [2, 3, 4, 5]

l1Outputs = np.dot(inputs, weights) + biases

output = np.dot(l1Outputs, weights) + biases

print(output) 


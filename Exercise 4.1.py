#Exercise 4.1
# Modify Example 4.1 to perform logical OR.
# Note: The original Example 4.1 already used the outputs for OR ([0, 1, 1, 1]).
# We explicitly set it up here as OR for clarity.
from numpy import exp, array, random, dot

inputs = array([[0, 0], [0, 1], [1, 0], [1, 1]])
# Logical OR outputs
outputs = array([[0, 1, 1, 1]]).T
random.seed(1)
weights = 2 * random.random((2, 1)) - 1

def neuron(inputs, weights):
    output = 1 / (1 + exp(-(dot(inputs, weights))))
    return output

for iteration in range(50000):
    output = 1 / (1 + exp(-(dot(inputs, weights))))
    weights += dot(inputs.T, (outputs - output) * output * (1 - output))

x = array([1, 0])
print("Prediction for [1, 0]:", neuron(x, weights))

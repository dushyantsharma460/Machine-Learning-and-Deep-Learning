# Perceptron
# -----------
# It is the simplest unit (building block) of a neural network.
#
# Consists of:
# - Inputs x1, x2, …, xn
# - Weights w1, w2, …, wn
# - Summation and Activation (threshold) function
#
# Can only solve linearly separable problems (like AND, OR).
# Single-layer, single neuron.
#
#
# Neural Network
# ---------------
# A collection of multiple perceptrons arranged in layers (input, hidden, output).
# Can solve complex, non-linear problems (like XOR, image recognition).
# Uses activation functions (ReLU, sigmoid, tanh, etc.) instead of just a step function.
# Has learning ability through backpropagation and gradient descent.
# Can have multiple hidden layers → known as Deep Neural Network (DNN).
#
#
# ✅ In short:
# Perceptron = one neuron.
# Neural Network = many perceptrons connected together.

# A Perceptron can be trained to represent logical gates like AND and OR because these are linearly separable problems.
# Now train perceptron for logical and

# • Inputs: x1, x2
# • Weights: w1, w2
# • Bias: b
# • Activation Function: Step function

# Writing Step function for this perceptron (Linear Function)
def step(x):
    return 1 if x >= 0 else 0

def perceptron(x1, x2, w1, w2, b):
    y = x1*w1 + x2*w2 + b
    return step(y)

print(perceptron(0, 0, 1, 1, -1.5))
print(perceptron(0, 1, 1, 1, -1.5))
print(perceptron(1, 0, 1, 1, -1.5))
print(perceptron(1, 1, 1, 1, -1.5))

# (1, 1, -1.5) -> (w1, w2, b) these are the solution of my perceptron
# We use linear funtion (step) and there is non linear funtion also 
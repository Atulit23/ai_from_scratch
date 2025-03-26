import numpy as np

class Perceptron:
    def __init__(self, input_size, lr=0.01, epochs=1000):
        self.weights = np.zeros(input_size + 1)  
        self.lr = lr
        self.epochs = epochs

    def activation(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        x = np.insert(x, 0, 1) 
        return self.activation(np.dot(self.weights, x))

    def train(self, X, y):
        for _ in range(self.epochs):
            for i in range(len(X)):
                x_i = np.insert(X[i], 0, 1)  
                y_pred = self.activation(np.dot(self.weights, x_i))
                error = y[i] - y_pred
                self.weights += self.lr * error * x_i 

    def evaluate(self, X, y):
        correct = sum(self.predict(x) == y[i] for i, x in enumerate(X))
        return correct / len(y)

# Simple AND gate
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])  

perceptron = Perceptron(input_size=2, lr=0.1, epochs=10)
perceptron.train(X, y)

for i in range(len(X)):
    print(f"Input: {X[i]} -> Predicted: {perceptron.predict(X[i])}")

import numpy as np
from tensorflow.keras.datasets import mnist

def one_hot_encode(y, num_classes):
    one_hot = np.zeros((y.size, num_classes))
    one_hot[np.arange(y.size), y] = 1
    return one_hot

def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

def leaky_relu_derivative(z, alpha=0.01):
    return np.where(z > 0, 1, alpha)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy_loss(y_pred, y_true):
    n_samples = y_true.shape[0]
    y_pred_clipped = np.clip(y_pred, 1e-9, 1 - 1e-9)
    loss = -np.sum(y_true * np.log(y_pred_clipped)) / n_samples
    return loss

def cross_entropy_derivative(y_pred, y_true):
    return y_pred - y_true

class DenseLayer:
    def __init__(self, input_size, output_size, learning_rate, activation=None, momentum=0.9):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.biases = np.zeros((1, output_size))
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.v_w = np.zeros_like(self.weights)
        self.v_b = np.zeros_like(self.biases)
        self.activation = activation

    def forward(self, inputs):
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.biases
        
        if self.activation == 'leaky_relu':
            self.output = leaky_relu(self.z)
        elif self.activation == 'softmax':
            self.output = softmax(self.z)
        else:
            self.output = self.z
            
        return self.output

    def backward(self, dvalues):
        if self.activation == 'leaky_relu':
            dvalues = dvalues * leaky_relu_derivative(self.z)
        
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)
        
        clip_value = 5.0
        self.dweights = np.clip(self.dweights, -clip_value, clip_value)
        self.dbiases = np.clip(self.dbiases, -clip_value, clip_value)
        
        return self.dinputs

    def update_params(self):
        weight_decay = 1e-4
        l2_penalty = weight_decay * self.weights
        
        self.v_w = self.momentum * self.v_w - self.learning_rate * (self.dweights + l2_penalty)
        self.v_b = self.momentum * self.v_b - self.learning_rate * self.dbiases
        
        self.weights = np.clip(self.weights + self.v_w, -10, 10)
        self.biases = np.clip(self.biases + self.v_b, -10, 10)

class DropoutLayer:
    def __init__(self, rate):
        self.rate = rate

    def forward(self, inputs, training=True):
        self.inputs = inputs
        if training:
            self.mask = np.random.binomial(1, 1 - self.rate, size=inputs.shape) / (1 - self.rate)
            self.output = inputs * self.mask
        else:
            self.output = inputs
        return self.output

    def backward(self, dvalues):
        self.dinputs = dvalues * self.mask
        return self.dinputs

class NeuralNetwork:
    def __init__(self, learning_rate, decay_rate, epochs):
        self.layers = []
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.epochs = epochs
        self.initial_lr = learning_rate
        self.batch_size = 128  

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, X, training=True):
        for layer in self.layers:
            if isinstance(layer, DropoutLayer):
                X = layer.forward(X, training)
            else:
                X = layer.forward(X)
        return X

    def backward(self, dvalues):
        for layer in reversed(self.layers):
            dvalues = layer.backward(dvalues)

    def update_params(self):
        for layer in self.layers:
            if isinstance(layer, DenseLayer):
                layer.update_params()

    def train(self, X_train, y_train, X_val, y_val):
        n_samples = X_train.shape[0]
        
        for epoch in range(self.epochs):
            indices = np.random.permutation(n_samples)
            X_train = X_train[indices]
            y_train = y_train[indices]
            
            epoch_loss = 0
            epoch_accuracy = 0
            n_batches = n_samples // self.batch_size
            
            for batch in range(n_batches):
                start_idx = batch * self.batch_size
                end_idx = start_idx + self.batch_size
                
                batch_X = X_train[start_idx:end_idx]
                batch_y = y_train[start_idx:end_idx]
                
                y_pred = self.forward(batch_X, training=True)
                loss = cross_entropy_loss(y_pred, batch_y)
                
                predictions = np.argmax(y_pred, axis=1)
                accuracy = np.mean(predictions == np.argmax(batch_y, axis=1))
                
                self.backward(cross_entropy_derivative(y_pred, batch_y))
                self.update_params()
                
                epoch_loss += loss
                epoch_accuracy += accuracy
            
            epoch_loss /= n_batches
            epoch_accuracy /= n_batches
            
            val_pred = self.forward(X_val, training=False)
            val_loss = cross_entropy_loss(val_pred, y_val)
            val_accuracy = np.mean(np.argmax(val_pred, axis=1) == np.argmax(y_val, axis=1))
            
            self.learning_rate = self.initial_lr * (1.0 / (1.0 + self.decay_rate * epoch))
            for layer in self.layers:
                if isinstance(layer, DenseLayer):
                    layer.learning_rate = self.learning_rate
            
            print(f"Epoch {epoch+1}/{self.epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, LR: {self.learning_rate:.6f}")

(X_train, y_train), (X_val, y_val) = mnist.load_data()
X_train = X_train.reshape(-1, 28 * 28) / 255.0
X_val = X_val.reshape(-1, 28 * 28) / 255.0
y_train = one_hot_encode(y_train, 10)
y_val = one_hot_encode(y_val, 10)

nn = NeuralNetwork(learning_rate=0.01, decay_rate=0.001, epochs=20)  
nn.add_layer(DenseLayer(input_size=28*28, output_size=128, learning_rate=0.01, activation='leaky_relu'))
nn.add_layer(DropoutLayer(rate=0.2))
nn.add_layer(DenseLayer(input_size=128, output_size=64, learning_rate=0.01, activation='leaky_relu'))
nn.add_layer(DropoutLayer(rate=0.2))
nn.add_layer(DenseLayer(input_size=64, output_size=10, learning_rate=0.01, activation='softmax'))

nn.train(X_train, y_train, X_val, y_val)
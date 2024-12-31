import numpy as np
from typing import Dict, List, Tuple
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import time

class Operation(object):

    def __init__(self):
        pass

    def forward(self, input_: np.ndarray) -> np.ndarray:

        self.input_ = input_

        self.output = self._output()

        return self.output

    def backward(self, output_grad: np.ndarray) -> np.ndarray:

        self.input_grad = self._input_grad(output_grad)

        return self.input_grad

    def _output(self) -> np.ndarray:
        raise NotImplementedError()

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class ParamOperation(Operation):

    def __init__(self, param: np.ndarray) -> np.ndarray:
        super().__init__()
        self.param = param

    def backward(self, output_grad: np.ndarray) -> np.ndarray:

        self.input_grad = self._input_grad(output_grad)
        self.param_grad = self._param_grad(output_grad)

        return self.input_grad

    def _param_grad(self, output_grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
    

class Conv2D_Op(ParamOperation):
    def __init__(self, W: np.ndarray):
        super().__init__(W)
        self.param_size = W.shape[2]
        self.param_pad = self.param_size // 2
    
    def _pad_1d(self, inp: np.ndarray) -> np.ndarray:
        z = np.array([0])
        z = np.repeat(z, self.param_pad)
        return np.concatenate([z, inp, z])

    def _pad_1d_batch(self, inp: np.ndarray) -> np.ndarray:
        outs = [self._pad_1d(obs) for obs in inp]
        return np.stack(outs)

    def _pad_2d_obs(self, inp: np.ndarray):
        inp_pad = self._pad_1d_batch(inp)

        other = np.zeros((self.param_pad, inp.shape[0] + self.param_pad * 2))

        return np.concatenate([other, inp_pad, other])

    def _pad_2d_channel(self, inp: np.ndarray):
        return np.stack([self._pad_2d_obs(channel) for channel in inp])

    def _get_image_patches(self, input_: np.ndarray):
        imgs_batch_pad = np.stack([self._pad_2d_channel(obs) for obs in input_])
        patches = []
        img_height = imgs_batch_pad.shape[2]
        for h in range(img_height - self.param_size + 1):
            for w in range(img_height - self.param_size + 1):
                patch = imgs_batch_pad[:, :, h : h + self.param_size, w : w + self.param_size]
                patches.append(patch)
        return np.stack(patches)
    
    def _output(self):
        batch_size = self.input_.shape[0]
        img_height = self.input_.shape[2]
        img_size = self.input_.shape[2] * self.input_.shape[3]
        patch_size = self.param.shape[0] * self.param.shape[2] * self.param.shape[3]

        patches = self._get_image_patches(self.input_)
        
        input_scale = np.sqrt(1.0 / (patch_size * img_size))
        patches_reshaped = patches.transpose(1, 0, 2, 3, 4).reshape(batch_size, img_size, -1) * input_scale
        param_reshaped = self.param.transpose(0, 2, 3, 1).reshape(patch_size, -1) * input_scale

        output_reshaped = (
            np.clip(np.matmul(patches_reshaped, param_reshaped), -1e3, 1e3)
            .reshape(batch_size, img_height, img_height, -1)
            .transpose(0, 3, 1, 2)
        )

        return output_reshaped
    
    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:

        batch_size = self.input_.shape[0]
        img_size = self.input_.shape[2] * self.input_.shape[3]
        img_height = self.input_.shape[2]

        output_patches = (
            self._get_image_patches(output_grad)
            .transpose(1, 0, 2, 3, 4)
            .reshape(batch_size * img_size, -1)
        )

        param_reshaped = self.param.reshape(self.param.shape[0], -1).transpose(1, 0)

        return (
            np.matmul(output_patches, param_reshaped)
            .reshape(batch_size, img_height, img_height, self.param.shape[0])
            .transpose(0, 3, 1, 2)
        )

    def _param_grad(self, output_grad: np.ndarray) -> np.ndarray:

        batch_size = self.input_.shape[0]
        img_size = self.input_.shape[2] * self.input_.shape[3]
        in_channels = self.param.shape[0]
        out_channels = self.param.shape[1]

        in_patches_reshape = (
            self._get_image_patches(self.input_).reshape(batch_size * img_size, -1).transpose(1, 0)
        )

        out_grad_reshape = output_grad.transpose(0, 2, 3, 1).reshape(batch_size * img_size, -1)

        return (
            np.matmul(in_patches_reshape, out_grad_reshape)
            .reshape(in_channels, self.param_size, self.param_size, out_channels)
            .transpose(0, 3, 1, 2)
        )
    
class Layer(object):

    def __init__(self, neurons: int) -> None:
        self.neurons = neurons
        self.first = True
        self.params: List[np.ndarray] = []
        self.param_grads: List[np.ndarray] = []
        self.operations: List[Operation] = []

    def _setup_layer(self, input_: np.ndarray) -> None:
        pass

    def forward(self, input_: np.ndarray) -> np.ndarray:

        if self.first:
            self._setup_layer(input_)
            self.first = False

        self.input_ = input_

        for operation in self.operations:

            input_ = operation.forward(input_)

        self.output = input_

        return self.output

    def backward(self, output_grad: np.ndarray) -> np.ndarray:

        for operation in self.operations[::-1]:
            output_grad = operation.backward(output_grad)

        input_grad = output_grad

        self._param_grads()

        return input_grad

    def _param_grads(self) -> None:

        self.param_grads = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.param_grads.append(operation.param_grad)

    def _params(self) -> None:

        self.params = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.params.append(operation.param)
                
class Linear(Operation):
    def __init__(self) -> None:
        super().__init__()

    def _output(self) -> np.ndarray:
        return self.input_

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        return output_grad

class Flatten(Operation):
    def __init__(self):
        super().__init__()

    def _output(self) -> np.ndarray:
        return self.input_.reshape(self.input_.shape[0], -1)

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        return output_grad.reshape(self.input_.shape)

class Conv2D(Layer):

    def __init__(
        self,
        out_channels: int,
        param_size: int,
        weight_init: str = "normal",
        activation: Operation = Linear(),
        flatten: bool = False,
    ) -> None:
        super().__init__(out_channels)
        self.param_size = param_size
        self.activation = activation
        self.flatten = flatten
        self.weight_init = weight_init
        self.out_channels = out_channels

    def _setup_layer(self, input_: np.ndarray) -> np.ndarray:

        self.params = []
        in_channels = input_.shape[1]

        if self.weight_init == "glorot":
            scale = 2 / (in_channels + self.out_channels)
        else:
            scale = 1.0

        conv_param = np.random.normal(
            loc=0,
            scale=scale,
            size=(
                input_.shape[1], 
                self.out_channels,
                self.param_size,
                self.param_size,
            ),
        )

        self.params.append(conv_param)

        self.operations = []
        self.operations.append(Conv2D_Op(conv_param))
        self.operations.append(self.activation)

        if self.flatten:
            self.operations.append(Flatten())
            
        return None

class Dense(Layer):
    def __init__(
        self,
        neurons: int,
        weight_init: str = "normal",
        activation: Operation = Linear()
    ) -> None:
        super().__init__(neurons)
        self.weight_init = weight_init
        self.activation = activation

    def _setup_layer(self, input_: np.ndarray) -> None:
        if self.weight_init == "glorot":
            scale = 2.0 / (input_.shape[1] + self.neurons)
        else:
            scale = 1.0

        self.params = []
        
        self.params.append(
            np.random.normal(
                loc=0,
                scale=scale,
                size=(input_.shape[1], self.neurons)
            )
        )
        
        self.params.append(
            np.random.normal(
                loc=0,
                scale=scale,
                size=(1, self.neurons)
            )
        )

        self.operations = []
        
        self.operations.append(WeightMultiply(self.params[0]))
        
        self.operations.append(BiasAdd(self.params[1]))
        
        self.operations.append(self.activation)
        
class WeightMultiply(ParamOperation):
    def __init__(self, W: np.ndarray):
        super().__init__(W)
    
    def _output(self) -> np.ndarray:
        scale = np.sqrt(1.0 / self.param.shape[0])
        return np.clip(np.matmul(self.input_ * scale, self.param * scale), -1e3, 1e3)
    
    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        return np.clip(np.matmul(output_grad, self.param.T), -1e3, 1e3)
    
    def _param_grad(self, output_grad: np.ndarray) -> np.ndarray:
        return np.clip(np.matmul(self.input_.T, output_grad), -1e3, 1e3)
    
class BiasAdd(ParamOperation):
    def __init__(self, B: np.ndarray):
        super().__init__(B)
    
    def _output(self) -> np.ndarray:
        return self.input_ + self.param
    
    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        return output_grad
    
    def _param_grad(self, output_grad: np.ndarray) -> np.ndarray:
        return np.sum(output_grad, axis=0).reshape(1, -1)

class NeuralNetwork():
    def __init__(self, layers: List[Layer]) -> None:
        self.layers = layers
        
    def forward(self, x_batch: np.ndarray, inference: bool = False) -> np.ndarray:
        output = x_batch
        for layer in self.layers:
            output = layer.forward(output)
            
        if not inference:
            output_shift = output - np.max(output, axis=1, keepdims=True)
            exp_scores = np.exp(output_shift)
            return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return output

    def backward(self, loss_grad: np.ndarray) -> None:
        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
            
    def get_params_and_gradients(self):
        params_and_grads = []
        for layer in self.layers:
            if hasattr(layer, 'params'):
                for param, param_grad in zip(layer.params, layer.param_grads):
                    params_and_grads.append((param, param_grad))
        return params_and_grads


class Trainer:
    def __init__(
        self,
        model,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        num_epochs: int = 10
    ):
        self.model = model
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def generate_batches(self, X: np.ndarray, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate batches for training"""
        num_samples = X.shape[0]
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        
        for i in range(0, num_samples, self.batch_size):
            batch_indices = indices[i:min(i + self.batch_size, num_samples)]
            yield X[batch_indices], y[batch_indices]
            
    def calculate_loss(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
        return -np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))
    
    def calculate_accuracy(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        return np.mean(np.argmax(predictions, axis=1) == np.argmax(targets, axis=1)) * 100
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        print("Starting training...")
        for epoch in range(self.num_epochs):
            print(epoch)
            start_time = time.time()
            epoch_losses = []
            epoch_accuracies = []
            
            for batch_X, batch_y in self.generate_batches(X_train, y_train):
                predictions = self.model.forward(batch_X)
                loss = self.calculate_loss(predictions, batch_y)
                accuracy = self.calculate_accuracy(predictions, batch_y)
                print(loss)
                print(accuracy)
                self.model.backward(predictions - batch_y)
                
                for layer in self.model.layers:
                    if hasattr(layer, 'params'):
                        for param, param_grad in zip(layer.params, layer.param_grads):
                            param -= self.learning_rate * param_grad
                
                epoch_losses.append(loss)
                epoch_accuracies.append(accuracy)
            
            # Calculate training metrics
            avg_loss = np.mean(epoch_losses)
            avg_accuracy = np.mean(epoch_accuracies)
            self.losses.append(avg_loss)
            self.train_accuracies.append(avg_accuracy)
            
            # Validation
            val_predictions = self.model.forward(X_val)
            val_loss = self.calculate_loss(val_predictions, y_val)
            val_accuracy = self.calculate_accuracy(val_predictions, y_val)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)
            
            epoch_time = time.time() - start_time
            
            print(f"Epoch {epoch+1}/{self.num_epochs}")
            print(f"Time: {epoch_time:.2f}s")
            print(f"Training Loss: {avg_loss:.4f} | Training Accuracy: {avg_accuracy:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%")
            print("-" * 50)
    
def load_mnist():
    mnist = fetch_openml('mnist_784', version=1)
    X = mnist.data.astype(np.float32).values
    y = mnist.target.astype(np.int32).values

    X /= 255.0

    X = X.reshape(-1, 1, 28, 28)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = load_mnist()

X_train, X_test = X_train - np.mean(X_train), X_test - np.mean(X_train)
X_train, X_test = X_train / np.std(X_train), X_test / np.std(X_train)

X_train_conv, X_test_conv = X_train.reshape(-1, 1, 28, 28), X_test.reshape(-1, 1, 28, 28)

num_labels = len(y_train)

train_labels = np.zeros((num_labels, 10))
for i in range(num_labels):
    train_labels[i][y_train[i]] = 1

num_labels = len(y_test)

test_labels = np.zeros((num_labels, 10))
for i in range(num_labels):
    test_labels[i][y_test[i]] = 1
    
def calc_accuracy_model(model, test_set):
    return print(f'''The model validation accuracy is: 
    {np.equal(np.argmax(model.forward(test_set, inference=True), axis=1), y_test).sum() * 100.0 / test_set.shape[0]:.2f}%''')
    

model = NeuralNetwork([
    Conv2D(out_channels=16, 
           param_size=3, 
           weight_init="glorot"),
    
    Conv2D(out_channels=32, 
           param_size=3, 
           weight_init="glorot", 
           flatten=True),
    
    Dense(neurons=10, 
          weight_init="glorot")
])

trainer = Trainer(
    model=model,
    batch_size=32,
    learning_rate=0.01,
    num_epochs=10
)

trainer.train(X_train_conv, train_labels, X_test_conv, test_labels)
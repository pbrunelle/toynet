# toynet

A toy neural networks library for solving toy problems.

## Building and Testing

```
cd toynet
mkdir -p build
cd build
bbcmake -64 ..
make && ./unit_tests.tsk
```

## Design

These are the main classes I want to implement, their meaning, and their relationships:

- `Activation`: an activation function
- `Layer`: a transformation on the post-activation output of the previous layer (or on the input for the first layer), and an optional activation function.
- `Workspace`: used for training, contains activations and gradients
- `Network`: contains layers, can perform inference (forward pass) and provide gradients
- `Optimizer`: given gradients and weights, updates weights (i.e. gradient descent)
- `Loss`: a loss function
- `Trainer`: given a corpus, a network, an optimizer, and a loss function, initializes and optimizes a network's parameters

Limitations:

- Activation functions currently aren't supported
- The fully connected layer currently doesn't support biases
- The fully connected layer currently only supports matrice transformations

# Simple Network: Compute Difference of Two Numbers

The goal of this document is to explain the simple network from module `diff`,
and show how algorithms 6.3 and 6.4 of the Deep Learning book are applied
to compute the gradient.  I'll use an example to show in details how the
forward and backward passes operate, i.e. how the activations and gradients
are computed.

## Network

We use the following network, made of 3 layers (1 input, 1 hidden, 1 output), 5 units, and 6 weights:

```
Output     U5       y_hat = [U5]
          /  \      2 weights: w_35, w_45
Hidden   U3 U4
         | X |      4 weights: w_13, w_14, w_23, w_24
Input    U1 U2      U1=x[0], U2=x[1]
```

The weight between units `i` and `j` (if a connection exists) is labelled `w_ij`.

There are no activation functions (i.e. all activations are linear).

The loss function is the mean square error (or just the square error in this
case since there's a single output node): `L(y, y_hat) = (y - y_hat)^2`.

## Training on a Single Example

We use a single training example: `x = (4, 3)`.
The weights are initialized as:

```
w_13 = -0.2,  w_14 = -0.1,  w_23 =  0.0,  w_24 =  0.1
w_35 =  0.2,  w_45 = -0.2
```

### First Pass

We can very simply compute the activations:

```
U1 = x[0] = 4.0
U2 = x[1] = 3.0
U3 = w_13 * U1 + w_23 * U2 = -0.2 *  4.0 +  0.0 *  3.0 = -0.8
U4 = w_14 * U1 + w_24 * U2 = -0.1 *  4.0 +  0.1 *  3.0 = -0.1
U5 = w_35 * U3 + w_45 * U4 =  0.2 * -0.8 + -0.2 * -0.1 = -0.14
```

And the loss:

```
L  = 1.14^2
```

We can then compute the gradient of the loss w.r.t. U5:

TODO

### Updating the Weights

### Second Pass

### After Convergence

## Training on Many Examples

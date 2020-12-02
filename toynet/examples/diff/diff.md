# Simple Network: Compute Difference of Two Numbers

The goal of this document is to explain the simple network from module `diff`,
and show how algorithms 6.3 and 6.4 of the Deep Learning book are applied
to compute the gradient.  I'll use an example to show in details how the
forward and backward passes operate, i.e. how the activations and gradients
are computed.

## Usage

Use default parameters:

```
$ ./diff.tsk

0 1.2996
1 1.16942
2 1.06337
3 0.971817
4 0.888807
5 0.81054
6 0.734664
7 0.659891
8 0.585777
9 0.512574
hidden 1 width 2 inputs 2 loss 0.512574
W : [[[-0.0879392, -0.269353], [0.0840456, -0.027015]], [[0.116585], [-0.299946]]]
DW: [[[-0.6794, 1.63131], [-0.50955, 1.22348]], [[0.203446], [1.51279]]]
A : [[4, 3], [-0.142082, -1.0565], [0.284057]]
G : [[-0.0871063, -0.0194374], [-0.16985, 0.407827], [-1.43189]]
```

Explicitly specify the default parameters.  We expect identical output.

```
$ ./diff.tsk -H 1 -W 2 -I 2 -e 10 --lr 0.01 --training "[[4.0, 3.0]]"
```

Use a higher learning rate, which in this case improves convergence speed:

```
$ ./diff.tsk --progress off --lr 0.05

hidden 1 width 2 inputs 2 loss 3.0573e-22
W : [[[-0.0242469, -0.428293], [0.131815, -0.14622]], [[0.100517], [-0.450779]]]
DW: [[[1.40604e-11, -6.30554e-11], [1.05453e-11, -4.72916e-11]], [[1.04371e-11], [-7.52501e-11]]]
A : [[4, 3], [0.298457, -2.15183], [1]]
G : [[6.66632e-12, 2.76833e-12], [3.5151e-12, -1.57639e-11], [3.49702e-11]]
```

Make sure the network also works if we reverse the order of the 2 values
(note that we don't expect the loss to be the same because the weights of
the network are not initialized in a symmetrical way):

```
$ ./diff.tsk --progress off --training "[[3.0, 4.0]]"

hidden 1 width 2 inputs 2 loss 0.16695
W : [[[-0.294739, -0.0166118], [-0.126319, 0.211184]], [[0.318955], [-0.251357]]]
DW: [[[0.755365, -0.601298], [1.00715, -0.80173]], [[-1.08404], [0.608638]]]
A : [[3, 4], [-1.32655, 0.744794], [-0.591405]]
G : [[-0.0677753, -0.0699911], [0.251788, -0.200433], [0.817189]]
```

Provide more than one example so the network can properly learn that it must
compute `x[0] - x[1]`:

```
$ ./diff.tsk --progress off -e 60 --training "[[4.0, 3.0], [3.0, 4.0], [-1.0, 5.0], [9.8, -3.1]]"

hidden 1 width 2 inputs 2 loss 5.04131e-30
W : [[[0.0119362, -0.845473], [-0.138718, 0.833658]], [[0.110082], [-1.18122]]]
DW: [[[8.31066e-16, -8.91761e-15], [1.72324e-15, -1.84909e-14]], [[-2.08141e-15], [6.66729e-15]]]
A : [[3.95, 2.225], [-0.2615, -1.48473], [1.725]]
G : [[3.88578e-15, -3.88578e-15], [4.27755e-16, -4.58994e-15], [3.88578e-15]]
```

Note that training can easily diverge.  I want to investigate this further to make sure this
divergence is not due to a bug in my code.

```
$ ./diff.tsk -e 5 --lr 0.5

0 1.2996
1 7.11609
2 11060.1
3 7.55131e+14
4 2.69121e+47
hidden 1 width 2 inputs 2 loss 2.69121e+47
W : [[[1.85049e+36, -2.3682e+36], [1.38786e+36, -1.77615e+36]], [[-1.59163e+36], [-1.00787e+36]]]
DW: [[[-3.70097e+36, 4.73639e+36], [-2.77573e+36, 3.55229e+36]], [[3.18326e+36], [2.01575e+36]]]
A : [[4, 3], [-3.06809e+12, -1.94282e+12], [-5.18769e+23]]
G : [[8.61187e+46, 6.4589e+46], [-9.25243e+35, 1.1841e+36], [-1.03754e+24]]
```

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

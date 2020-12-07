# Simple Network: Compute Difference of Two Numbers

Diff2 has been copied from Diff.  We are changing the design
to make it easier to add optimizers.

In Diff, the optimization function is directly part of the
network, and the network also keeps track of all data
needed by the optimization function (i.e. the gradients
over the weights 'DW` and the activations `G`).

We divide the network into 5 components:
- `Loss`: the loss function
- `Network`: contains the weights and is able to run the forward pass (i.e. predict)
- `Workspace`: contains data structures to hold weights and gradients used during
  the forward and backpropagation steps.
- `Optimizer`: base class with derived classes `GradientOptimizer` and `MomentumOptimizer`,
  are responsible to compute whatever gradient or velocity and update weights; tensors
  to store the gradients and weights are provided by caller
- `Trainer`: the maestro; given a `network`, a `loss` and an `optimizer`, takes care of
  initializing the weights, creating an appropriate workspace, and training for a
  number of epochs by doing a forward pass, computing the loss, doing a backward pass (and
  computing whichever gradient or velocity is required by the optimizer), and updating
  the weights.

## Commands

Tentative steps towrads word2vec:

```
$ ./diff2.tsk --inputs 4 --outputs 4 --loss softmax --lr 5 --epochs 10000 --progress false \
    --trainingx "[[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]" \
    --trainingy "[[0,1,0,0], [0,0,1,0], [0,0,0,1], [1,0,0,0]]" \
    --testing "[[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1], [0,0,0,0], [1,1,1,1], [1,0,1,0]]"

loss 2.8595e-06 weights [[[1.60526, -3.12376, -1.07817, 3.64396], [3.61117, 1.52653, -3.30446, -1.39427]], [[3.34515, -1.2111], [1.45876, 3.26241], [-3.48308, 1.67641], [-1.32082, -3.62772]]]
[1, 0, 0, 0] -> [0.996357, 14.1228, 0.462534, -15.2206]
[0, 1, 0, 0] -> [-12.2982, 0.423354, 13.4394, -1.41188]
[0, 0, 1, 0] -> [0.395383, -12.3533, -1.78426, 13.4117]
[0, 0, 0, 1] -> [13.8782, 0.766961, -15.0296, 0.245002]
[0, 0, 0, 0] -> [0, 0, 0, 0]
[1, 1, 1, 1] -> [2.9717, 2.95982, -2.9119, -2.97572]
[1, 0, 1, 0] -> [1.39174, 1.7695, -1.32173, -1.80884]
```

Notes:
- convergence is slow, therefore we run for 10,000 epochs
- **the softmax should be treated as the output's activation function**, not as part of the loss function

Using momentum improves convergence speed a lot:

```
$ ./diff2.tsk --inputs 4 --outputs 4 --loss softmax --optimizer momentum --lr 1.5 --alpha 0.97 --epochs 30 --progress false \
    --trainingx "[[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]" \
    --trainingy "[[0,1,0,0], [0,0,1,0], [0,0,0,1], [1,0,0,0]]" \
    --testing "[[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1], [0,0,0,0], [1,1,1,1], [1,0,1,0]]"

loss 0 weights [[[5.41641, -7.26237, 3.31467, 9.49974], [9.68141, 12.2268, -13.3618, -3.07268]], [[8.57214, -2.90461], [3.80086, 9.41443], [-12.1174, 9.09306], [-0.255582, -15.5029]]]
[1, 0, 0, 0] -> [18.3096, 111.732, 22.4007, -151.474]
[0, 1, 0, 0] -> [-97.7682, 87.5053, 199.18, -187.695]
[0, 0, 1, 0] -> [67.2245, -113.195, -161.665, 206.299]
[0, 0, 0, 1] -> [90.3581, 7.17965, -143.053, 45.2075]
[0, 0, 0, 0] -> [0, 0, 0, 0]
[1, 1, 1, 1] -> [78.124, 93.2222, -83.136, -87.6628]
[1, 0, 1, 0] -> [85.5341, -1.46279, -139.264, 54.8246]
```

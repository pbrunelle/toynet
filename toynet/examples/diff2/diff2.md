# Simple Network: Compute Difference of Two Numbers

Diff2 has been copied from Diff.  We are changing the design
to make it easier to add optimizers.

In Diff, the optimization function is directly part of the
network, and the network also keeps track of all data
needed by the optimization function (i.e. the gradients
over the weights 'DW` and the activations `G`).

We divide the network into 4 components:
- `Network`: only contains the weights
- `Forward`: performs the forward pass, tensors to store the activations are provided by caller
- `Optimizer`: base class with derived classes `GradientOptimizer` and `MomentumOptimizer`,
  are responsible to compute whatever gradient or velocity and update weights; tensors
  to store the gradients and weights are provided by caller
- `Trainer`: the maestro; given a `network`, a `forward` and an `optimizer`, takes care of
  initializing the weights, creating appropriate data structures, and training for a
  number of epochs by doing a forward pass, computing the loss, doing a backward pass (and
  computing whichever gradient or velocity is required by the optimizer), and updating
  the weights.

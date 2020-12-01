#include <toynet/ublas/ublas.h>
#include <iostream>
#include <vector>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>

namespace toynet {

// Inmplementation of algorithms 6.3 and 6.4 (pp. 205-6) of the "Deep Learning"
// book on a toy network.
//
// Our toy network has `inputs` input units.
// The goal is to learn f(x) = x[0] - x[1] + x[2] - x[3] ...
// We use the squared error as our loss function: loss = (y - y_hat)^2
// We implement a fully connected feed forward network with `hidden` hidden
// layers, each of which has `width` units; `hidden` can be 0.
// All layers have a *linear* activation function (i.e. no activation function)
// There are no bias terms.
// There are no regularization terms.
// We perform gradient descent (i.e. no SGD).
// We perform simple weight update: `W' = W - lr * DW`.
//
// See `diff.md` for a complete description.
struct DiffNumbers {
    DiffNumbers(int hidden, int width, int inputs);

    // Deterministically initialize weights.
    void init_weights();

    // Given a list of training examples `training_set`, each containing
    // a vector of length `inputs`, compute the activations and the loss
    // (forward pass) and the gradients on the weights and activations
    // (backward pass).  Update fields `A`, `loss`, `DW` and `G`.
    // This operation is idem-potent.
    void forward_backward(const std::vector<ublas::vector<double>>& training_set);

    // Use the computed gradients `DW`, the current weights `W` and the
    // learning rate `lr` to update `W`.  This method should be called
    // exactly once after each call to forward_backward.
    void update_weights(double lr);

    std::string print() const;

    // The weights of the FFN, stored as:
    // W[layer][unit index of previous layer][unit index of layer]
    // where layer = 0 is the layer closest to the input
    std::vector<ublas::matrix<double>> W;

    // The gradients of the loss for a training set w.r.t. the weights
    std::vector<ublas::matrix<double>> DW;

    // The pre-activations (since everything is linear in our network,
    // there are no post-activations), stored as:
    // A[layer][unit]
    std::vector<ublas::vector<double>> A;

    // The gradients of the loss for a given example w.r.t. the pre-activations
    std::vector<ublas::vector<double>> G;

    int hidden;  // the number of hidden layers
    int width;  // the width of each hidden layer
    int inputs;  // the width of the input layer
    double loss;  // the last computed loss
};

std::ostream& operator<<(std::ostream& os, const DiffNumbers& obj);

} // namespace toynet

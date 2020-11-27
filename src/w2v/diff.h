#include <iostream>
#include <vector>

namespace w2v {

// Inmplementation of algorithms 6.3 and 6.4 of the "Deep Learning" book on a
// toy network.
//
// Our toy network has `inputs` input units.
// The goal is to learn f(x) = x[0] - x[1] + x[2] - x[3] ...
// We use MSE as our loss function.
// We implement a fully connected FFN.  There are `hidden` hidden layers,
// each of which has `width` units; `hidden` can be 0.
struct DiffNumbers {
    DiffNumbers(int hidden, int width, int inputs);

    void init_weights();

    void forward_backward(const std::vector<double>& x);

    // The weights of the FFN
    // W[layer][unit index of previous layer][unit index of layer]
    // Where layer = 0 is the layer closest to the input
    std::vector<std::vector<std::vector<double>>> W;

    // The gradients of the loss for a given example w.r.t. the weights
    std::vector<std::vector<std::vector<double>>> DW;

    // The pre-activations (since everything is linear in our network,
    // there are no post-activations)
    // A[layer][unit]
    std::vector<std::vector<double>> A;

    // The gradients of the loss for a given example w.r.t. the pre-activations
    std::vector<std::vector<double>> G;

    int hidden;  // the number of hidden layers
    int width;  // the width of each hidden layer
    int inputs;  // the width of the input layer
    double loss;  // the last computed loss
};

std::ostream& operator<<(std::ostream& os, const DiffNumbers& obj);

} // namespace w2v

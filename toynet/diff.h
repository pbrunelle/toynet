#include <iostream>
#include <vector>
#include <boost/numeric/ublas/matrix.hpp>

// https://stackoverflow.com/a/1211402
namespace ublas = boost::numeric::ublas;

namespace toynet {

// Inmplementation of algorithms 6.3 and 6.4 (pp. 205-6) of the "Deep Learning"
// book on a toy network.
//
// Our toy network has `inputs` input units.
// The goal is to learn f(x) = x[0] - x[1] + x[2] - x[3] ...
// We use MSE as our loss function: loss = (y - y_hat)^2
// We implement a fully connected FFN with `hidden` hidden layers, each of
// which has `width` units; `hidden` can be 0.
// All layers have a *linear* activation function (i.e. no activation function)
// There are no bias terms.
// There are no regularization terms.
struct DiffNumbers {
    DiffNumbers(int hidden, int width, int inputs);

    void init_weights();

    void forward_backward(const std::vector<ublas::vector<double>>& training_set);

    void update_weights(double lr);

    std::string print() const;

    // The weights of the FFN
    // W[layer][unit index of previous layer][unit index of layer]
    // Where layer = 0 is the layer closest to the input
    std::vector<ublas::matrix<double>> W;

    // The gradients of the loss for a given example w.r.t. the weights
    std::vector<ublas::matrix<double>> DW;

    // The pre-activations (since everything is linear in our network,
    // there are no post-activations)
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

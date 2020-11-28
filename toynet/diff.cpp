#include <diff.h>
#include <w2v.h>
#include <loss.h>
#include <stlio.h>
#include <ublasio.h>
#include <sstream>

namespace toynet {

void init_tensor(std::vector<ublas::matrix<double>>&V, int hidden, int width, int inputs)
{
    V.resize(hidden + 1);
    for (int i = 0;  i < hidden + 1;  ++i) {
        int rows = (i == 0 ? inputs : width);
        int cols = (i == hidden ? 1 : width);
        V[i] = ublas::zero_matrix<double>(rows, cols);
    }
}

void init_tensor(std::vector<ublas::vector<double>>& V, int hidden, int width, int inputs)
{
    V.resize(hidden + 2);  // 1 input layer, `hidden` hidden layers, 1 output layer
    // input layer
    V[0] = ublas::vector<double>(inputs, 0.0);
    for (int i = 1;  i <= hidden;  ++i)
        V[i] = ublas::vector<double>(width, 0.0);
    V[hidden+1] = ublas::vector<double>(1, 0.0);
}

void DiffNumbers::init_weights()
{
    const std::vector<double> pool{-0.2, -0.1, 0.0, 0.1, 0.2};
    int n = 0;
    for (int i = 0;  i < W.size();  ++i)
        for (int j = 0;  j < W[i].size1();  ++j)
            for (int k = 0;  k < W[i].size2();  ++k)
                W[i](j, k) = pool[n++ % pool.size()];
}

DiffNumbers::DiffNumbers(int hidden, int width, int inputs)
    : hidden(hidden)
    , width(width)
    , inputs(inputs)
    , loss(0.0)
{
    init_tensor(W, hidden, width, inputs);
    init_tensor(DW, hidden, width, inputs);
    init_tensor(A, hidden, width, inputs);
    init_tensor(G, hidden, width, inputs);
    init_weights();
}

void DiffNumbers::forward_backward(const std::vector<ublas::vector<double>>& training_set)
{
    MSELoss mseloss;

    init_tensor(DW, hidden, width, inputs);
    init_tensor(A, hidden, width, inputs);
    init_tensor(G, hidden, width, inputs);
    loss = 0.0;

    for (const auto & x : training_set) {
        std::vector<ublas::matrix<double>> exDW;
        std::vector<ublas::vector<double>> exA;
        std::vector<ublas::vector<double>> exG;

        init_tensor(exDW, hidden, width, inputs);
        init_tensor(exA, hidden, width, inputs);
        init_tensor(exG, hidden, width, inputs);

        // ground truth: y = f(x) = x[0] - x[1] + x[2] - x[3] ...
        double y = 0.0;
        for (int i = 0;  i < x.size();  ++i)
            y += (i % 2) ? -x(i) : x(i);
    
        // input layer
        exA[0] = x;
    
        // forward: hidden layers and output layer
        for (int i = 0;  i < hidden+1;  ++i)
            for (int j = 0;  j < W[i].size1();  ++j)
                for (int k = 0;  k < W[i].size2();  ++k)
                    exA[i+1](k) += exA[i](j) * W[i](j, k);
    
        // MSE loss and gradient of loss w.r.t. y_hat
        const double y_hat = exA[hidden+1](0);
        double exloss;
        double delta_loss_y_hat;
        std::tie(exloss, delta_loss_y_hat) = mseloss(y, y_hat);
        exG[hidden+1](0) = delta_loss_y_hat;
    
        // backward: for each layer
        for (int i = hidden;  i >= 0;  --i) {
            const auto& g = exG[i+1];
    
            // Convert the gradient into the pre-nonlinearity activation
            // This is a no-op because all activations are linear
    
            // Compute gradients on weights, biases, and regularization terms
            // Note: there are no bias nor regularization terms
            // d(J, W(i)) = g * h(i-1)
            for (int j = 0;  j < exA[i].size();  ++j)
                for (int k = 0;  k < g.size();  ++k)
                    exDW[i](j, k) = g(k) * exA[i](j);
    
            // Propagate the gradient w.r.t. the next lower-level hidden layer's
            // activation
            // g = d(J, h(i-1)) = W(i) * g
            for (int j = 0;  j < exA[i].size();  ++j)
                for (int k = 0;  k < g.size();  ++k)
                    exG[i](j) += W[i](j, k) * g(k);
        }

        // Add the matrices for example `x` to the overall matrices across all training examples
        add(DW, exDW);
        add(A, exA);
        add(G, exG);
        loss += exloss;
    }

    // Compute averages
    const int n = training_set.size();
    normalize(DW, n);
    normalize(A, n);
    normalize(G, n);
    loss /= n;
}

void DiffNumbers::update_weights(double lr)
{
    for (int i = 0;  i < W.size();  ++i)
        W[i] -= lr * DW[i];
}

std::string DiffNumbers::print() const
{
    std::stringstream oss;
    oss << *this;
    return oss.str();
}

std::ostream& operator<<(std::ostream& os, const DiffNumbers& obj)
{
    os << "hidden " << obj.hidden << " width " << obj.width << " inputs " << obj.inputs << " loss " << obj.loss << "\n"
       << "W : " << obj.W << "\n"
       << "DW: " << obj.DW << "\n"
       << "A : " << obj.A << "\n"
       << "G : " << obj.G << "\n"
       ;
    return os;
}

} // namespace toynet

#include <diff.h>
#include <w2v.h>
#include <stlio.h>
#include <sstream>

namespace w2v {

void init_tensor(std::vector<std::vector<std::vector<double>>>&V, int hidden, int width, int inputs)
{
    V.resize(hidden + 1);
    for (int i = 0;  i < hidden + 1;  ++i) {
        V[i].resize(i == 0 ? inputs : width);
        for (int j = 0;  j < V[i].size();  ++j) {
            V[i][j].resize(i == hidden ? 1 : width);
            for (int k = 0;  k < V[i][j].size();  ++k) {
                V[i][j][k] = 0.0;
            }
        }
    }
}

void init_tensor(std::vector<std::vector<double>>&V, int hidden, int width, int inputs)
{
    V.resize(hidden + 2);  // 1 input layer, `hidden` hidden layers, 1 output layer
    // input layer
    V[0] = std::vector<double>(inputs, 0.0);
    for (int i = 1;  i <= hidden;  ++i)
        V[i] = std::vector<double>(width, 0.0);
    V[hidden+1] = std::vector<double>(1, 0.0);
}

void DiffNumbers::init_weights()
{
    const std::vector<double> pool{-0.2, -0.1, 0.0, 0.1, 0.2};
    int n = 0;
    for (int i = 0;  i < W.size();  ++i) {
        for (int j = 0;  j < W[i].size();  ++j) {
            for (int k = 0;  k < W[i][j].size();  ++k) {
                W[i][j][k] = pool[n++ % pool.size()];
                // DW[i][j][k] = pool[n++ % pool.size()];
            }
        }
    }
    for (int i = 0;  i < G.size();  ++i) {
        for (int j = 0;  j < G[i].size();  ++j) {
            // A[i][j] = pool[n++ % pool.size()];
            // G[i][j] = pool[n++ % pool.size()];
        }
    }
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

void DiffNumbers::forward_backward(const std::vector<double>& x)
{
    // ground truth: y = f(x) = x[0] - x[1] + x[2] - x[3] ...
    double y = 0.0;
    for (int i = 0;  i < x.size();  ++i)
        y += (i % 2) ? -x[i] : x[i];

    // input layer
    A[0] = x;

    // forward: hidden layers and output layer
    for (int i = 0;  i < hidden+1;  ++i) {
        for (int j = 0;  j < A[i+1].size();  ++j)
            A[i+1][j] = 0.0;
        for (int j = 0;  j < W[i].size();  ++j) {
            for (int k = 0;  k < W[i][j].size();  ++k) {
                A[i+1][k] += A[i][j] * W[i][j][k];
            }
        }
    }

    // forward: MSE loss
    const double y_hat = A[hidden+1][0];
    const double diff = y - y_hat;
    loss = diff * diff;

    // backward: derivative of loss w.r.t. y_hat
    // L(y_hat, y) = (y - y_hat)^2
    //             = y^2 - 2*y*y_hat + y_hat^2
    // d(L, y_hat) = -2*y + 2*y_hat
    //             = -2 * diff
    const double delta_loss_y_hat = -2.0 * diff;
    G[hidden+1][0] = delta_loss_y_hat;

    // backward: for each layer
    for (int i = hidden;  i >= 0;  --i) {
        const std::vector<double>& g = G[i+1];

        // Convert the gradient into the pre-nonlinearity activation
        // This is a no-op because all activations are linear

        // Compute gradients on weights, biases, and regularization terms
        // Note: there are no bias nor regularization terms
        // d(J, W(i)) = g * h(i-1)
        for (int j = 0;  j < A[i].size();  ++j)
            for (int k = 0;  k < g.size();  ++k)
                DW[i][j][k] = g[k] * A[i][j];

        // Propagate the gradient w.r.t. the next lower-level hidden layer's
        // activation
        // g = d(J, h(i-1)) = W(i) * g
        for (int j = 0;  j < A[i].size();  ++j) {
            G[i][j] = 0.0;
            for (int k = 0;  k < g.size();  ++k)
                G[i][j] += W[i][j][k] * g[k];
        }
    }
}

void DiffNumbers::update_weights(double lr)
{
    for (int i = 0;  i < W.size();  ++i)
        for (int j = 0;  j < W[i].size();  ++j)
            for (int k = 0;  k < W[i][j].size();  ++k)
                W[i][j][k] -= lr * DW[i][j][k];
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

} // namespace w2v

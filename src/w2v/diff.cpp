#include <diff.h>
#include <w2v.h>
#include <stlio.h>

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
    V.resize(hidden + 1);
    for (int i = 0;  i < hidden + 1;  ++i) {
        V[i].resize(i == hidden ? 1 : width);
        for (int j = 0;  j < V[i].size();  ++j) {
            V[i][j] = 0.0;
        }
    }
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
    // forward: hidden layers
    for (int i = 0;  i < hidden;  ++i) {
        const std::vector<double>& prev = (i == 0) ? x : A[i-1];
        for (int j = 0;  j < A.size();  ++j)
            A[i][j] = 0.0;
        for (int j = 0;  j < W[i].size();  ++j) {
            for (int k = 0;  k < W[i][j].size();  ++k) {
                A[i][k] += prev[j] * W[i][j][k];
            }
        }
    }
    // forward: output (y_hat)
    double& y_hat = A.back().back();
    y_hat = 0.0;
    const std::vector<double>& prev = (hidden == 0) ? x : A[hidden-1];
    for (int i = 0;  i < W.back().size();  ++i)
        y_hat += W.back()[i].back() * prev[i];
    // forward: MSE loss
    double diff = y - y_hat;
    loss = diff * diff;
    // backward: G and DW 
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

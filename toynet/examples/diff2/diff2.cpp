#include <toynet/examples/diff2/diff2.h>
#include <toynet/math.h>
#include <toynet/stlio.h>
#include <toynet/ublas/io.h>
#include <sstream>

namespace toynet {
namespace diff2 {

std::vector<Tensor1D> Network::get1D() const
{
    std::vector<Tensor1D> V;
    V.resize(hidden + 2);  // 1 input layer, `hidden` hidden layers, 1 output layer
    V[0] = Tensor1D(inputs, 0.0);
    for (int i = 1;  i <= hidden;  ++i)
        V[i] = Tensor1D(width, 0.0);
    V[hidden+1] = Tensor1D(outputs, 0.0);
    return V;
}

std::vector<Tensor2D> Network::get2D() const
{
    std::vector<Tensor2D> V;
    V.resize(hidden + 1);
    for (int i = 0;  i < hidden + 1;  ++i) {
        int rows = (i == hidden ? outputs : width);
        int cols = (i == 0 ? inputs : width);
        V[i] = ublas::zero_matrix<double>(rows, cols);
    }
    return V;
}

void init_weights(std::vector<Tensor2D>& W)
{
    const std::vector<double> pool{-0.2, -0.1, 0.0, 0.1, 0.2};
    int n = 0;
    for (int i = 0;  i < W.size();  ++i)
        for (int k = 0;  k < W[i].size2();  ++k)
            for (int j = 0;  j < W[i].size1();  ++j)
                W[i](j, k) = pool[n++ % pool.size()];
}

Network::Network(int hidden, int width, int inputs, int outputs)
    : hidden(hidden)
    , width(width)
    , inputs(inputs)
    , outputs(outputs)
    , W(get2D())
{
    init_weights(W);
}

void Forward::forward(const Network& network, std::vector<Tensor1D>& A, const Tensor1D& x) const
{
    A[0] = x;  // input layer
    for (int i = 0;  i < network.hidden+1;  ++i)
        A[i+1] = ublas::prod(network.W[i], A[i]);  // hidden and output layers
}

Workspace::Workspace(const Network& network, bool needs_dA, bool needs_dW, bool needs_v)
    : A(network.get1D())
    , dA(needs_dA ? new std::vector<Tensor1D>(network.get1D()) : nullptr)
    , dW(needs_dW ? new std::vector<Tensor2D>(network.get2D()) : nullptr)
    , v(needs_v ? new std::vector<Tensor2D>(network.get2D()) : nullptr)
    , loss(0.0)
{
}

void Workspace::init_before_epoch()
{
    // clear dA and dW if they have been computed
    // no need to reset A -> activations will be overwritten in next forward pass
    // don't reset v! -> we need to keep it across epochs
    if (dA)
        for (auto& x : *dA)
            x = ublas::scalar_vector<double>(x.size(), 0.0);
    if (dW)
        for (auto& x : *dW)
            x = ublas::scalar_matrix<double>(x.size1(), x.size2(), 0.0);
}

void Workspace::add(const Workspace& ex)
{
    toynet::add(A, ex.A);
    if (dA) toynet::add(*dA, *ex.dA);
    if (dW) toynet::add(*dW, *ex.dW);
    loss += ex.loss;
}

void Workspace::average(int n)
{
    normalize(A, n);
    if (dA) normalize(*dA, n);
    if (dW) normalize(*dW, n);
    loss /= n;
}

void GradientOptimizer::compute_gradients(const Network& network, Workspace& workspace, Loss& loss, const Tensor1D &y) const
{
    std::tie(workspace.loss, (*workspace.dA)[network.hidden+1]) = loss(y, workspace.A[network.hidden+1]);
    for (int i = network.hidden;  i >= 0;  --i) {
        // (*workspace.dW)[i] = ublas::prod((*workspace.dA)[i+1], trans(workspace.A[i]));
        (*workspace.dW)[i] = ublas::outer_prod((*workspace.dA)[i+1], trans(workspace.A[i]));
        (*workspace.dA)[i] = ublas::prod(ublas::trans(network.W[i]), (*workspace.dA)[i+1]);
    }
}

void GradientOptimizer::update_weights(int epoch, Network& network, Workspace& workspace) const
{
    for (int i = 0;  i < network.W.size();  ++i)
        network.W[i] -= lr * (*workspace.dW)[i];
}

void MomentumOptimizer::update_weights(int epoch, Network& network, Workspace& workspace) const
{
    for (int i = 0;  i < network.W.size();  ++i) {
        (*workspace.v)[i] = alpha * (*workspace.v)[i]
                          - lr * (*workspace.dW)[i];
        network.W[i] += (*workspace.v)[i];
    }
}

Trainer::Trainer(Network& network, Loss& loss, Forward& forward, Optimizer& optimizer)
    : network(network)
    , loss(loss)
    , forward(forward)
    , optimizer(optimizer)
    , workspace(network, optimizer.computes_dA(), optimizer.computes_dW(), optimizer.computes_v())
{
}

void Trainer::train(int epoch, const std::vector<Tensor1D>& trainingX, const std::vector<Tensor1D>& trainingY)
{
    workspace.init_before_epoch();
    for (int i = 0;  i < trainingX.size();  ++i) {
        const auto& x = trainingX[i];
        const auto& y = trainingY[i];
        Workspace ex(network, optimizer.computes_dA(), optimizer.computes_dW(), false);
        forward.forward(network, ex.A, x);
        optimizer.compute_gradients(network, ex, loss, y);
        workspace.add(ex);
    }
    workspace.average(trainingX.size());
    optimizer.update_weights(epoch, network, workspace);
}

} // namespace diff2
} // namespace toynet

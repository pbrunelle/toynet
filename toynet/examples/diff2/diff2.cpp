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

void FixedWeightInitializer::initialize(std::vector<Tensor2D>& W) const
{
    const std::vector<double> pool{-0.2, -0.1, 0.0, 0.1, 0.2};
    int n = 0;
    for (auto& m : W)
        for (int k = 0;  k < m.size2();  ++k)
            for (int j = 0;  j < m.size1();  ++j)
                m(j, k) = pool[n++ % pool.size()];
}

GlorotBengio2010Initializer::GlorotBengio2010Initializer(double seed)
    : rng(seed)
{
}

void GlorotBengio2010Initializer::initialize(std::vector<Tensor2D>& W) const
{
    for (auto& m : W) {
        int denom = std::max<int>(1, m.size1() + m.size2());
        double v = std::sqrt(6.0 / denom);
        std::uniform_real_distribution<> dist(-v, v);
        for (int i = 0;  i < m.size1();  ++i)
            for (int j = 0;  j < m.size2();  ++j)
                m(i, j) = dist(rng);
    }
}

Network::Network(int hidden, int width, int inputs, int outputs,
                 const WeightInitializer *initializer)
    : hidden(hidden)
    , width(width)
    , inputs(inputs)
    , outputs(outputs)
    , W(get2D())
{
    if (initializer)
        initializer->initialize(W);
}

Tensor1D Network::predict(const Tensor1D& x) const
{
    Workspace workspace = build_workspace(*this);
    forward(workspace, x);
    return workspace.A.back();
}

void Network::forward(Workspace& workspace, const Tensor1D& x) const
{
    workspace.A[0] = x;  // input layer
    for (int i = 0;  i < hidden+1;  ++i)
        workspace.A[i+1] = ublas::prod(W[i], workspace.A[i]);  // hidden and output layers
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
    loss = 0.0;
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

void GradientOptimizer::compute_gradients(const Network& network, Workspace& workspace, const Loss& loss, const Tensor1D &y) const
{
    std::tie(workspace.loss, (*workspace.dA)[network.hidden+1]) = loss(y, workspace.A[network.hidden+1]);
    for (int i = network.hidden;  i >= 0;  --i) {
        (*workspace.dW)[i] = ublas::outer_prod((*workspace.dA)[i+1], ublas::trans(workspace.A[i]));
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

Workspace build_workspace(const Network& network)
{
    Workspace ret;
    ret.A = network.get1D();
    return ret;
}

Workspace build_workspace(const Network& network, const Optimizer& opt)
{
    Workspace ret;
    ret.A = network.get1D();
    if (opt.computes_dA()) ret.dA.reset(new std::vector<Tensor1D>(network.get1D()));
    if (opt.computes_dW()) ret.dW.reset(new std::vector<Tensor2D>(network.get2D()));
    if (opt.computes_v()) ret.v.reset(new std::vector<Tensor2D>(network.get2D()));
    return ret;
}

Trainer::Trainer(Network& network, const Loss& loss, const Optimizer& optimizer)
    : network(network)
    , loss(loss)
    , optimizer(optimizer)
    , workspace(build_workspace(network, optimizer))
{
}

void Trainer::train(int epoch, const std::vector<Tensor1D>& trainingX, const std::vector<Tensor1D>& trainingY)
{
    workspace.init_before_epoch();
    for (int i = 0;  i < trainingX.size();  ++i) {
        const auto& x = trainingX[i];
        const auto& y = trainingY[i];
        Workspace ex = build_workspace(network, optimizer);  // Bug: if optimizer is gradient, we needlessly create `v`
        network.forward(ex, x);
        optimizer.compute_gradients(network, ex, loss, y);
        workspace.add(ex);
    }
    workspace.average(trainingX.size());
    optimizer.update_weights(epoch, network, workspace);
}

} // namespace diff2
} // namespace toynet

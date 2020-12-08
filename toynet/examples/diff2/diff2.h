#include <toynet/ublas/ublas.h>
#include <toynet/loss.h>
#include <iostream>
#include <random>
#include <vector>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>

namespace toynet {
namespace diff2 {

typedef ublas::vector<double> Tensor1D;
typedef ublas::matrix<double> Tensor2D;

// Holds the required data structures for optimization
struct Workspace {
    Workspace() : loss(0.0) {}

    void init_before_epoch();

    void add(const Workspace& ex);

    void average(int n);

    // forward
    std::vector<Tensor1D> A;  // pre-non-linearity
    // backward
    std::unique_ptr<std::vector<Tensor1D>> dA;  // d(L, A)
    std::unique_ptr<std::vector<Tensor2D>> dW;  // d(L, network.W)
    std::unique_ptr<std::vector<Tensor2D>> v;  // for momentum
    double loss;
};

struct WeightInitializer {
    virtual void initialize(std::vector<Tensor2D>& W) const = 0;
};

struct FixedWeightInitializer : public WeightInitializer {
    virtual void initialize(std::vector<Tensor2D>& W) const override;
};

// W(i,j) ~ U(-sqrt(6/(m+n)), sqrt(6/(m+n)))
// Where: m = number of input units, n = number of output units
struct GlorotBengio2010Initializer : public WeightInitializer {
    GlorotBengio2010Initializer(double seed=0.0);
    virtual void initialize(std::vector<Tensor2D>& W) const override;
    mutable std::mt19937 rng;
};

struct Network {
    Network(int hidden, int width, int inputs, int outputs);

    Tensor1D predict(const Tensor1D& x) const;

    // Computes forward activations
    void forward(Workspace& workspace, const Tensor1D& x) const;

    std::vector<Tensor1D> get1D() const;

    std::vector<Tensor2D> get2D() const;

    int hidden;  // number of hidden layers
    int width;  // width of hidden layer
    int inputs;  // width of input layer
    int outputs;  // width of output layer

    // The weights of the FFN
    // NOTE - WARNING: we have changed how the weights are indexed compared to `Diff`:
    // W[layer][unit index of layer][unit index of previous layer]
    std::vector<Tensor2D> W;
};

struct Optimizer {
    virtual void compute_gradients(const Network& network, Workspace& workspace, const Loss& loss, const Tensor1D& y) const = 0;
    virtual void update_weights(int epoch, Network& network, Workspace& workspace) const = 0;

    virtual bool computes_dA() const {return false;}
    virtual bool computes_dW() const {return false;}
    virtual bool computes_v() const {return false;}
};

struct GradientOptimizer : public Optimizer {
    GradientOptimizer(double lr) : lr(lr) {}

    virtual void compute_gradients(const Network& network, Workspace& workspace, const Loss& loss, const Tensor1D& y) const override;
    virtual void update_weights(int epoch, Network& network, Workspace& workspace) const override;

    virtual bool computes_dA() const override {return true;}
    virtual bool computes_dW() const override {return true;}

    double lr;  // learning rate
};

struct MomentumOptimizer : public GradientOptimizer {
    MomentumOptimizer(double lr, double alpha) : GradientOptimizer(lr), alpha(alpha) {}

    virtual void update_weights(int epoch, Network& network, Workspace& workspace) const override;

    virtual bool computes_dA() const override {return true;}
    virtual bool computes_dW() const override {return true;}
    virtual bool computes_v() const override {return true;}

    double alpha;  // momentum
};

Workspace build_workspace(const Network& network);

Workspace build_workspace(const Network& network, const Optimizer& opt);

class Trainer {
public:
    Trainer(Network& network);

    // Chained setters
    Trainer& initializer(std::shared_ptr<WeightInitializer>);
    Trainer& loss(std::shared_ptr<Loss>);
    Trainer& optimizer(std::shared_ptr<Optimizer>);

    const Workspace& workspace() const {return d_workspace;}

    // Train for 1 epoch on a training set
    // Pre-conditions:
    // - trainingX.size() == trainingY.size()
    // - trainingX.size() >= 1
    void train(int epoch, const std::vector<Tensor1D>& trainingX, const std::vector<Tensor1D>& trainingY);

private:
    Network& d_network;
    std::shared_ptr<WeightInitializer> d_initializer;
    std::shared_ptr<Loss> d_loss;
    std::shared_ptr<Optimizer> d_optimizer;
    Workspace d_workspace;
};

} // namespace diff2
} // namespace toynet

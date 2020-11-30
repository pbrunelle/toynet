#include <toynet/ublas.h>
#include <boost/numeric/ublas/vector.hpp>

namespace toynet {

struct Loss {
    // Compute the loss and gradient for a single output
    virtual std::pair<double, double> operator()(
        const double& y, const double& y_hat) const;

    // Compute the loss and the vector of gradients for a vector of outputs
    // Pre-condition: y.size() == y_hat.size()
    // Pre-condition: y.size() >= 1
    virtual std::pair<double, ublas::vector<double>> operator()(
        const ublas::vector<double>& y, const ublas::vector<double>& y_hat) const;

    // The name of the loss function
    virtual std::string name() const = 0;
};

// MSE loss function: avg((y - y_hat)^2)
struct MSELoss : public Loss {
    virtual std::pair<double, double> operator()(
        const double& y, const double& y_hat) const override;
    virtual std::pair<double, ublas::vector<double>> operator()(
        const ublas::vector<double>& y, const ublas::vector<double>& y_hat) const override;
    virtual std::string name() const override {return "MSELoss";}
};

// Softmax loss function
struct SoftmaxLoss : public Loss {
    // Pre-condition: there exists one j such that y[j] == 1 and y[i] == 0 for all i != j
    virtual std::pair<double, ublas::vector<double>> operator()(
        const ublas::vector<double>& y, const ublas::vector<double>& y_hat) const override;
    virtual std::string name() const override {return "SoftmaxLoss";}
};

} // namespace toynet

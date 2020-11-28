#include <boost/numeric/ublas/vector.hpp>

// https://stackoverflow.com/a/1211402
namespace ublas = boost::numeric::ublas;

namespace toynet {

struct Loss {
    // Report the loss and the derivative of the loss w.r.t. y_hat
    virtual std::pair<double, double> operator()(
        const double& y, const double& y_hat) const = 0;
    virtual std::pair<double, ublas::vector<double>> operator()(
        const ublas::vector<double>& y, const ublas::vector<double>& y_hat) const = 0;
};

// MSE loss function: (y - y_hat)^2
struct MSELoss : public Loss {
    virtual std::pair<double, double> operator()(
        const double& y, const double& y_hat) const override;
    virtual std::pair<double, ublas::vector<double>> operator()(
        const ublas::vector<double>& y, const ublas::vector<double>& y_hat) const override;
};

} // namespace toynet

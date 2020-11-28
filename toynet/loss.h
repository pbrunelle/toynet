#include <boost/numeric/ublas/vector.hpp>

// https://stackoverflow.com/a/1211402
namespace ublas = boost::numeric::ublas;

namespace toynet {

struct Loss {
    virtual double operator()(const double& y, const double& y_hat) const = 0;
    virtual double operator()(const ublas::vector<double>& y, const ublas::vector<double>& y_hat) const = 0;
};

// MSE loss function: loss = (y - y_hat)^2
// Assumption: `y` and `y_hat` are scalars
struct MSELoss : public Loss {
    virtual double operator()(const double& y, const double& y_hat) const override;
    virtual double operator()(const ublas::vector<double>& y, const ublas::vector<double>& y_hat) const override;
};

} // namespace toynet

#include <loss.h>

namespace toynet {

double MSELoss::operator()(const ublas::vector<double>& y, const ublas::vector<double>& y_hat) const
{
    return (*this)(y(0), y_hat(0));
}

double MSELoss::operator()(const double& y, const double& y_hat) const
{
    const double diff = y - y_hat;
    const double loss = diff * diff;
    return loss;
}

} // namespace toynet

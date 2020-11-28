#include <loss.h>

namespace toynet {

std::pair<double, double> MSELoss::operator()(
    const double& y, const double& y_hat) const
{
    // The MSE loss function:
    // L(y, y_hat) = (y - y_hat)^2
    //             = y^2 - 2*y*y_hat + y_hat^2
    //
    // The derivative of the MSE loss function w.r.t. y_ht:
    // d(L, y_hat) = -2*y + 2*y_hat
    //             = -2 * (y - y_hat)
    const double diff = y - y_hat;
    const double loss = diff * diff;
    const double delta_loss_y_hat = -2.0 * diff;
    return std::make_pair(loss, delta_loss_y_hat);
}

std::pair<double, ublas::vector<double>> MSELoss::operator()(
    const ublas::vector<double>& y, const ublas::vector<double>& y_hat) const
{
    auto res = (*this)(y(0), y_hat(0));
    return std::make_pair(res.first, ublas::vector<double>(1, res.second));
}

} // namespace toynet

#include <toynet/loss.h>
#include <toynet/w2v.h> // softmax
#include <exception>
#include <math.h>
#include <sstream>

namespace toynet {

std::pair<double, double> Loss::operator()(
    const double& y, const double& y_hat) const
{
    std::ostringstream oss;
    oss << name() << "(double, double): cannot be called";
    throw std::runtime_error(oss.str());
}

std::pair<double, ublas::vector<double>> Loss::operator()(
    const ublas::vector<double>& y, const ublas::vector<double>& y_hat) const
{
    std::ostringstream oss;
    oss << name() << "(vector<double>: " << y.size() << ", vector<double>: "
        << y_hat.size() << "): cannot be called";
    throw std::runtime_error(oss.str());
}

std::pair<double, double> MSELoss::operator()(
    const double& y, const double& y_hat) const
{
    // The MSE loss function:
    // L(y, y_hat) = (y - y_hat)^2
    //             = y^2 - 2*y*y_hat + y_hat^2
    //
    // The derivative of the MSE loss function w.r.t. y_hat:
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
    double loss = 0.0;
    ublas::vector<double> gradients = ublas::zero_vector<double>(y.size());
    for (int i = 0;  i < y.size();  ++i) {
        auto res = (*this)(y[i], y_hat[i]);
        loss += res.first;
        gradients[i] = res.second;
    }
    loss /= y.size();
    return std::make_pair(loss, gradients);
}

std::pair<double, ublas::vector<double>> SoftmaxLoss::operator()(
    const ublas::vector<double>& y, const ublas::vector<double>& y_hat) const
{
    // See loss.md for explanations
    ublas::vector<double> soft = softmax(y_hat);
    // Find the index of the only element of `y` with non-zero value
    int i = 0;
    for ( ;  i < y.size();  ++i)
        if (y[i] > 0.5)
            break;
    // Compute the loss
    double loss = -std::log(soft[i]);
    // Compute the gradient of the loss w.r.t. y_hat
    ublas::vector<double> g = soft;
    g[i] -= 1.0;
    return std::make_pair(loss, g);
}

} // namespace toynet

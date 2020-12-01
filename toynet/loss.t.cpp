#include <toynet/loss.h>
#include <toynet/ublas/convert.h>
#include <toynet/ublas/test.h>
#include <boost/test/unit_test.hpp>

using namespace toynet;

// shorthand
ublas::vector<double> v1(double d)
{
    return ublas::vector<double>(1, d);
}

BOOST_AUTO_TEST_CASE(mse_loss)
{
    MSELoss loss;

    {
        auto res = loss(1.0, -0.14);
        BOOST_CHECK_CLOSE(1.2996, res.first, 1e-6);
        BOOST_CHECK_CLOSE(-2.28, res.second, 1e-6);
    }

    {
        auto res = loss(v1(1.0), v1(-0.14));
        BOOST_CHECK_CLOSE(1.2996, res.first, 1e-6);
        BOOST_REQUIRE_EQUAL(1, res.second.size());
        BOOST_CHECK_CLOSE(-2.28, res.second(0), 1e-6);
    }

    {
        auto res = loss(1.0, 0.282256);
        BOOST_CHECK_CLOSE(0.515156449536, res.first, 1e-6);
        BOOST_CHECK_CLOSE(-1.435488, res.second, 1e-6);
    }

    {
        auto res = loss(v1(1.0), v1(0.282256));
        BOOST_CHECK_CLOSE(0.515156449536, res.first, 1e-6);
        BOOST_REQUIRE_EQUAL(1, res.second.size());
        BOOST_CHECK_CLOSE(-1.435488, res.second(0), 1e-6);
    }

    {
        auto res = loss(convert({1.0, 1.0}), convert({-0.14, 0.282256}));
        BOOST_CHECK_CLOSE(0.907378224768, res.first, 1e-6);
        check_close_vectors(convert({-2.28, -1.435488}), res.second);
    }
}

BOOST_AUTO_TEST_CASE(softmax_loss_3)
{
    // https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
    SoftmaxLoss loss;
    const ublas::vector<double> y = convert({0.0, 1.0, 0.0});
    const ublas::vector<double> y_hat = convert({3.0, 1.0, 0.2});
    auto res = loss(y, y_hat);
    BOOST_CHECK_CLOSE(2.1791041747850026, res.first, 1e-6);
    check_close_vectors(convert({0.8360188, -1.0+0.11314284, 0.05083836}), res.second, 1e-5);
}

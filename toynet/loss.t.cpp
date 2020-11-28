#include <loss.h>
#include <stlio.h>
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
}

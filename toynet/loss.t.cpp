#include <loss.h>
#include <stlio.h>
#include <w2v.h> // convert
#include <boost/test/unit_test.hpp>

using namespace toynet;

// shorthand
ublas::vector<double> v1(double d)
{
    return ublas::vector<double>(1, d);
}

template<class T, class U>
void check_close_vectors(const T& expected, const U& got, double tol=0.000001)
{
    BOOST_REQUIRE_EQUAL(expected.size(), got.size());
    for (int i = 0;  i < expected.size();  ++i) {
        BOOST_CHECK_CLOSE(expected[i], got[i], tol);
    }
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

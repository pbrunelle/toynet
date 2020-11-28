#include <loss.h>
#include <boost/test/unit_test.hpp>

using namespace toynet;

BOOST_AUTO_TEST_CASE(loss)
{
    MSELoss loss;

    BOOST_CHECK_CLOSE(1.2996, loss(1.0, -0.14), 1e-4);
    BOOST_CHECK_CLOSE(0.515156, loss(1.0, 0.282256), 1e-4);

    BOOST_CHECK_CLOSE(1.2996, loss(ublas::vector<double>(1, 1.0), ublas::vector<double>(1, -0.14)), 1e-4);
    BOOST_CHECK_CLOSE(0.515156, loss(ublas::vector<double>(1, 1.0), ublas::vector<double>(1, 0.282256)), 1e-4);
}

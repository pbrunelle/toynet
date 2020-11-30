#include <toynet/ublasconvert.h>
#include <boost/test/unit_test.hpp>

using namespace toynet;

template<class T, class U>
void check_close_vectors(const ublas::vector<T>& expected, const ublas::vector<U>& got, double tol=0.000001)
{
    BOOST_REQUIRE_EQUAL(expected.size(), got.size());
    for (int i = 0;  i < expected.size();  ++i) {
        BOOST_CHECK_CLOSE(expected[i], got[i], tol);
    }
}

BOOST_AUTO_TEST_CASE(convert_vector_size_0)
{
    ublas::vector<double> expected;
    check_close_vectors(expected, convert({}));
}

BOOST_AUTO_TEST_CASE(convert_vector_size_1)
{
    ublas::vector<double> expected(1);
    expected[0] = 7;
    check_close_vectors(expected, convert({7}));
}

BOOST_AUTO_TEST_CASE(convert_vector_size_2)
{
    ublas::vector<double> expected(2);
    expected[0] = 4;
    expected[1] = 8;
    check_close_vectors(expected, convert({4, 8}));
}

#include <toynet/ublas/ublas.h>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/test/unit_test.hpp>

template<class T, class U>
void check_close_vectors(const ublas::vector<T>& expected, const ublas::vector<U>& got, double tol=0.000001)
{
    BOOST_REQUIRE_EQUAL(expected.size(), got.size());
    for (int i = 0;  i < expected.size();  ++i) {
        BOOST_CHECK_CLOSE(expected[i], got[i], tol);
    }
}

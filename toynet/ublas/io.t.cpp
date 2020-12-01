#include <toynet/ublas/io.h>
#include <toynet/ublas/ublas.h>
#include <sstream>
#include <boost/test/unit_test.hpp>

template<class T>
std::string print(const T& v)
{
    std::ostringstream oss;
    oss << v;
    return oss.str();
}

BOOST_AUTO_TEST_CASE(print_vector_size_0)
{
    const ublas::vector<double> v;
    BOOST_CHECK_EQUAL("[]", print(v));
}

BOOST_AUTO_TEST_CASE(print_vector_size_1)
{
    const ublas::vector<double> v(1, 1.5);
    BOOST_CHECK_EQUAL("[1.5]", print(v));
}

BOOST_AUTO_TEST_CASE(print_vector_size_2)
{
    ublas::vector<double> v(2);
    v[0] = -1;  v[1] = 3;
    BOOST_CHECK_EQUAL("[-1, 3]", print(v));
}

BOOST_AUTO_TEST_CASE(print_matrix_size_1_1)
{
    ublas::matrix<double> v(1, 1);
    v(0, 0) = 6;
    BOOST_CHECK_EQUAL("[[6]]", print(v));
}

BOOST_AUTO_TEST_CASE(print_matrix_size_1_2)
{
    ublas::matrix<double> v(1, 2);
    v(0, 0) = 6;
    v(0, 1) = 4;
    BOOST_CHECK_EQUAL("[[6, 4]]", print(v));
}

BOOST_AUTO_TEST_CASE(print_matrix_size_2_1)
{
    ublas::matrix<double> v(2, 1);
    v(0, 0) = 6;
    v(1, 0) = 4;
    BOOST_CHECK_EQUAL("[[6], [4]]", print(v));
}

BOOST_AUTO_TEST_CASE(print_matrix_size_2_2)
{
    ublas::matrix<double> v(2, 2);
    v(0, 0) = 6;
    v(0, 1) = 3;
    v(1, 0) = 4;
    v(1, 1) = 8;
    BOOST_CHECK_EQUAL("[[6, 3], [4, 8]]", print(v));
}

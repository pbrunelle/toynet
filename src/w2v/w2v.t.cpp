#include <w2v.h>
#include <stlio.h>
#define BOOST_TEST_MODULE w2v
#include <boost/test/included/unit_test.hpp>

using namespace w2v;

template<class T, class U>
void check_close_vectors(const std::vector<T>& expected, const std::vector<U>& got, double tol=0.000001)
{
    BOOST_REQUIRE_EQUAL(expected.size(), got.size());
    for (int i = 0;  i < expected.size();  ++i) {
        BOOST_CHECK_CLOSE(expected[i], got[i], tol);
    }
}

BOOST_AUTO_TEST_CASE(softmax_numerical_stability)
{
    // https://ogunlao.github.io/2020/04/26/you_dont_really_know_softmax.html#numerical-stability-of-softmax
    std::vector<double> v{10, 2, 10000, 4};
    std::vector<double> expected{0, 0, 1, 0};
    std::vector<double> got = softmax(v);
    check_close_vectors(expected, got);
}

BOOST_AUTO_TEST_CASE(naive_softmax_size_0)
{
    std::vector<double> v{};
    std::vector<double> expected{};
    std::vector<double> got = naive_softmax(v);
    BOOST_CHECK_EQUAL(expected, got);
}

BOOST_AUTO_TEST_CASE(naive_softmax_size_1)
{
    std::vector<double> v{5.0};
    std::vector<double> expected{1.0};
    std::vector<double> got = naive_softmax(v);
    BOOST_CHECK_EQUAL(expected, got);
}

BOOST_AUTO_TEST_CASE(naive_softmax_size_1_sum_0)
{
    // Make sure we don't divide by 0!
    std::vector<double> v{0.0};
    std::vector<double> expected{1.0};
    std::vector<double> got = naive_softmax(v);
    BOOST_CHECK_EQUAL(expected, got);
}

BOOST_AUTO_TEST_CASE(naive_softmax_size_2)
{
    std::vector<double> v{1.0, 2.0};
    std::vector<double> expected{0.26894142, 0.73105858};  // numpy 1.15.4
    std::vector<double> got = naive_softmax(v);
    check_close_vectors(expected, got);
}

BOOST_AUTO_TEST_CASE(stable_softmax_size_0)
{
    std::vector<double> v{};
    std::vector<double> expected{};
    std::vector<double> got = stable_softmax(v);
    BOOST_CHECK_EQUAL(expected, got);
}

BOOST_AUTO_TEST_CASE(stable_softmax_size_1)
{
    std::vector<double> v{5.0};
    std::vector<double> expected{1.0};
    std::vector<double> got = stable_softmax(v);
    BOOST_CHECK_EQUAL(expected, got);
}

BOOST_AUTO_TEST_CASE(stable_softmax_size_1_sum_0)
{
    // Make sure we don't divide by 0!
    std::vector<double> v{0.0};
    std::vector<double> expected{1.0};
    std::vector<double> got = stable_softmax(v);
    BOOST_CHECK_EQUAL(expected, got);
}

BOOST_AUTO_TEST_CASE(stable_softmax_size_2)
{
    std::vector<double> v{1.0, 2.0};
    std::vector<double> expected{0.26894142, 0.73105858};  // numpy 1.15.4
    std::vector<double> got = stable_softmax(v);
    check_close_vectors(expected, got);
}

BOOST_AUTO_TEST_CASE(stable_softmax_numerical_stability)
{
    // https://ogunlao.github.io/2020/04/26/you_dont_really_know_softmax.html#numerical-stability-of-softmax
    std::vector<double> v{10, 2, 10000, 4};
    std::vector<double> expected{0, 0, 1, 0};
    std::vector<double> got = stable_softmax(v);
    check_close_vectors(expected, got);
}

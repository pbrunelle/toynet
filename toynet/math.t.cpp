#include <toynet/math.h>
#include <toynet/stlio.h>
#include <toynet/ublas/convert.h>
#include <toynet/ublas/io.h>
#include <toynet/ublas/test.h>
#include <boost/test/unit_test.hpp>

using namespace toynet;

void help_test_softmax_size_0(ublas::vector<double> (*fn)(const ublas::vector<double>&))
{
    const ublas::vector<double> v;
    const ublas::vector<double> expected;
    const ublas::vector<double> got = (*fn)(v);
    check_close_vectors(expected, got);
}

void help_test_softmax_size_1(ublas::vector<double> (*fn)(const ublas::vector<double>&))
{
    const ublas::vector<double> v = convert({5.0});
    const ublas::vector<double> expected = convert({1});
    const ublas::vector<double> got = (*fn)(v);
    check_close_vectors(expected, got);
}

void help_test_softmax_size_1_sum_0(ublas::vector<double> (*fn)(const ublas::vector<double>&))
{
    // Make sure we don't divide by 0!
    const ublas::vector<double> v = convert({0.0});
    const ublas::vector<double> expected = convert({1.0});
    const ublas::vector<double> got = (*fn)(v);
    check_close_vectors(expected, got);
}

void help_test_softmax_size_2(ublas::vector<double> (*fn)(const ublas::vector<double>&))
{
    const ublas::vector<double> v = convert({1.0, 2.0});
    const ublas::vector<double> expected = convert({0.26894142, 0.73105858});  // numpy 1.15.4
    const ublas::vector<double> got = (*fn)(v);
    check_close_vectors(expected, got);
}

void help_test_softmax_numerical_stability(ublas::vector<double> (*fn)(const ublas::vector<double>&))
{
    // https://ogunlao.github.io/2020/04/26/you_dont_really_know_softmax.html#numerical-stability-of-softmax
    const ublas::vector<double> v = convert({10, 2, 10000, 4});
    const ublas::vector<double> expected = convert({0, 0, 1, 0});
    const ublas::vector<double> got = (*fn)(v);
    check_close_vectors(expected, got);
}

BOOST_AUTO_TEST_CASE(test_softmax)
{
    help_test_softmax_size_0(softmax);
    help_test_softmax_size_1(softmax);
    help_test_softmax_size_1_sum_0(softmax);
    help_test_softmax_size_2(softmax);
    help_test_softmax_numerical_stability(softmax);
}

BOOST_AUTO_TEST_CASE(test_naive_softmax)
{
    help_test_softmax_size_0(naive_softmax);
    help_test_softmax_size_1(naive_softmax);
    help_test_softmax_size_1_sum_0(naive_softmax);
    help_test_softmax_size_2(naive_softmax);
}

BOOST_AUTO_TEST_CASE(test_stable_softmax)
{
    help_test_softmax_size_0(stable_softmax);
    help_test_softmax_size_1(stable_softmax);
    help_test_softmax_size_1_sum_0(stable_softmax);
    help_test_softmax_size_2(stable_softmax);
    help_test_softmax_numerical_stability(stable_softmax);
}

void help_test_magnitude_size_0(double (*fn)(const ublas::vector<double>&))
{
    const ublas::vector<double> v;
    const double expected = 0.0;
    const double got = (*fn)(v);
    BOOST_CHECK_CLOSE(expected, got, 0.000001);
}

void help_test_magnitude_size_1(double (*fn)(const ublas::vector<double>&))
{
    const ublas::vector<double> v = convert({6});
    const double expected = 6;
    const double got = (*fn)(v);
    BOOST_CHECK_CLOSE(expected, got, 0.000001);
}

void help_test_magnitude_size_2(double (*fn)(const ublas::vector<double>&))
{
    const ublas::vector<double> v = convert({4, 3});
    const double expected = 5;
    const double got = (*fn)(v);
    BOOST_CHECK_CLOSE(expected, got, 0.000001);
}

BOOST_AUTO_TEST_CASE(test_magnitude)
{
    help_test_magnitude_size_0(magnitude);
    help_test_magnitude_size_1(magnitude);
    help_test_magnitude_size_2(magnitude);
}

BOOST_AUTO_TEST_CASE(test_naive_magnitude)
{
    help_test_magnitude_size_0(naive_magnitude);
    help_test_magnitude_size_1(naive_magnitude);
    help_test_magnitude_size_2(naive_magnitude);
}

BOOST_AUTO_TEST_CASE(test_ublas_magnitude)
{
    help_test_magnitude_size_0(ublas_magnitude);
    help_test_magnitude_size_1(ublas_magnitude);
    help_test_magnitude_size_2(ublas_magnitude);
}

void help_test_dot_product_size_0(double (*fn)(const ublas::vector<double>& v1, const ublas::vector<double>& v2))
{
    const ublas::vector<double> v1;
    const ublas::vector<double> v2;
    const double expected = 0.0;
    const double got = (*fn)(v1, v2);
    BOOST_CHECK_EQUAL(expected, got);
}

void help_test_dot_product_size_1(double (*fn)(const ublas::vector<double>& v1, const ublas::vector<double>& v2))
{
    const ublas::vector<double> v1 = convert({1.5});
    const ublas::vector<double> v2 = convert({-2.0});
    const double expected = -3.0;
    const double got = (*fn)(v1, v2);
    BOOST_CHECK_EQUAL(expected, got);
}

void help_test_dot_product_size_2(double (*fn)(const ublas::vector<double>& v1, const ublas::vector<double>& v2))
{
    const ublas::vector<double> v1 = convert({1.5, 10});
    const ublas::vector<double> v2 = convert({-2.0, 0.25});
    const double expected = -0.5;
    const double got = (*fn)(v1, v2);
    BOOST_CHECK_EQUAL(expected, got);
}

BOOST_AUTO_TEST_CASE(test_dot_product)
{
    help_test_dot_product_size_0(dot_product);
    help_test_dot_product_size_1(dot_product);
    help_test_dot_product_size_2(dot_product);
}

BOOST_AUTO_TEST_CASE(test_naive_dot_product)
{
    help_test_dot_product_size_0(naive_dot_product);
    help_test_dot_product_size_1(naive_dot_product);
    help_test_dot_product_size_2(naive_dot_product);
}

BOOST_AUTO_TEST_CASE(test_ublas_dot_product)
{
    help_test_dot_product_size_0(ublas_dot_product);
    help_test_dot_product_size_1(ublas_dot_product);
    help_test_dot_product_size_2(ublas_dot_product);
}

BOOST_AUTO_TEST_CASE(cosine_distance_a)
{
    // https://stackoverflow.com/a/1750187
    const ublas::vector<double> v1 = convert({2, 0, 1, 1, 0, 2, 1, 1});
    const ublas::vector<double> v2 = convert({2, 1, 1, 0, 1, 1, 1, 1});
    const double expected = 0.822;
    const double got12 = cosine_distance(v1, v2);
    const double got21 = cosine_distance(v2, v1);
    const double got11 = cosine_distance(v1, v1);
    const double got22 = cosine_distance(v2, v2);
    BOOST_CHECK_CLOSE(expected, got12, 0.1);
    BOOST_CHECK_EQUAL(got12, got21);
    BOOST_CHECK_CLOSE(1.0, got11, 1e-12);
    BOOST_CHECK_CLOSE(1.0, got22, 1e-12);
}

BOOST_AUTO_TEST_CASE(cosine_distance_b)
{
    // https://stackoverflow.com/a/14038820
    const ublas::vector<double> v1 = convert({-1, -1, 0});
    const ublas::vector<double> v2 = convert({-1, 0, -1});
    const double expected = 0.5;
    const double got12 = cosine_distance(v1, v2);
    const double got21 = cosine_distance(v2, v1);
    const double got11 = cosine_distance(v1, v1);
    const double got22 = cosine_distance(v2, v2);
    BOOST_CHECK_CLOSE(expected, got12, 0.1);
    BOOST_CHECK_EQUAL(got12, got21);
    BOOST_CHECK_CLOSE(1.0, got11, 1e-12);
    BOOST_CHECK_CLOSE(1.0, got22, 1e-12);
}

BOOST_AUTO_TEST_CASE(nearest_neighbors_size_0)
{
    const ublas::vector<double> v = convert({-1, -1, 0});
    const std::vector<ublas::vector<double>> points{};
    const std::vector<int> expected{};
    const std::vector<int> got = nearest_neighbors(v, points);
    BOOST_CHECK_EQUAL(expected, got);
}

BOOST_AUTO_TEST_CASE(nearest_neighbors_size_1_equal)
{
    const ublas::vector<double> v = convert({-1, -1, 0});
    const std::vector<ublas::vector<double>> points{convert({-1, -1, 0})};
    const std::vector<int> expected{0};
    const std::vector<int> got = nearest_neighbors(v, points);
    BOOST_CHECK_EQUAL(expected, got);
}

BOOST_AUTO_TEST_CASE(nearest_neighbors_size_1_not_equal)
{
    const ublas::vector<double> v = convert({-1, -1, 0});
    const std::vector<ublas::vector<double>> points{convert({-1, 0, -1})};
    const std::vector<int> expected{0};
    const std::vector<int> got = nearest_neighbors(v, points);
    BOOST_CHECK_EQUAL(expected, got);
}

BOOST_AUTO_TEST_CASE(nearest_neighbors_size_2)
{
    const ublas::vector<double> v = convert({-1, -1, 0});
    const std::vector<ublas::vector<double>> points{
        convert({-1, 0, -1}),
        convert({-1, -1, 0})
    };
    const std::vector<int> expected{1, 0};
    const std::vector<int> got = nearest_neighbors(v, points);
    BOOST_CHECK_EQUAL(expected, got);
}

BOOST_AUTO_TEST_CASE(nearest_neighbors_size_4)
{
    const ublas::vector<double> v = convert({-1, -1, 0});
    const std::vector<ublas::vector<double>> points{
        convert({-2, -1, 0}),
        convert({-1, 0, -1}),
        convert({-1, -1, 0}),
        convert({-3, -1, 0})
    };
    const std::vector<int> expected{2, 0, 3, 1};
    const std::vector<int> got = nearest_neighbors(v, points);
    BOOST_CHECK_EQUAL(expected, got);
}

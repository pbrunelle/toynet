#include <w2v.h>
#include <stlio.h>
#include <iostream>
#define BOOST_TEST_MODULE w2v
#include <boost/test/unit_test.hpp>

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
    const std::vector<double> v{10, 2, 10000, 4};
    const std::vector<double> expected{0, 0, 1, 0};
    const std::vector<double> got = softmax(v);
    check_close_vectors(expected, got);
}

BOOST_AUTO_TEST_CASE(naive_softmax_size_0)
{
    const std::vector<double> v{};
    const std::vector<double> expected{};
    const std::vector<double> got = naive_softmax(v);
    BOOST_CHECK_EQUAL(expected, got);
}

BOOST_AUTO_TEST_CASE(naive_softmax_size_1)
{
    const std::vector<double> v{5.0};
    const std::vector<double> expected{1.0};
    const std::vector<double> got = naive_softmax(v);
    BOOST_CHECK_EQUAL(expected, got);
}

BOOST_AUTO_TEST_CASE(naive_softmax_size_1_sum_0)
{
    // Make sure we don't divide by 0!
    const std::vector<double> v{0.0};
    const std::vector<double> expected{1.0};
    const std::vector<double> got = naive_softmax(v);
    BOOST_CHECK_EQUAL(expected, got);
}

BOOST_AUTO_TEST_CASE(naive_softmax_size_2)
{
    const std::vector<double> v{1.0, 2.0};
    const std::vector<double> expected{0.26894142, 0.73105858};  // numpy 1.15.4
    const std::vector<double> got = naive_softmax(v);
    check_close_vectors(expected, got);
}

BOOST_AUTO_TEST_CASE(stable_softmax_size_0)
{
    const std::vector<double> v{};
    const std::vector<double> expected{};
    const std::vector<double> got = stable_softmax(v);
    BOOST_CHECK_EQUAL(expected, got);
}

BOOST_AUTO_TEST_CASE(stable_softmax_size_1)
{
    const std::vector<double> v{5.0};
    const std::vector<double> expected{1.0};
    const std::vector<double> got = stable_softmax(v);
    BOOST_CHECK_EQUAL(expected, got);
}

BOOST_AUTO_TEST_CASE(stable_softmax_size_1_sum_0)
{
    // Make sure we don't divide by 0!
    const std::vector<double> v{0.0};
    const std::vector<double> expected{1.0};
    const std::vector<double> got = stable_softmax(v);
    BOOST_CHECK_EQUAL(expected, got);
}

BOOST_AUTO_TEST_CASE(stable_softmax_size_2)
{
    const std::vector<double> v{1.0, 2.0};
    const std::vector<double> expected{0.26894142, 0.73105858};  // numpy 1.15.4
    const std::vector<double> got = stable_softmax(v);
    check_close_vectors(expected, got);
}

BOOST_AUTO_TEST_CASE(stable_softmax_numerical_stability)
{
    // https://ogunlao.github.io/2020/04/26/you_dont_really_know_softmax.html#numerical-stability-of-softmax
    const std::vector<double> v{10, 2, 10000, 4};
    const std::vector<double> expected{0, 0, 1, 0};
    const std::vector<double> got = stable_softmax(v);
    check_close_vectors(expected, got);
}

BOOST_AUTO_TEST_CASE(magnitude_size_0)
{
    const std::vector<double> v{};
    const double expected = 0.0;
    const double got = magnitude(v);
    BOOST_CHECK_CLOSE(expected, got, 0.000001);
}

BOOST_AUTO_TEST_CASE(magnitude_size_1)
{
    const std::vector<double> v{6};
    const double expected = 6;
    const double got = magnitude(v);
    BOOST_CHECK_CLOSE(expected, got, 0.000001);
}

BOOST_AUTO_TEST_CASE(magnitude_size_2)
{
    const std::vector<double> v{4, 3};
    const double expected = 5;
    const double got = magnitude(v);
    BOOST_CHECK_CLOSE(expected, got, 0.000001);
}

BOOST_AUTO_TEST_CASE(dot_product_size_0)
{
    const std::vector<double> v1{};
    const std::vector<double> v2{};
    const double expected = 0.0;
    const double got = dot_product(v1, v2);
    BOOST_CHECK_EQUAL(expected, got);
}

BOOST_AUTO_TEST_CASE(dot_product_size_1)
{
    const std::vector<double> v1{1.5};
    const std::vector<double> v2{-2.0};
    const double expected = -3.0;
    const double got = dot_product(v1, v2);
    BOOST_CHECK_EQUAL(expected, got);
}

BOOST_AUTO_TEST_CASE(dot_product_size_2)
{
    const std::vector<double> v1{1.5, 10};
    const std::vector<double> v2{-2.0, 0.25};
    const double expected = -0.5;
    const double got = dot_product(v1, v2);
    BOOST_CHECK_EQUAL(expected, got);
}

BOOST_AUTO_TEST_CASE(cosine_distance_a)
{
    // https://stackoverflow.com/a/1750187
    const std::vector<double> v1{2, 0, 1, 1, 0, 2, 1, 1};
    const std::vector<double> v2{2, 1, 1, 0, 1, 1, 1, 1};
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
    const std::vector<double> v1{-1, -1, 0};
    const std::vector<double> v2{-1, 0, -1};
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
    const std::vector<double> v{-1, -1, 0};
    const std::vector<std::vector<double>> points{};
    const std::vector<int> expected{};
    const std::vector<int> got = nearest_neighbors(v, points);
    BOOST_CHECK_EQUAL(expected, got);
}

BOOST_AUTO_TEST_CASE(nearest_neighbors_size_1_equal)
{
    const std::vector<double> v{-1, -1, 0};
    const std::vector<std::vector<double>> points{{-1, -1, 0}};
    const std::vector<int> expected{0};
    const std::vector<int> got = nearest_neighbors(v, points);
    BOOST_CHECK_EQUAL(expected, got);
}

BOOST_AUTO_TEST_CASE(nearest_neighbors_size_1_not_equal)
{
    const std::vector<double> v{-1, -1, 0};
    const std::vector<std::vector<double>> points{{-1, 0, -1}};
    const std::vector<int> expected{0};
    const std::vector<int> got = nearest_neighbors(v, points);
    BOOST_CHECK_EQUAL(expected, got);
}

BOOST_AUTO_TEST_CASE(nearest_neighbors_size_2)
{
    const std::vector<double> v{-1, -1, 0};
    const std::vector<std::vector<double>> points{
        {-1, 0, -1},
        {-1, -1, 0}
    };
    const std::vector<int> expected{1, 0};
    const std::vector<int> got = nearest_neighbors(v, points);
    BOOST_CHECK_EQUAL(expected, got);
}

BOOST_AUTO_TEST_CASE(nearest_neighbors_size_4)
{
    const std::vector<double> v{-1, -1, 0};
    const std::vector<std::vector<double>> points{
        {-2, -1, 0},
        {-1, 0, -1},
        {-1, -1, 0},
        {-3, -1, 0}
    };
    const std::vector<int> expected{2, 0, 3, 1};
    const std::vector<int> got = nearest_neighbors(v, points);
    BOOST_CHECK_EQUAL(expected, got);
}

BOOST_AUTO_TEST_CASE(CBOWModel_constructor_W1_defaults)
{
    CBOWModel model(1);
    BOOST_CHECK_EQUAL(1, model.W);
    BOOST_CHECK_EQUAL(50, model.D);
    BOOST_CHECK_EQUAL(4, model.historyN);
    BOOST_CHECK_EQUAL(4, model.futureN);
    BOOST_CHECK_EQUAL(1, model.P.size());
    BOOST_CHECK_EQUAL(50, model.P[0].size());
}

BOOST_AUTO_TEST_CASE(CBOWModel_constructor_W2_defaults)
{
    CBOWModel model(2);
    BOOST_CHECK_EQUAL(2, model.W);
    BOOST_CHECK_EQUAL(50, model.D);
    BOOST_CHECK_EQUAL(4, model.historyN);
    BOOST_CHECK_EQUAL(4, model.futureN);
    BOOST_CHECK_EQUAL(2, model.P.size());
    BOOST_CHECK_EQUAL(50, model.P[0].size());
    BOOST_CHECK_EQUAL(50, model.P[1].size());
}

BOOST_AUTO_TEST_CASE(CBOWModel_constructor_no_defaults)
{
    CBOWModel model(3, 4, 5, 6);
    BOOST_CHECK_EQUAL(3, model.W);
    BOOST_CHECK_EQUAL(4, model.D);
    BOOST_CHECK_EQUAL(5, model.historyN);
    BOOST_CHECK_EQUAL(6, model.futureN);
    BOOST_CHECK_EQUAL(3, model.P.size());
    BOOST_CHECK_EQUAL(4, model.P[0].size());
}

BOOST_AUTO_TEST_CASE(CBOWModel_constructor_W_1000000_D_100)
{
    // Each matrix takes 1e6 * 1e2 * 8 bytes = 8e9 bytes = 800 MiB
    CBOWModel model(1000000, 100);
    BOOST_CHECK_EQUAL(1000000, model.W);
    BOOST_CHECK_EQUAL(100, model.D);
    BOOST_CHECK_EQUAL(4, model.historyN);
    BOOST_CHECK_EQUAL(4, model.futureN);
    BOOST_CHECK_EQUAL(1000000, model.P.size());
    BOOST_CHECK_EQUAL(100, model.P[0].size());
    BOOST_CHECK_EQUAL(100, model.P[1000000-1].size());
    BOOST_CHECK_EQUAL(100, model.O[0].size());
    BOOST_CHECK_EQUAL(100, model.O[1000000-1].size());
}

BOOST_AUTO_TEST_CASE(CBOWModel_save_load)
{
    CBOWModel model(3, 4, 5, 6);
    std::stringstream ss;
    model.save(ss);
    CBOWModel model2(1);
    model2.load(ss);
    BOOST_CHECK_EQUAL(3, model2.W);
    BOOST_CHECK_EQUAL(4, model2.D);
    BOOST_CHECK_EQUAL(5, model2.historyN);
    BOOST_CHECK_EQUAL(6, model2.futureN);
    BOOST_CHECK_EQUAL(3, model2.P.size());
    BOOST_CHECK_EQUAL(4, model2.P[0].size());
    BOOST_CHECK_EQUAL(3, model2.O.size());
    BOOST_CHECK_EQUAL(4, model2.O[0].size());
}


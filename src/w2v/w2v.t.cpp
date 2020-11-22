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

BOOST_AUTO_TEST_CASE(add_size_0)
{
    std::vector<double> to{};
    const std::vector<double> other{};
    const std::vector<double> expected{};
    add(to, other);
    BOOST_CHECK_EQUAL(expected, to);
}

BOOST_AUTO_TEST_CASE(add_size_1)
{
    std::vector<double> to{1.5};
    const std::vector<double> other{-2.5};
    const std::vector<double> expected{-1.0};
    add(to, other);
    BOOST_CHECK_EQUAL(expected, to);
}

BOOST_AUTO_TEST_CASE(add_size_2)
{
    std::vector<double> to{1.5, 0.25};
    const std::vector<double> other{-2.5, 0.75};
    const std::vector<double> expected{-1.0, 1.0};
    add(to, other);
    BOOST_CHECK_EQUAL(expected, to);
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

BOOST_AUTO_TEST_CASE(CBOWModel_predict_context_1)
{
    CBOWModel model(4, 3, 2, 2);
    model.P = {
        {0.19,  0.11, -0.14},
        {0.04,  0.30, -0.20},
        {0.77, -0.03,  0.11},
        {0.33, -0.43,  0.81},
    };
    model.O = {
        {0.42, -0.28,  0.19},
        {0.77, -0.93,  0.11},
        {0.33, -0.43, -0.81},
        {0.94,  0.90, -0.20},
    };
    std::vector<std::pair<double, int>> prediction = model.predict({1});
    BOOST_CHECK_EQUAL(4, prediction.size());
    double sum = 0;
    for (const auto& pred : prediction) {
        sum += pred.first;
        BOOST_CHECK(pred.first >= 0.0);
        BOOST_CHECK(pred.first <= 1.0);
        BOOST_CHECK(pred.second >= 0);
        BOOST_CHECK(pred.second < 4);
    }
    BOOST_CHECK_CLOSE(sum, 1.0, 1e-12);
    // output values (before softmax):
    // [0]: (0.04, 0.30, -0.20) dot (0.42, -0.28, 0.19) = -0.1052
    // [1]: (0.04, 0.30, -0.20) dot (0.77, -0.93, 0.11) = -0.2702
    // [2]: (0.04, 0.30, -0.20) dot (0.33, -0.43, -0.81) = 0.0462
    // [3]: (0.04, 0.30, -0.20) dot (0.94, 0.90, -0.20) = 0.3476
    // after softmax:
    // (0.21814698, 0.18496545, 0.25380571, 0.34308185)
    BOOST_CHECK_EQUAL(prediction[0].second, 3);
    BOOST_CHECK_CLOSE(prediction[0].first, 0.34308185, 1e-4);
    BOOST_CHECK_EQUAL(prediction[1].second, 2);
    BOOST_CHECK_CLOSE(prediction[1].first, 0.25380571, 1e-4);
    BOOST_CHECK_EQUAL(prediction[2].second, 0);
    BOOST_CHECK_CLOSE(prediction[2].first, 0.21814698, 1e-4);
    BOOST_CHECK_EQUAL(prediction[3].second, 1);
    BOOST_CHECK_CLOSE(prediction[3].first, 0.18496545, 1e-4);
}


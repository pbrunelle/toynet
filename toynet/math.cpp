#include <toynet/math.h>
#include <algorithm>
#include <cmath>
#include <functional>
#include <boost/numeric/ublas/matrix_proxy.hpp>

namespace toynet {

ublas::vector<double> softmax(const ublas::vector<double>& v)
{
    return stable_softmax(v);
}

ublas::vector<double> naive_softmax(const ublas::vector<double>& v)
{
    double sum = 0;
    for (auto x : v)
        sum += exp(x);
    ublas::vector<double> ret(v.size());
    for (int i = 0;  i < v.size();  ++i)
        ret[i] = exp(v[i]) / sum;
    return ret;
}

ublas::vector<double> stable_softmax(const ublas::vector<double> &v)
{
    double max = v.empty() ? 0.0 : *std::max_element(v.begin(), v.end());
    double sum = 0;
    for (auto x : v)
        sum += exp(x - max);
    ublas::vector<double> ret(v.size());
    for (int i = 0;  i < v.size();  ++i)
        ret[i] = exp(v[i] - max) / sum;
    return ret;
}

double magnitude(const ublas::vector<double>& v)
{
    return ublas_magnitude(v);
}

double naive_magnitude(const ublas::vector<double>& v)
{
    double ret = 0.0;
    for (auto x : v)
        ret += x * x;
    return sqrt(ret);
}

double ublas_magnitude(const ublas::vector<double>& v)
{
    return ublas::norm_2(v);
}

double dot_product(const ublas::vector<double>& v1, const ublas::vector<double>& v2)
{
    return ublas_dot_product(v1, v2);
}

double naive_dot_product(const ublas::vector<double>& v1, const ublas::vector<double>& v2)
{
    double ret = 0.0;
    for (int i = 0;  i < v1.size();  ++i)
        ret += v1[i] * v2[i];
    return ret;
}

double ublas_dot_product(const ublas::vector<double>& v1, const ublas::vector<double>& v2)
{
    return ublas::inner_prod(v1, v2);
}

double cosine_distance(const ublas::vector<double>& v1, const ublas::vector<double>& v2)
{
    return dot_product(v1, v2) / (magnitude(v1) * magnitude(v2));
}

std::vector<int> nearest_neighbors(const ublas::vector<double>& v, const std::vector<ublas::vector<double>>& points)
{
    std::vector<std::pair<double, int>> distances(points.size());
    for (int i = 0;  i < points.size();  ++i)
        distances[i] = std::make_pair(cosine_distance(v, points[i]), i);
    std::sort(distances.begin(), distances.end(), std::greater<>());
    std::vector<int> ret(points.size());
    for (int i = 0;  i < points.size();  ++i)
        ret[i] = distances[i].second;
    return ret;
}

void add(std::vector<ublas::vector<double>>& to, const std::vector<ublas::vector<double>>& other)
{
    for (int i = 0;  i < to.size();  ++i)
        to[i] += other[i];
}

void add(std::vector<ublas::matrix<double>>& to, const std::vector<ublas::matrix<double>>& other)
{
    for (int i = 0;  i < to.size();  ++i)
        to[i] += other[i];
}

void normalize(std::vector<ublas::vector<double>>& to, double denom)
{
    for (auto& d : to)
        d /= denom;
}

void normalize(std::vector<ublas::matrix<double>>& to, double denom)
{
    for (auto& d : to)
        d /= denom;
}

} // namespace toynet

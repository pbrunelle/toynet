#include <w2v.h>
#include <algorithm>
#include <cmath>
#include <functional>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

namespace w2v {

std::vector<double> softmax(const std::vector<double>& v)
{
    return stable_softmax(v);
}

std::vector<double> naive_softmax(const std::vector<double>& v)
{
    double sum = 0;
    for (auto x : v)
        sum += exp(x);
    std::vector<double> ret(v.size());
    for (int i = 0;  i < v.size();  ++i)
        ret[i] = exp(v[i]) / sum;
    return ret;
}

std::vector<double> stable_softmax(const std::vector<double> &v)
{
    double max = v.empty() ? 0.0 : *std::max_element(v.begin(), v.end());
    double sum = 0;
    for (auto x : v)
        sum += exp(x - max);
    std::vector<double> ret(v.size());
    for (int i = 0;  i < v.size();  ++i)
        ret[i] = exp(v[i] - max) / sum;
    return ret;
}

double magnitude(const std::vector<double>& v)
{
    double ret = 0.0;
    for (auto x : v)
        ret += x * x;
    return sqrt(ret);
}

double dot_product(const std::vector<double>& v1, const std::vector<double>& v2)
{
    double ret = 0.0;
    for (int i = 0;  i < v1.size();  ++i)
        ret += v1[i] * v2[i];
    return ret;
}

double cosine_distance(const std::vector<double>& v1, const std::vector<double>& v2)
{
    return dot_product(v1, v2) / (magnitude(v1) * magnitude(v2));
}

std::vector<int> nearest_neighbors(const std::vector<double>& v, const std::vector<std::vector<double>>& points)
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

CBOWModel::CBOWModel(int W, int D, int historyN, int futureN)
    : W(W)
    , D(D)
    , historyN(historyN)
    , futureN(futureN)
    , P(W, std::vector<double>(D))
{
}

void CBOWModel::save(std::ostream& os) const
{
    boost::archive::text_oarchive oa(os);
    oa << *this;
}

void CBOWModel::load(std::istream& is)
{
    boost::archive::text_iarchive ia(is);
    ia >> *this;
}

} // namespace w2v

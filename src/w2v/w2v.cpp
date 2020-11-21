#include <w2v.h>
#include <algorithm>
#include <cmath>

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

} // namespace w2v

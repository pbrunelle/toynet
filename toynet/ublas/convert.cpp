#include <toynet/ublas/convert.h>

namespace toynet {

ublas::vector<double> convert(const std::vector<double>& v)
{
    ublas::vector<double> ret(v.size());
    for (int i = 0;  i < v.size();  ++i)
        ret(i) = v[i];
    return ret;
}

} // namespace toynet

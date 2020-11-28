#include <vector>
#include <ostream>
#include <boost/numeric/ublas/matrix.hpp>

namespace std {

template<class T>
ostream& operator<<(ostream& os, const std::vector<T>& o)
{
    os << "[";
    for (int i = 0;  i < o.size();  ++i) {
        if (i > 0)
            os << ", ";
        os << o[i];
    }
    os << "]";
    return os;
}

template<class T, class U>
ostream& operator<<(ostream& os, const pair<T, U>& o)
{
    os << "(" << o.first << ", " << o.second << ")";
    return os;
}

} // namespace std

namespace boost {
namespace numeric {
namespace ublas {

template<class T>
std::ostream& operator<<(std::ostream& os, const matrix<T>& obj)
{
    std::vector<std::vector<T>> m(obj.size1());
    for (int i = 0;  i < obj.size1();  ++i)
        m[i].resize(obj.size2());
    for (int i = 0;  i < obj.size1();  ++i)
        for (int j = 0;  j < obj.size2();  ++j)
            m[i][j] = obj(i, j);
    return os << m;
}

} // namespace ublas
} // namespace numeric
} // namespace boost

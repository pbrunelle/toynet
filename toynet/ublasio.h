#include <ostream>
#include <boost/numeric/ublas/matrix.hpp>

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

#include <ostream>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/vector.hpp>

namespace boost {
namespace numeric {
namespace ublas {

template<class T>
void _help_print_ublas_1d(std::ostream& os, const T& o)
{
    os << "[";
    for (int i = 0;  i < o.size();  ++i) {
        if (i > 0)
            os << ", ";
        os << o[i];
    }
    os << "]";
}

template<class T>
std::ostream& operator<<(std::ostream& os, const vector<T>& o)
{
    _help_print_ublas_1d(os, o);
    return os;
}

template<class T>
std::ostream& operator<<(std::ostream& os, const matrix<T>& o)
{
    os << "[";
    for (int i = 0;  i < o.size1();  ++i) {
        if (i > 0)
            os << ", ";
        _help_print_ublas_1d(os, row(o, i));
    }
    os << "]";
    return os;
}

} // namespace ublas
} // namespace numeric
} // namespace boost

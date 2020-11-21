#include <vector>
#include <ostream>

namespace std {

template<class T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& o)
{
    os << "[";
    for (int i = 0;  i < o.size();  ++i) {
        if (i > 0)
            os << ", ";
        os << o[i];
    }
    os << "]";
}

} // namespace std

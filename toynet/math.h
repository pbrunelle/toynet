#include <toynet/ublas/ublas.h>
#include <vector>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>

namespace toynet {

// softmax
ublas::vector<double> softmax(const ublas::vector<double>& v);

ublas::vector<double> naive_softmax(const ublas::vector<double>& v);

ublas::vector<double> stable_softmax(const ublas::vector<double>& v);

// magnitude
double magnitude(const ublas::vector<double>& v);

double naive_magnitude(const ublas::vector<double>& v);

double ublas_magnitude(const ublas::vector<double>& v);

// dot product
// pre-condition: v1.size() == v2.size()
double dot_product(const ublas::vector<double>& v1, const ublas::vector<double>& v2);

double naive_dot_product(const ublas::vector<double>& v1, const ublas::vector<double>& v2);

double ublas_dot_product(const ublas::vector<double>& v1, const ublas::vector<double>& v2);

// pre-condition: v1.size() == v2.size()
// pre-condition: ||v1|| != 0 && ||v2|| != 0
double cosine_distance(const ublas::vector<double>& v1, const ublas::vector<double>& v2);

// pre-condition: ||v|| != 0
// pre-condition: for each p in points: p.size() == v.size() && ||p|| != 0
std::vector<int> nearest_neighbors(const ublas::vector<double>& v, const std::vector<ublas::vector<double>>& points);

// element-wise addition
// pre-condition: to.size() == other.size()
void add(std::vector<ublas::vector<double>>& to, const std::vector<ublas::vector<double>>& other);

// element-wise addition
void add(std::vector<ublas::matrix<double>>& to, const std::vector<ublas::matrix<double>>& other);

// divide each element by a constant
// pre-condition: denom != 0.0
void normalize(std::vector<ublas::vector<double>>& to, double denom);

// divide each element by a constant
void normalize(std::vector<ublas::matrix<double>>& to, double denom);

} // namespace toynet

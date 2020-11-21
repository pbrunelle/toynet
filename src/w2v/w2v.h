#include <vector>

namespace w2v {

std::vector<double> softmax(const std::vector<double>& v);

std::vector<double> naive_softmax(const std::vector<double>& v);

std::vector<double> stable_softmax(const std::vector<double>& v);

double magnitude(const std::vector<double>& v);

// pre-condition: v1.size() == v2.size()
double dot_product(const std::vector<double>& v1, const std::vector<double>& v2);

// pre-condition: v1.size() == v2.size()
// pre-condition: ||v1|| != 0 && ||v2|| != 0
double cosine_distance(const std::vector<double>& v1, const std::vector<double>& v2);

// pre-condition: ||v|| != 0
// pre-condition: for each p in points: p.size() == v.size() && ||p|| != 0
std::vector<int> nearest_neighbors(const std::vector<double>& v, const std::vector<std::vector<double>>& points);

} // namespace w2v

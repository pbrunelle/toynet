#include <vector>

namespace w2v {

std::vector<double> softmax(const std::vector<double>& v);

std::vector<double> naive_softmax(const std::vector<double>& v);

std::vector<double> stable_softmax(const std::vector<double>& v);

} // namespace w2v

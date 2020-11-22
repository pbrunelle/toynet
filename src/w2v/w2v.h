#include <iostream>
#include <vector>
#include <boost/serialization/access.hpp>
#include <boost/serialization/vector.hpp>

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

// An implementation of the Continuous Bag-of-Words Model from
// https://arxiv.org/abs/1301.3781.
struct CBOWModel {
    // Parameters:
    // - W: the number of words in the vocabulary
    // - D: the embedding size
    // - historyN: the number of words *before* the current (middle) word
    // - futureN: the number of words *after* the current (middle) word
    // Pre-conditions:
    // - W > 0
    // - D > 0
    // - historyN + futureN > 0
    CBOWModel(int W, int D=50, int historyN=4, int futureN=4);

    // Save the model to an output stream using Boost serialization
    void save(std::ostream& os) const;

    // Load a model from an input stream using Boost serialization
    void load(std::istream& is);

    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive& ar, const unsigned int version)
    {
        ar & W;
        ar & D;
        ar & historyN;
        ar & futureN;
        ar & P;
        ar & O;
    }

    int W;
    int D;
    int historyN;
    int futureN;
    // The matrix between the input layer and the projection layer
    // P[word_index][projection_layer_index]
    // (i.e. the words embeddings)
    std::vector<std::vector<double>> P;
    // The matrix between the projection layer and the output layer
    // P[word_index][projection_layer_index]
    // (i.e. used to predict most likely words)
    std::vector<std::vector<double>> O;
};

} // namespace w2v

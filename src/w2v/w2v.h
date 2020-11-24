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

// element-wise addition
// pre-condition: to.size() == other.size()
void add(std::vector<double>& to, const std::vector<double>& other);

// Get the context around word `index` from a corpus `words`, using a maximum
// of `historyN` words before `index` and `futureN` words after `index`.
// The word at `index` is *not* part of the context.
// The number of context words is less than `historyN + futureN` for words
// close to the beginning or end of the corpus.
std::vector<int> get_context(const std::vector<int>& words, int index, int historyN, int futureN);

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

    // Return the list of word indices with their associated probability
    // given a list of context word indices.
    // Parameters:
    //   - context: the list of context word indices, i.e. if there are 2 context
    //      words, this list should contain 2 elements, the indices of each context
    //      word.  A word index can appear more than once.  The order is not important.
    //      This model computes the *average* of each word embedding before using
    //      those to compute output probabilities.
    // Return value: A list of dimension W, where each element is a pair:
    //      first = the probability of the corresponding word index
    //      second = the word index
    //      The list is sorted descending according to the probability
    // Pre-consitions:
    //   -  For each `i` in `context`: 0 <= `i` < W
    // Post-conditions:
    //   -  The returned probabilities sum to 1
    //   -  The returned list is of size W
    std::vector<std::pair<double, int>> predict(const std::vector<int>& context) const;

    // Like the previous `predict` function, but only returns the probability
    // for word index `word`: p(word | context)
    // Pre-conditions:
    //   - 0 <= `word` < W
    double predict(const std::vector<int>& context, int word) const;

    // Helper function for `predict` methods above
    std::vector<double> predict_helper(const std::vector<int>& context) const;

    // Given a corpus `words`, compute the average log probability, as defined
    // in https://arxiv.org/abs/1310.4546:
    //     1/T * sum{t=1, T}(log(p(w_t | context)))
    // where T is the size of `words`.
    // Parameters:
    //   - words: the corpus, a list of word indices
    // Pre-conditions:
    //   - For each `w` in `words`: 0 <= `w` < W
    double avg_log_prob(const std::vector<int>& words) const;

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

struct ReportData {
    int epoch;
    double avg_log_prob;
};

struct SimpleReporter {
    SimpleReporter(std::ostream& os);
    void operator()(const ReportData& data) const;
};

struct Trainer {
    Trainer();

    Trainer& setEpochs(int epochs);

    Trainer& setEmbeddingSize(int D);

    Trainer& setHistoryN(int historyN);

    Trainer& setFutureN(int futureN);

    // Pre-conditions: corpus.size() > 0
    CBOWModel train(const std::vector<int>& corpus) const;

    int epochs;
    int D;
    int historyN;
    int futureN;
};

} // namespace w2v

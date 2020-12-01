#include <toynet/ublas/ublas.h>
#include <iostream>
#include <vector>
#include <boost/serialization/access.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>

namespace toynet {

// Get the context around word `index` from a corpus `words`, using a maximum
// of `historyN` words before `index` and `futureN` words after `index`.
// The word at `index` is *not* part of the context.
// The number of context words is less than `historyN + futureN` for words
// close to the beginning or end of the corpus.
std::vector<int> get_context(const std::vector<int>& words, int index, int historyN, int futureN);

// Given two matrices `out` and `gradients` of the same dimensions, modify out such that:
// out[i][j] -= gradients[i][j] * lr
void gradient_descent(ublas::matrix<double>& out, const ublas::matrix<double>& gradients, double lr);

struct CBOWModelGradients {
    CBOWModelGradients(int W, int D);
    ublas::matrix<double> P;
    ublas::matrix<double> O;
};

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
    ublas::vector<double> predict_helper(const std::vector<int>& context) const;

    // Given a corpus `words`, compute the average log probability, as defined
    // in https://arxiv.org/abs/1310.4546:
    //     1/T * sum{t=1, T}(log(p(w_t | context)))
    // where T is the size of `words`.
    // Parameters:
    //   - words: the corpus, a list of word indices
    // Pre-conditions:
    //   - For each `w` in `words`: 0 <= `w` < W
    double avg_log_prob(const std::vector<int>& words) const;

    // Compute gradients of loss function w.r.t. P and O
    CBOWModelGradients gradients() const;

    // Given gradients and a learning rate, update the matrices P and O
    void update(const CBOWModelGradients& gradients, double lr);

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
    ublas::matrix<double> P;
    // The matrix between the projection layer and the output layer
    // P[word_index][projection_layer_index]
    // (i.e. used to predict most likely words)
    ublas::matrix<double> O;
};

struct ReportData {
    int epoch;
    double avg_log_prob;
    double lr;
};

struct Reporter {
    virtual void operator()(const ReportData& data) const = 0;
};

struct SimpleReporter : public Reporter {
    SimpleReporter(std::ostream& os);
    virtual void operator()(const ReportData& data) const override;
    std::ostream& os;
};

struct LearningRate {
    virtual double operator()(int epoch) const = 0;
};

// For the first `cutoff` epochs, learning rate = lr
// Afterwards, learning rate = lr * base ^ (cutoff - epoch)
struct SimpleLearningRate : public LearningRate {
    SimpleLearningRate(double lr=1.0, int cutoff=3, double base=0.5);
    virtual double operator()(int epoch) const override;
    double lr;
    int cutoff;
    double base;
};

struct Trainer {
    Trainer();

    Trainer& setEpochs(int epochs);

    Trainer& setEmbeddingSize(int D);

    Trainer& setHistoryN(int historyN);

    Trainer& setFutureN(int futureN);

    Trainer& setInitReporter(const Reporter *initReporter);

    Trainer& setEpochReporter(const Reporter *epochReporter);

    Trainer& setExitReporter(const Reporter *exitReporter);

    Trainer& setLearningRate(const LearningRate *learningRate);

    // Pre-conditions: corpus.size() > 0
    CBOWModel train(const std::vector<int>& corpus) const;

    int epochs;
    int D;
    int historyN;
    int futureN;
    const Reporter *initReporter;
    const Reporter *epochReporter;
    const Reporter *exitReporter;
    const LearningRate *learningRate;
};

} // namespace toynet

#include <toynet/w2v.h>
#include <toynet/math.h>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>

namespace toynet {

std::vector<int> get_context(const std::vector<int>& words, int index, int historyN, int futureN)
{
    std::vector<int> ret;
    for (int j = std::max(0, index - historyN);  j < index;  ++j)
        ret.push_back(words[j]);
    for (int j = 1;  j <= futureN && index+j < words.size();  ++j)
        ret.push_back(words[index+j]);
    return ret;
}

void gradient_descent(ublas::matrix<double>& out, const ublas::matrix<double>& gradients, double lr)
{
    out -= gradients * lr;
}

CBOWModelGradients::CBOWModelGradients(int W, int D)
    : P(ublas::zero_matrix<double>(W, D))
    , O(ublas::zero_matrix<double>(W, D))
{
}

CBOWModel::CBOWModel(int W, int D, int historyN, int futureN)
    : W(W)
    , D(D)
    , historyN(historyN)
    , futureN(futureN)
    , P(W, D)
    , O(W, D)
{
}

void CBOWModel::save(std::ostream& os) const
{
    boost::archive::text_oarchive oa(os);
    oa << *this;
}

void CBOWModel::load(std::istream& is)
{
    boost::archive::text_iarchive ia(is);
    ia >> *this;
}

std::vector<std::pair<double, int>> CBOWModel::predict(const std::vector<int>& context) const
{
    ublas::vector<double> smax = predict_helper(context);
    std::vector<std::pair<double, int>> ret(W);
    for (int i = 0;  i < W;  ++i)
        ret[i] = std::make_pair(smax[i], i);
    std::sort(ret.begin(), ret.end(), std::greater<>());
    return ret;
}

double CBOWModel::predict(const std::vector<int>& context, int word) const
{
    ublas::vector<double> smax = predict_helper(context);
    return smax[word];
}

ublas::vector<double> CBOWModel::predict_helper(const std::vector<int>& context) const
{
    // average embedding of all context words
    ublas::vector<double> avg(D, 0.0);
    for (int wordidx : context)
        avg += ublas::row(P, wordidx);
    avg /= context.size();
    // output layer (before softmax)
    ublas::vector<double> out(W, 0.0);
    for (int i = 0;  i < W;  ++i)
        out[i] = dot_product(avg, row(O, i));
    // softmax
    ublas::vector<double> smax = softmax(out);
    return smax;
}

double CBOWModel::avg_log_prob(const std::vector<int>& words) const
{
    double sum = 0.0;
    for (int i = 0;  i < words.size();  ++i) {
        std::vector<int> context = get_context(words, i, historyN, futureN);
        double p = predict(context, words[i]);
        sum += std::log(p);
    }
    return sum / words.size();
}

CBOWModelGradients CBOWModel::gradients() const
{
    CBOWModelGradients ret(W, D);
    // TODO!
    return ret;
}

void CBOWModel::update(const CBOWModelGradients& gradients, double lr)
{
    gradient_descent(P, gradients.P, lr);
    gradient_descent(O, gradients.O, lr);
}

SimpleReporter::SimpleReporter(std::ostream& os)
    : os(os)
{
}

SimpleLearningRate::SimpleLearningRate(double lr, int cutoff, double base)
    : lr(lr)
    , cutoff(cutoff)
    , base(base)
{
}

void SimpleReporter::operator()(const ReportData& data) const
{
    os << "epoch " << data.epoch
       << " prob " << data.avg_log_prob
       << " lr " << data.lr
       << "\n";
}

double SimpleLearningRate::operator()(int epoch) const
{
    return (epoch <= cutoff)
         ? lr
         : lr * std::pow(base, epoch - cutoff);
}

Trainer::Trainer()
    : epochs(1)
    , D(50)
    , historyN(4)
    , futureN(4)
    , initReporter(nullptr)
    , epochReporter(nullptr)
    , exitReporter(nullptr)
    , learningRate(nullptr)
{
}

Trainer& Trainer::setEpochs(int epochs)
{
    this->epochs = epochs;
    return *this;
}

Trainer& Trainer::setEmbeddingSize(int D)
{
    this->D = D;
    return *this;
}

Trainer& Trainer::setHistoryN(int historyN)
{
    this->historyN = historyN;
    return *this;
}

Trainer& Trainer::setFutureN(int futureN)
{
    this->futureN = futureN;
    return *this;
}

Trainer& Trainer::setInitReporter(const Reporter *initReporter)
{
    this->initReporter = initReporter;
    return *this;
}

Trainer& Trainer::setEpochReporter(const Reporter *epochReporter)
{
    this->epochReporter = epochReporter;
    return *this;
}

Trainer& Trainer::setExitReporter(const Reporter *exitReporter)
{
    this->exitReporter = exitReporter;
    return *this;
}

Trainer& Trainer::setLearningRate(const LearningRate *learningRate)
{
    this->learningRate = learningRate;
    return *this;
}

CBOWModel Trainer::train(const std::vector<int>& corpus) const
{
    // Find W, the maximum number of words
    int W = *std::max_element(corpus.begin(), corpus.end()) + 1;
    CBOWModel model(W, D, historyN, futureN);
    int e = 0;
    double avg_log_prob = model.avg_log_prob(corpus);
    double lr = (*learningRate)(e);
    if (initReporter)
        (*initReporter)({e, avg_log_prob, lr});
    while (e <= epochs) {
        ++e;
        CBOWModelGradients gradients = model.gradients();
        lr = learningRate ? (*learningRate)(e) : 1.0;
        model.update(gradients, lr);
        avg_log_prob = model.avg_log_prob(corpus);
        if (epochReporter)
            (*epochReporter)({e, avg_log_prob, lr});
        // TODO: compute loss on validation data
    }
    if (exitReporter)
        (*exitReporter)({-1, avg_log_prob, lr});
    return model;
}

} // namespace toynet

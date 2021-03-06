#include <toynet/w2v.h>
#define BOOST_TEST_MODULE toynet
#include <toynet/stlio.h>
#include <toynet/ublas/convert.h>
#include <toynet/ublas/io.h>
#include <toynet/ublas/test.h>
#include <iostream>
#include <boost/test/unit_test.hpp>

using namespace toynet;

BOOST_AUTO_TEST_CASE(get_context_6)
{
    const std::vector<int> words{78, 333, 0, 99, 15, 4, 2, 2, 78, 99};
    const std::vector<int> expected1{333, 0, 99, 15, 4, 2, 2, 78, 99};
    const std::vector<int> expected2{333, 0};
    const std::vector<int> expected3{78, 333, 99, 15, 4};
    const std::vector<int> expected4{4, 2};
    const std::vector<int> expected5{4, 2, 2, 99};
    const std::vector<int> expected6{2, 2, 78};
    BOOST_CHECK_EQUAL(expected1, get_context(words, 0, 10, 10));
    BOOST_CHECK_EQUAL(expected2, get_context(words, 0, 10, 2));
    BOOST_CHECK_EQUAL(expected3, get_context(words, 2, 4, 3));
    BOOST_CHECK_EQUAL(expected4, get_context(words, 6, 1, 1));
    BOOST_CHECK_EQUAL(expected5, get_context(words, 8, 3, 3));
    BOOST_CHECK_EQUAL(expected6, get_context(words, 9, 3, 3));
}

BOOST_AUTO_TEST_CASE(gradient_descent_2_by_2)
{
    ublas::matrix<double> m(2, 2);
    m(0, 0) = 2.0;
    m(0, 1) = 7.5;
    m(1, 0) = -3.0;
    m(1, 1) = -19.5;
    ublas::matrix<double> g(2, 2);
    g(0, 0) = 3.0;
    g(0, 1) = -5.0;
    g(1, 0) = 9.0;
    g(1, 1) = -2.5;
    gradient_descent(m, g, 0.5);
    BOOST_CHECK_EQUAL(0.5, m(0, 0)); // 2.0 - 3.0 * 0.5
    BOOST_CHECK_EQUAL(10.0, m(0, 1)); // 7.5 - (-5.0 * 0.5)
    BOOST_CHECK_EQUAL(-7.5, m(1, 0)); // -3.0 - (9.0 * 0.5)
    BOOST_CHECK_EQUAL(-18.25, m(1, 1)); // -19.5 - (-2.5 * 0.5)
}

BOOST_AUTO_TEST_CASE(CBOWModel_constructor_W1_defaults)
{
    CBOWModel model(1);
    BOOST_CHECK_EQUAL(1, model.W);
    BOOST_CHECK_EQUAL(50, model.D);
    BOOST_CHECK_EQUAL(4, model.historyN);
    BOOST_CHECK_EQUAL(4, model.futureN);
    BOOST_CHECK_EQUAL(1, model.P.size1());
    BOOST_CHECK_EQUAL(50, model.P.size2());
}

BOOST_AUTO_TEST_CASE(CBOWModel_constructor_W2_defaults)
{
    CBOWModel model(2);
    BOOST_CHECK_EQUAL(2, model.W);
    BOOST_CHECK_EQUAL(50, model.D);
    BOOST_CHECK_EQUAL(4, model.historyN);
    BOOST_CHECK_EQUAL(4, model.futureN);
    BOOST_CHECK_EQUAL(2, model.P.size1());
    BOOST_CHECK_EQUAL(50, model.P.size2());
}

BOOST_AUTO_TEST_CASE(CBOWModel_constructor_no_defaults)
{
    CBOWModel model(3, 4, 5, 6);
    BOOST_CHECK_EQUAL(3, model.W);
    BOOST_CHECK_EQUAL(4, model.D);
    BOOST_CHECK_EQUAL(5, model.historyN);
    BOOST_CHECK_EQUAL(6, model.futureN);
    BOOST_CHECK_EQUAL(3, model.P.size1());
    BOOST_CHECK_EQUAL(4, model.P.size2());
}

BOOST_AUTO_TEST_CASE(CBOWModel_constructor_W_1000000_D_100)
{
    // Each matrix takes 1e6 * 1e2 * 8 bytes = 8e9 bytes = 800 MiB
    CBOWModel model(1000000, 100);
    BOOST_CHECK_EQUAL(1000000, model.W);
    BOOST_CHECK_EQUAL(100, model.D);
    BOOST_CHECK_EQUAL(4, model.historyN);
    BOOST_CHECK_EQUAL(4, model.futureN);
    BOOST_CHECK_EQUAL(1000000, model.P.size1());
    BOOST_CHECK_EQUAL(100, model.P.size2());
    BOOST_CHECK_EQUAL(1000000, model.O.size1());
    BOOST_CHECK_EQUAL(100, model.O.size2());
}

BOOST_AUTO_TEST_CASE(CBOWModel_save_load)
{
    CBOWModel model(3, 4, 5, 6);
    std::stringstream ss;
    model.save(ss);
    CBOWModel model2(1);
    model2.load(ss);
    BOOST_CHECK_EQUAL(3, model2.W);
    BOOST_CHECK_EQUAL(4, model2.D);
    BOOST_CHECK_EQUAL(5, model2.historyN);
    BOOST_CHECK_EQUAL(6, model2.futureN);
    BOOST_CHECK_EQUAL(3, model2.P.size1());
    BOOST_CHECK_EQUAL(4, model2.P.size2());
    BOOST_CHECK_EQUAL(3, model2.O.size1());
    BOOST_CHECK_EQUAL(4, model2.O.size2());
}

CBOWModel get_model()
{
    CBOWModel model(4, 3, 2, 2);

    // {0.19,  0.11, -0.14}
    model.P(0, 0) = 0.19;
    model.P(0, 1) = 0.11;
    model.P(0, 2) = -0.14;
    // {0.04,  0.30, -0.20}
    model.P(1, 0) = 0.04;
    model.P(1, 1) = 0.30;
    model.P(1, 2) = -0.20;
    // {0.77, -0.03,  0.11}
    model.P(2, 0) = 0.77;
    model.P(2, 1) = -0.03;
    model.P(2, 2) = 0.11;
    // {0.33, -0.43,  0.81}
    model.P(3, 0) = 0.33;
    model.P(3, 1) = -0.43;
    model.P(3, 2) = 0.81;

    // {0.42, -0.28,  0.19}
    model.O(0, 0) = 0.42;
    model.O(0, 1) = -0.28;
    model.O(0, 2) = 0.19;
    // {0.77, -0.93,  0.11}
    model.O(1, 0) = 0.77;
    model.O(1, 1) = -0.93;
    model.O(1, 2) = 0.11;
    // {0.33, -0.43, -0.81}
    model.O(2, 0) = 0.33;
    model.O(2, 1) = -0.43;
    model.O(2, 2) = -0.81;
    // {0.94,  0.90, -0.20}
    model.O(3, 0) = 0.94;
    model.O(3, 1) = 0.90;
    model.O(3, 2) = -0.20;

    return model;
}

BOOST_AUTO_TEST_CASE(CBOWModel_predict_context_1)
{
    CBOWModel model = get_model();
    std::vector<std::pair<double, int>> prediction = model.predict({1});
    BOOST_CHECK_EQUAL(4, prediction.size());
    double sum = 0;
    for (const auto& pred : prediction) {
        sum += pred.first;
        BOOST_CHECK(pred.first >= 0.0);
        BOOST_CHECK(pred.first <= 1.0);
        BOOST_CHECK(pred.second >= 0);
        BOOST_CHECK(pred.second < 4);
    }
    BOOST_CHECK_CLOSE(sum, 1.0, 1e-12);
    // output values (before softmax):
    // [0]: (0.04, 0.30, -0.20) dot (0.42, -0.28, 0.19) = -0.1052
    // [1]: (0.04, 0.30, -0.20) dot (0.77, -0.93, 0.11) = -0.2702
    // [2]: (0.04, 0.30, -0.20) dot (0.33, -0.43, -0.81) = 0.0462
    // [3]: (0.04, 0.30, -0.20) dot (0.94, 0.90, -0.20) = 0.3476
    // after softmax:
    // (0.21814698, 0.18496545, 0.25380571, 0.34308185)
    BOOST_CHECK_EQUAL(prediction[0].second, 3);
    BOOST_CHECK_CLOSE(prediction[0].first, 0.34308185, 1e-4);
    BOOST_CHECK_EQUAL(prediction[1].second, 2);
    BOOST_CHECK_CLOSE(prediction[1].first, 0.25380571, 1e-4);
    BOOST_CHECK_EQUAL(prediction[2].second, 0);
    BOOST_CHECK_CLOSE(prediction[2].first, 0.21814698, 1e-4);
    BOOST_CHECK_EQUAL(prediction[3].second, 1);
    BOOST_CHECK_CLOSE(prediction[3].first, 0.18496545, 1e-4);
}

BOOST_AUTO_TEST_CASE(CBOWModel_predict_context_3)
{
    CBOWModel model = get_model();
    std::vector<std::pair<double, int>> prediction = model.predict({1, 1, 0});
    BOOST_CHECK_EQUAL(4, prediction.size());
    double sum = 0;
    for (const auto& pred : prediction) {
        sum += pred.first;
        BOOST_CHECK(pred.first >= 0.0);
        BOOST_CHECK(pred.first <= 1.0);
        BOOST_CHECK(pred.second >= 0);
        BOOST_CHECK(pred.second < 4);
    }
    BOOST_CHECK_CLOSE(sum, 1.0, 1e-12);
    // output values (before softmax):
    // [0]: 2/3 * (0.04, 0.30, -0.20) dot (0.42, -0.28, 0.19) + 1/3 * (0.19, 0.11, -0.14) dot (0.42, -0.28, 0.19) = -0.062666
    // [1]: 2/3 * (0.04, 0.30, -0.20) dot (0.77, -0.93, 0.11) + 1/3 * (0.19, 0.11, -0.14) dot (0.77, -0.93, 0.11) = -0.1706
    // [2]: 2/3 * (0.04, 0.30, -0.20) dot (0.33, -0.43, -0.81) + 1/3 * (0.19, 0.11, -0.14) dot (0.33, -0.43, -0.81) = 0.073733
    // [3]: 2/3 * (0.04, 0.30, -0.20) dot (0.94, 0.90, -0.20) + 1/3 * ((0.19, 0.11, -0.14)  dot (0.94, 0.90, -0.20) = 0.3336
    // after softmax:
    // (0.22074601, 0.19816092, 0.25300588, 0.32808719)
    BOOST_CHECK_EQUAL(prediction[0].second, 3);
    BOOST_CHECK_CLOSE(prediction[0].first, 0.32808719, 1e-4);
    BOOST_CHECK_EQUAL(prediction[1].second, 2);
    BOOST_CHECK_CLOSE(prediction[1].first, 0.25300588, 1e-4);
    BOOST_CHECK_EQUAL(prediction[2].second, 0);
    BOOST_CHECK_CLOSE(prediction[2].first, 0.22074601, 1e-4);
    BOOST_CHECK_EQUAL(prediction[3].second, 1);
    BOOST_CHECK_CLOSE(prediction[3].first, 0.19816092, 1e-4);
}

BOOST_AUTO_TEST_CASE(CBOWModel_avg_log_prob)
{
    CBOWModel model = get_model();
    const std::vector<int> words{0, 2, 0, 1, 1, 2, 0};
    const double log_p = model.avg_log_prob(words);
    BOOST_CHECK(log_p <= 0.0);
    // std::cout << log_p << std::endl;
}

BOOST_AUTO_TEST_CASE(ReportData_constructor)
{
    ReportData data{10, -0.05, 1.0};
    BOOST_CHECK_EQUAL(10, data.epoch);
    BOOST_CHECK_EQUAL(-0.05, data.avg_log_prob);
    BOOST_CHECK_EQUAL(1.0, data.lr);
};

BOOST_AUTO_TEST_CASE(SimpleReporter_operator_parens)
{
    ReportData data{10, -0.05, 1.0};
    std::stringstream ss;
    SimpleReporter reporter(ss);
    reporter(data);
    BOOST_CHECK_EQUAL("epoch 10 prob -0.05 lr 1\n", ss.str());
}

BOOST_AUTO_TEST_CASE(SimpleLearningRate_constructor_default)
{
    SimpleLearningRate slr;
    BOOST_CHECK_EQUAL(1.0, slr.lr);
    BOOST_CHECK_EQUAL(3, slr.cutoff);
    BOOST_CHECK_EQUAL(0.5, slr.base);
}

BOOST_AUTO_TEST_CASE(SimpleLearningRate_constructor)
{
    SimpleLearningRate slr(4.0, 10, 0.25);
    BOOST_CHECK_EQUAL(4.0, slr.lr);
    BOOST_CHECK_EQUAL(10, slr.cutoff);
    BOOST_CHECK_EQUAL(0.25, slr.base);
}

BOOST_AUTO_TEST_CASE(SimpleLearningRate_operator_parens)
{
    SimpleLearningRate slr;
    BOOST_CHECK_EQUAL(1.0, slr(0));
    BOOST_CHECK_EQUAL(1.0, slr(1));
    BOOST_CHECK_EQUAL(1.0, slr(2));
    BOOST_CHECK_EQUAL(1.0, slr(3));
    BOOST_CHECK_EQUAL(0.5, slr(4));
    BOOST_CHECK_EQUAL(0.25, slr(5));
    BOOST_CHECK_EQUAL(0.125, slr(6));
}

BOOST_AUTO_TEST_CASE(Trainer_constructor)
{
    Trainer trainer;
    BOOST_CHECK_EQUAL(1, trainer.epochs);
    BOOST_CHECK_EQUAL(50, trainer.D);
    BOOST_CHECK_EQUAL(4, trainer.historyN);
    BOOST_CHECK_EQUAL(4, trainer.futureN);
    BOOST_CHECK_EQUAL(nullptr, trainer.initReporter);
    BOOST_CHECK_EQUAL(nullptr, trainer.epochReporter);
    BOOST_CHECK_EQUAL(nullptr, trainer.exitReporter);
    BOOST_CHECK_EQUAL(nullptr, trainer.learningRate);
}

BOOST_AUTO_TEST_CASE(Trainer_setters)
{
    std::stringstream ss;
    SimpleReporter initReporter(ss);
    SimpleReporter epochReporter(ss);
    SimpleReporter exitReporter(ss);
    SimpleLearningRate learningRate;
    Trainer trainer;
    trainer.setEpochs(20)
           .setEmbeddingSize(640)
           .setHistoryN(9)
           .setFutureN(7)
           .setInitReporter(&initReporter)
           .setEpochReporter(&epochReporter)
           .setExitReporter(&exitReporter)
           .setLearningRate(&learningRate);
    BOOST_CHECK_EQUAL(20, trainer.epochs);
    BOOST_CHECK_EQUAL(640, trainer.D);
    BOOST_CHECK_EQUAL(9, trainer.historyN);
    BOOST_CHECK_EQUAL(7, trainer.futureN);
    BOOST_CHECK_EQUAL(&initReporter, trainer.initReporter);
    BOOST_CHECK_EQUAL(&epochReporter, trainer.epochReporter);
    BOOST_CHECK_EQUAL(&exitReporter, trainer.exitReporter);
    BOOST_CHECK_EQUAL(&learningRate, trainer.learningRate);
}

BOOST_AUTO_TEST_CASE(Trainer_train)
{
    std::stringstream ss;
    SimpleReporter initReporter(ss);
    SimpleLearningRate learningRate;
    Trainer trainer;
    trainer.setEpochs(5)
           .setEmbeddingSize(2)
           .setHistoryN(1)
           .setFutureN(1)
           .setInitReporter(&initReporter)
           .setEpochReporter(&initReporter)
           .setExitReporter(&initReporter)
           .setLearningRate(&learningRate);
    std::vector<int> words{1, 0};
    CBOWModel model = trainer.train(words);
    BOOST_CHECK_EQUAL(2, model.W);
    BOOST_CHECK_EQUAL(2, model.D);
    BOOST_CHECK_EQUAL(1, model.historyN);
    BOOST_CHECK_EQUAL(1, model.futureN);
    BOOST_CHECK_EQUAL(2, model.P.size1());
    BOOST_CHECK_EQUAL(2, model.P.size2());
    BOOST_CHECK_EQUAL(2, model.O.size1());
    BOOST_CHECK_EQUAL(2, model.O.size2());
    // std::cout << ss.str() << std::endl;
    // BOOST_CHECK_EQUAL(std::string(""), ss.str());
}


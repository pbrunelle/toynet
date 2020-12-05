#include <toynet/examples/diff2/diff2.h>
#include <toynet/stlio.h>
#include <toynet/ublas/io.h>
#include <toynet/ublas/convert.h>
#include <iostream>
#include <random>
#include <boost/test/unit_test.hpp>

using namespace toynet;

template<class T>
std::string print(const T& v)
{
    std::ostringstream oss;
    oss << v;
    return oss.str();
}

BOOST_AUTO_TEST_CASE(test_Network)
{
    diff2::Network network(1, 3, 6, 2);
    BOOST_CHECK_EQUAL(1, network.hidden);
    BOOST_CHECK_EQUAL(3, network.width);
    BOOST_CHECK_EQUAL(6, network.inputs);
    BOOST_CHECK_EQUAL(2, network.outputs);
    BOOST_REQUIRE_EQUAL(2, network.W.size()); // 2 matrices: input -> hidden1, hidden1 -> output
    BOOST_CHECK_EQUAL(3, network.W[0].size1()); // 3 units per hidden layer
    BOOST_CHECK_EQUAL(6, network.W[0].size2()); // 6 units in input layer
    BOOST_CHECK_EQUAL(2, network.W[1].size1()); // 2 units in output layer
    BOOST_CHECK_EQUAL(3, network.W[1].size2()); // 3 units per hidden layer
}

BOOST_AUTO_TEST_CASE(test_Forward)
{
    diff2::Forward fwd;
    diff2::Network network(1, 2, 2, 1);
    std::vector<diff2::Tensor1D> A = network.get1D();
    BOOST_CHECK_EQUAL("[[0, 0], [0, 0], [0]]", print(A));
    fwd.forward(network, A, convert({4.0, 3.0}));
    BOOST_CHECK_EQUAL("[[4, 3], [-0.8, -0.1], [-0.14]]", print(A));
    fwd.forward(network, A, convert({3.0, 4.0}));
    BOOST_CHECK_EQUAL("[[3, 4], [-0.6, 0.1], [-0.14]]", print(A));
}

BOOST_AUTO_TEST_CASE(test_Workspace)
{
    diff2::Network network(1, 2, 3, 1);
    diff2::Workspace workspace(network, false, false, false);
    BOOST_CHECK(!workspace.dA);
    BOOST_CHECK(!workspace.dW);
    BOOST_CHECK(!workspace.v);
    BOOST_REQUIRE_EQUAL(3, workspace.A.size());
    BOOST_CHECK_EQUAL(3, workspace.A[0].size());
    BOOST_CHECK_EQUAL(2, workspace.A[1].size());
    BOOST_CHECK_EQUAL(1, workspace.A[2].size());
    BOOST_CHECK_EQUAL(0.0, workspace.loss);
}

BOOST_AUTO_TEST_CASE(test_GradientOptimizer)
{
    diff2::Forward fwd;
    diff2::Network network(1, 2, 2, 1);
    BOOST_CHECK_EQUAL("[[[-0.2, 0], [-0.1, 0.1]], [[0.2, -0.2]]]", print(network.W));
    diff2::GradientOptimizer opt(0.1);
    diff2::Workspace workspace(network, opt.computes_dA(), opt.computes_dW(), opt.computes_v());
    MSELoss loss;
    fwd.forward(network, workspace.A, convert({4.0, 3.0}));
    opt.compute_gradients(network, workspace, loss, convert({1.0}));
    BOOST_CHECK_EQUAL("[[4, 3], [-0.8, -0.1], [-0.14]]", print(workspace.A));
    BOOST_CHECK_EQUAL("[[[-1.824, -1.368], [1.824, 1.368]], [[1.824, 0.228]]]", print(*workspace.dW));
    BOOST_CHECK_EQUAL("[[0.0456, 0.0456], [-0.456, 0.456], [-2.28]]", print(*workspace.dA));
    BOOST_CHECK_CLOSE(1.2996, workspace.loss, 1e-6);
    opt.update_weights(1, network, workspace);
    BOOST_CHECK_EQUAL("[[[-0.0176, 0.1368], [-0.2824, -0.0368]], [[0.0176, -0.2228]]]", print(network.W));
}

// MomentumOptimizer

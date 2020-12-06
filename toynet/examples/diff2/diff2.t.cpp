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
    diff2::Network network(1, 2, 2, 1);
    diff2::Workspace workspace = diff2::build_workspace(network);
    BOOST_CHECK_EQUAL("[[0, 0], [0, 0], [0]]", print(workspace.A));
    network.forward(workspace, convert({4.0, 3.0}));
    BOOST_CHECK_EQUAL("[[4, 3], [-0.8, -0.1], [-0.14]]", print(workspace.A));
    network.forward(workspace, convert({3.0, 4.0}));
    BOOST_CHECK_EQUAL("[[3, 4], [-0.6, 0.1], [-0.14]]", print(workspace.A));
}

BOOST_AUTO_TEST_CASE(test_Workspace)
{
    diff2::Network network(1, 2, 3, 1);
    diff2::Workspace workspace = diff2::build_workspace(network);
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
    diff2::Network network(1, 2, 2, 1);
    diff2::GradientOptimizer opt(0.1);
    MSELoss loss;
    diff2::Workspace workspace = diff2::build_workspace(network, opt);
    BOOST_CHECK_EQUAL("[[[-0.2, 0], [-0.1, 0.1]], [[0.2, -0.2]]]", print(network.W));

    network.forward(workspace, convert({4.0, 3.0}));
    BOOST_CHECK_EQUAL("[[4, 3], [-0.8, -0.1], [-0.14]]", print(workspace.A));

    opt.compute_gradients(network, workspace, loss, convert({1.0}));
    BOOST_CHECK_EQUAL("[[[-1.824, -1.368], [1.824, 1.368]], [[1.824, 0.228]]]", print(*workspace.dW));
    BOOST_CHECK_EQUAL("[[0.0456, 0.0456], [-0.456, 0.456], [-2.28]]", print(*workspace.dA));
    BOOST_CHECK_CLOSE(1.2996, workspace.loss, 1e-6);

    opt.update_weights(1, network, workspace);
    BOOST_CHECK_EQUAL("[[[-0.0176, 0.1368], [-0.2824, -0.0368]], [[0.0176, -0.2228]]]", print(network.W));
}

// If the learning rate is too high, training will quickly diverge
BOOST_AUTO_TEST_CASE(test_GradientOptimizer_divergence)
{
    const double lr = 0.5;
    const diff2::Tensor1D x = convert({4.0, 3.0});
    const diff2::Tensor1D y = convert({1.0});

    diff2::Network network(1, 2, 2, 1);
    diff2::GradientOptimizer opt(lr);
    MSELoss loss;
    diff2::Workspace workspace = diff2::build_workspace(network, opt);

    // First epoch
    network.forward(workspace, x);
    opt.compute_gradients(network, workspace, loss, y);
    opt.update_weights(1, network, workspace);
    BOOST_CHECK_EQUAL("[[4, 3], [-0.8, -0.1], [-0.14]]", print(workspace.A));
    BOOST_CHECK_EQUAL("[[[0.712, 0.684], [-1.012, -0.584]], [[-0.712, -0.314]]]", print(network.W));
    BOOST_CHECK_CLOSE(1.2996, workspace.loss, 1e-6);

    // Second epoch
    network.forward(workspace, x);
    opt.compute_gradients(network, workspace, loss, y);
    opt.update_weights(2, network, workspace);
    BOOST_CHECK_EQUAL("[[4, 3], [4.9, -5.8], [-1.6676]]", print(workspace.A));
    BOOST_CHECK_EQUAL("[[1.00929, 1.61994], [3.79866, 1.67525], [-5.3352]]", print(*workspace.dA));
    BOOST_CHECK_EQUAL("[[[15.1946, 11.396], [6.70101, 5.02576]], [[-26.1425, 30.9442]]]", print(*workspace.dW));
    BOOST_CHECK_EQUAL("[[[-6.88532, -5.01399], [-4.36251, -3.09688]], [[12.3592, -15.7861]]]", print(network.W));
    BOOST_CHECK_CLOSE(7.11608976, workspace.loss, 1e-6);

    // Third epoch (forward and loss)
    network.forward(workspace, x);
    opt.compute_gradients(network, workspace, loss, y);
    BOOST_CHECK_EQUAL("[[4, 3], [-42.5833, -26.7407], [-104.167]]", print(workspace.A));
    BOOST_CHECK_CLOSE(11060.0515, workspace.loss, 1e-6);
}

BOOST_AUTO_TEST_CASE(test_MomentumOptimizer)
{
}

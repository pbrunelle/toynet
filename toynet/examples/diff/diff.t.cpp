#include <toynet/examples/diff/diff.h>
#include <toynet/stlio.h>
#include <toynet/ublas/io.h>
#include <toynet/ublas/convert.h>
#include <iostream>
#include <random>
#include <boost/test/unit_test.hpp>

using namespace toynet;

BOOST_AUTO_TEST_CASE(difference_numbers)
{
    DiffNumbers network(1, 2, 2);
    const std::vector<ublas::vector<double>> x(1, convert({4.0, 3.0}));

    BOOST_CHECK_EQUAL(
        "hidden 1 width 2 inputs 2 loss 0\n"
        "W : [[[-0.2, -0.1], [0, 0.1]], [[0.2], [-0.2]]]\n"
        "DW: [[[0, 0], [0, 0]], [[0], [0]]]\n"
        "A : [[0, 0], [0, 0], [0]]\n"
        "G : [[0, 0], [0, 0], [0]]\n",
        network.print());

    network.forward_backward(x);

    BOOST_CHECK_EQUAL(
        "hidden 1 width 2 inputs 2 loss 1.2996\n"
        "W : [[[-0.2, -0.1], [0, 0.1]], [[0.2], [-0.2]]]\n"
        "DW: [[[-1.824, 1.824], [-1.368, 1.368]], [[1.824], [0.228]]]\n"
        "A : [[4, 3], [-0.8, -0.1], [-0.14]]\n"
        "G : [[0.0456, 0.0456], [-0.456, 0.456], [-2.28]]\n",
        network.print());

    network.update_weights(0.1);
    network.forward_backward(x);

    // ./diff.tsk -e 2 --lr 0.1
    const std::string expected =
        "hidden 1 width 2 inputs 2 loss 0.515156\n"
        "W : [[[-0.0176, -0.2824], [0.1368, -0.0368]], [[0.0176], [-0.2228]]]\n"
        "DW: [[[-0.101058, 1.27931], [-0.0757938, 0.95948]], [[-0.488066], [1.78001]]]\n" // NOTE: not manually computed
        "A : [[4, 3], [0.34, -1.24], [0.282256]]\n"
        "G : [[-0.0898744, -0.0152258], [-0.0252646, 0.319827], [-1.43549]]\n"; // NOTE: not manually computed except for G[2]

    BOOST_CHECK_EQUAL(expected, network.print());

    // forward_backward is idempotent
    network.forward_backward(x);

    BOOST_CHECK_EQUAL(expected, network.print());

    for (int i = 0;  i < 50;  ++i) {
        network.forward_backward(x);
        network.update_weights(0.01);
    }
    BOOST_CHECK(network.loss < 1e-6);
}

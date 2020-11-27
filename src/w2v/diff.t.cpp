#include <diff.h>
#include <w2v.h>
#include <stlio.h>
#include <iostream>
#include <boost/test/unit_test.hpp>

using namespace w2v;

BOOST_AUTO_TEST_CASE(difference_numbers)
{
    DiffNumbers network(1, 2, 2);

    {
        std::stringstream ss;
        ss << network;
        const std::string expected =
            "hidden 1 width 2 inputs 2 loss 0\n"
            "W : [[[-0.2, -0.1], [0, 0.1]], [[0.2], [-0.2]]]\n"
            "DW: [[[0, 0], [0, 0]], [[0], [0]]]\n"
            "A : [[0, 0], [0]]\n"
            "G : [[0, 0], [0]]\n";
        BOOST_CHECK_EQUAL(expected, ss.str());
    }

    network.forward_backward({4.0, 3.0});

    {
        std::stringstream ss;
        ss << network;
        const std::string expected =
            "hidden 1 width 2 inputs 2 loss 1.2996\n"
            "W : [[[-0.2, -0.1], [0, 0.1]], [[0.2], [-0.2]]]\n"
            "DW: [[[0, 0], [0, 0]], [[0], [0]]]\n"
            "A : [[-0.8, -0.1], [-0.14]]\n"
            "G : [[0, 0], [0]]\n";
        BOOST_CHECK_EQUAL(expected, ss.str());
    }
}

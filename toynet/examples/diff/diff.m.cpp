#include <toynet/examples/diff/diff.h>
#include <toynet/stlio.h>
#include <toynet/ublas/io.h>
#include <toynet/ublas/convert.h>
#include <iostream>
#include <sstream>
#include <random>
#include <boost/program_options.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

using namespace boost::program_options;
using namespace toynet;

void print(const boost::property_tree::ptree& pt)
{
    boost::property_tree::ptree::const_iterator end = pt.end();
    for (boost::property_tree::ptree::const_iterator it = pt.begin(); it != end; ++it) {
        std::cout << it->first << ": " << it->second.get_value<std::string>() << std::endl;
        print(it->second);
    }
}

std::vector<ublas::vector<double>> parse_json_examples(const std::string& s)
{
    std::vector<ublas::vector<double>> ret;
    std::stringstream ss(s);
    boost::property_tree::ptree pt;
    boost::property_tree::read_json(ss, pt);
    for (const auto& e1 : pt) {
        ublas::vector<double> v;
        for (const auto& e2 : e1.second) {
            v.resize(v.size() + 1);
            v[v.size() - 1] = e2.second.get_value<double>();
        }
        ret.push_back(v);
    }
    return ret;
}

struct Options
{
    options_description desc;
    variables_map vm;
    bool help;
    int hidden;
    int width;
    int inputs;
    int epochs;
    float lr;
    std::string training;
    bool progress;

    Options(int argc, char* argv[])
        : desc("Allowed options")
        , help(false)
        , hidden(1)
        , width(2)
        , inputs(2)
        , epochs(10)
        , lr(0.01)
        , training("[[4.0, 3.0]]")
        , progress(true)
    {
        desc.add_options()
            // First parameter describes option name/short name
            // The second is parameter to option
            // The third is description
            ("help,h", bool_switch(&help), "print help message")
            ("hidden,H", value(&hidden), "number of hidden layers")
            ("width,W", value(&width), "width of hidden layers")
            ("inputs,I", value(&hidden), "width of input layer")
            ("epochs,e", value(&epochs), "number of training epochs")
            ("lr", value(&lr), "learning rate")
            ("training", value(&training), "training data")
            ("progress", value(&progress), "show loss at each epoch")
            ;
        store(parse_command_line(argc, argv, desc), vm);
        notify(vm);
    }
};

int main(int argc, char* argv[])
{
    Options opts(argc, argv);

    if (opts.help) {
        std::cout << opts.desc << std::endl;
        return 0;
    }

    const std::vector<ublas::vector<double>> training_examples = parse_json_examples(opts.training);

    // Make sure the dimensions of each training example is equal to opts.inputs
    for (const auto&v : training_examples) {
        if (v.size() != opts.inputs) {
            std::cerr << "Size mismatch: the network has input size "
                      << opts.inputs << " but an example has size "
                      << v.size() << ": " << v << std::endl;
            return 1;
        }
    }

    DiffNumbers network(opts.hidden, opts.width, opts.inputs);
    for (int i = 0;  i < opts.epochs;  ++i) {
        network.forward_backward(training_examples);
        if (opts.progress)
            std::cout << i << " " << network.loss << std::endl;
        network.update_weights(opts.lr);
    }
    std::cout << network << std::endl;

#if 0
    // Let's go for more examples!
    network.init_weights();
    std::random_device rd;
    std::mt19937 e2(rd());
    std::uniform_real_distribution<> dist(-20, 150);
    std::vector<ublas::vector<double>> x4(1000, convert({0.0, 0.0}));
    for (auto & ex : x4) {
        ex[0] = dist(e2);
        ex[1] = dist(e2);
    }
    for (int i = 0;  i < 20;  ++i) {
        network.forward_backward(x4);
        // std::cout << i << " " << network.loss << std::endl;
        // std::cout << network << std::endl;
        network.update_weights(0.00005); // we need a small step size otherwise training doesn't converge
    }
    // std::cout << network << std::endl;
    BOOST_CHECK(network.loss < 1e-6);
#endif

    return 0;
}

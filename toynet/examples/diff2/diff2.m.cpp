#include <toynet/examples/diff2/diff2.h>
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
    int outputs;
    int epochs;
    double lr;
    double alpha;
    std::string trainingx;
    std::string trainingy;
    std::string testing;
    bool progress;
    std::string loss;
    std::string optimizer;

    Options(int argc, char* argv[])
        : desc("Allowed options")
        , help(false)
        , hidden(1)
        , width(2)
        , inputs(2)
        , outputs(1)
        , epochs(10)
        , lr(0.01)
        , alpha(0.5)
        , trainingx("[[4.0, 3.0]]")
        , trainingy("[[1.0]]")
        , testing("")
        , progress(true)
        , loss("mse")
        , optimizer("gradient")
    {
        desc.add_options()
            // First parameter describes option name/short name
            // The second is parameter to option
            // The third is description
            ("help,h", bool_switch(&help), "print help message")
            ("hidden,H", value(&hidden), "number of hidden layers")
            ("width,W", value(&width), "width of hidden layers")
            ("inputs,I", value(&inputs), "width of input layer")
            ("outputs,O", value(&outputs), "width of output layer")
            ("epochs,e", value(&epochs), "number of training epochs")
            ("lr", value(&lr), "learning rate")
            ("alpha", value(&alpha), "momentum's alpha")
            ("trainingx", value(&trainingx), "training inputs")
            ("trainingy", value(&trainingy), "training outputs")
            ("testing", value(&testing), "testing X")
            ("progress", value(&progress), "show loss at each epoch")
            ("loss", value(&loss), "loss function")
            ("optimizer", value(&optimizer), "optimizer")
            ;
        store(parse_command_line(argc, argv, desc), vm);
        notify(vm);
    }
};

std::unique_ptr<Loss> get_loss(const std::string& name)
{
    if (name == "mse")
        return std::make_unique<MSELoss>();
    if (name == "softmax")
        return std::make_unique<SoftmaxLoss>();
    throw std::runtime_error("unknown loss function: " + name);
}

std::unique_ptr<diff2::Optimizer> get_optimizer(const std::string& name, double lr, double alpha)
{
    if (name == "gradient")
        return std::make_unique<diff2::GradientOptimizer>(lr);
    if (name == "momentum")
        return std::make_unique<diff2::MomentumOptimizer>(lr, alpha);
    throw std::runtime_error("unknown optimizer: " + name);
}

int main(int argc, char* argv[])
{
    Options opts(argc, argv);

    if (opts.help) {
        std::cout << opts.desc << std::endl;
        return 0;
    }

    const std::vector<diff2::Tensor1D> training_X = parse_json_examples(opts.trainingx);
    const std::vector<diff2::Tensor1D> training_Y = parse_json_examples(opts.trainingy);
    const std::vector<diff2::Tensor1D> testing = parse_json_examples(opts.testing);

    // Make sure the dimensions of each training example's X is equal to opts.inputs
    for (const auto&v : training_X) {
        if (v.size() != opts.inputs) {
            std::cerr << "Size mismatch: the network has input size "
                      << opts.inputs << " but a training input has size "
                      << v.size() << ": " << v << std::endl;
            return 1;
        }
    }

    // Make sure the dimensions of each training example's Y is equal to opts.outputs
    for (const auto&v : training_Y) {
        if (v.size() != opts.outputs) {
            std::cerr << "Size mismatch: the network has output size "
                      << opts.inputs << " but a training output has size "
                      << v.size() << ": " << v << std::endl;
            return 1;
        }
    }

    // Make sure the dimensions of each testing example is equal to opts.inputs
    for (const auto&v : testing) {
        if (v.size() != opts.inputs) {
            std::cerr << "Size mismatch: the network has input size "
                      << opts.inputs << " but a testing input has size "
                      << v.size() << ": " << v << std::endl;
            return 1;
        }
    }

    diff2::Network network(opts.hidden, opts.width, opts.inputs, opts.outputs);
    std::unique_ptr<Loss> loss = get_loss(opts.loss);
    std::unique_ptr<diff2::Optimizer> opt = get_optimizer(opts.optimizer, opts.lr, opts.alpha);
    diff2::Trainer trainer(network, *loss, *opt);
    for (int e = 1;  e <= opts.epochs;  ++e) {
        trainer.train(e, training_X, training_Y);
        if (opts.progress)
            std::cout << e << " " << trainer.workspace.loss << std::endl;
    }
    std::cout << "loss " << trainer.workspace.loss << " weights " << network.W << std::endl;

    for (const auto&v : testing)
        std::cout << v << " -> " << network.predict(v) << std::endl;

    return 0;
}

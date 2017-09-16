#include "dynet/dynet.h"

#include "logistic_regression.hh"

using namespace dyex;

int main(int argc, char** argv) {
    dynet::initialize(argc, argv);

    std::vector<std::vector<dynet::real>> x_values = {{1., 2}, {-0.5, -1.0}, {0.8, 2.3}};
    std::vector<dynet::real> y_value = {1, 0, 1};

    std::vector<std::vector<dynet::real>> x_test = {{0.3, 0.24}, {-1.7, -0.7}};

    logistic_regression clf = logistic_regression(2000, 0.5);
    auto pred = clf.fit_predict(x_values, y_value, x_test);
    for(auto e : pred) std::cout << e << std::endl;

}



#include "dynet/dynet.h"

#include "regression/logistic_regression.hh"

using namespace dyex;

int main(int argc, char** argv) {
    dynet::initialize(argc, argv);

    std::vector<std::vector<dynet::real>> x_values = {
        {1.0, 2},{-0.5, -1.0},{0.8, 2.3},{-1.0, -0.4},
        {0.2, 0.16}, {0.5, 0.1}, {-0.1, -0.3}, {0.1, 0.1}};
    std::vector<dynet::real> y_value = {1, 0, 1, 0, 1, 1, 0, 1};

    std::vector<std::vector<dynet::real>> x_test = {{0.3, 0.24}, {-1.7, -0.7}};

    logistic_regression clf = logistic_regression(100, 0.2);
    auto pred = clf.fit_predict(x_values, y_value, x_test);
    for(auto e : pred) std::cout << e << std::endl;

}



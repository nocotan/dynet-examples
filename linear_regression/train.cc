#include "dynet/dynet.h"

#include "linear_regression.hh"

using namespace dyex;

int main(int argc, char** argv) {
    dynet::initialize(argc, argv);

    std::vector<std::vector<dynet::real>> x_values = {{-3},{1},{2},{2}, {3},{4}};
    std::vector<dynet::real> y_value = {-6.1, 2.2, 2.4, 4.1, 3.8, 6.2, 8.3};

    std::vector<std::vector<dynet::real>> x_test = {{0}, {2}};

    linear_regression clf = linear_regression(100, 0.3);
    auto pred = clf.fit_predict(x_values, y_value, x_test);
    for(auto e : pred) std::cout << e << std::endl;

}



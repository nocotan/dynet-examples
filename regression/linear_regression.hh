#ifndef LINEAR_REGRESSION_HH
#define LINEAR_REGRESSION_HH

#include <iostream>
#include <tuple>
#include <vector>

#include "dynet/dynet.h"
#include "dynet/expr.h"
#include "dynet/io.h"
#include "dynet/model.h"
#include "dynet/training.h"


namespace dyex {

class linear_regression {
    private:
        unsigned max_iter;
        long double learning_rate;
        std::tuple<unsigned, unsigned> shape;

        dynet::ParameterCollection pc;
        dynet::Expression w;

    public:
        linear_regression(unsigned max_iter=100, long double learning_rate=0.5);
        std::vector<dynet::real> fit_predict(std::vector<std::vector<dynet::real>>, std::vector<dynet::real>,
                std::vector<std::vector<dynet::real>>);
};

linear_regression::linear_regression(unsigned max_iter, long double learning_rate)
    : max_iter(max_iter), learning_rate(learning_rate) { }

std::vector<dynet::real>  linear_regression::fit_predict(std::vector<std::vector<dynet::real>> x,
        std::vector<dynet::real> y, std::vector<std::vector<dynet::real>> x_test) {

    dynet::SimpleSGDTrainer trainer(pc);
    dynet::ComputationGraph cg;


    this->shape = std::tuple<int, int>(x.size(), x[0].size());
    unsigned r = std::get<0>(this->shape);
    unsigned c = std::get<1>(this->shape);


    w = dynet::parameter(cg, pc.add_parameters({1, c}));
    std::vector<dynet::Expression> X;
    std::vector<dynet::Expression> Y;

    for(int i=0; i<r; ++i) {
        X.push_back(dynet::input(cg, {c}, x[i]));
        Y.push_back(dynet::input(cg, y[i]));
    }

    for(int i=0; i<=max_iter; ++i) {
        dynet::Expression y_pred = w*X[0];
        dynet::Expression l = dynet::squared_distance(y_pred, Y[0]);
        for(int j=1; j<r; ++j) {
            y_pred = w*X[j];
            l = l + dynet::squared_distance(y_pred, Y[j]);
        }
        l = l / r;
        dynet::real loss = dynet::as_scalar(cg.forward(l));
        if(i%10==0) std::cout << "iter: " << i << ", loss = " << loss << std::endl;
        cg.backward(l);
        trainer.update();
    }

    std::vector<dynet::real> pred;
    for(int i=0; i<x_test.size(); ++i) {
        dynet::Expression e = dynet::input(cg, {c}, x_test[i]);
        pred.push_back(dynet::as_scalar(cg.forward(w*e)));
    }

    return pred;
}

} // namespace dyex

#endif

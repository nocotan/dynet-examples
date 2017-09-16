## DyNet examples
Implementation of various neural networks with DyNet.
* [DyNet](https://github.com/clab/dynet)

## Implementation list
* Linear Regression
* Logistic Regression

## How to Compile

```c++
g++ -std=c++14 -I$HOME/GitHub/dynet -L$HOME/GitHub/dynet/build/dynet train.cc -ldynet
```

## Instalation DyNet

### Instalation dependencies

#### On Ubuntu

```
sudo apt-get install build-essential cmake mercurial
```

#### On MacOS

```
xcode-select --install
brew install cmake hg  # Using homebrew.
sudo port install cmake mercurial # Using macports.
```

### Install Eigen

```
hg clone https://bitbucket.org/eigen/eigen/ -r 346ecdb
```

## C++ Instalation

```
# Clone the github repository
git clone https://github.com/clab/dynet.git
cd dynet
# Checkout the latest release
git checkout tags/v2.0
mkdir build
cd build
# Run CMake
cmake .. -DEIGEN3_INCLUDE_DIR=/path/to/eigen -DENABLE_CPP_EXAMPLES=ON
# Compile using 2 processes
make -j 2
# Test with an example
./examples/train_xo
```


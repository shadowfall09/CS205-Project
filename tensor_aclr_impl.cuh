#ifndef TENSOR_ACLR_IMPL_CUH
#define TENSOR_ACLR_IMPL_CUH

#include "tensor_aclr.cuh"
#include <random>
#include <algorithm>
#include <typeinfo>
#include <numeric>

namespace ts {
    // Random Generation Part
    const double MIN_RANDOM = 0, MAX_RANDOM = 10;
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<> dis(MIN_RANDOM, MAX_RANDOM);

    double random() {
        return dis(gen);
    }

    // Helper function to calculate total size from shape
    int totalSize(const std::vector<int> &shape) {
        return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    }


    template<typename T>
    Tensor<T>::Tensor(std::vector<int> shape, T defaultValue)
            : data(totalSize(shape), defaultValue), shape(shape), type(typeid(T).name()) {}

    template<typename T>
    Tensor<T>::Tensor(std::vector<int> shape)
            : Tensor(shape, T{}) {}

    template<typename T>
    Tensor<T> rand(std::vector<int> size) {
        Tensor<T> tensor(size);
        std::generate(tensor.data.begin(), tensor.data.end(),
                      []() { return static_cast<T>(random()); });
        return tensor;
    }

    template<typename T>
    Tensor<T> zeros(std::vector<int> size) {
        return Tensor<T>(size, 0);
    }

    template<typename T>
    Tensor<T> ones(std::vector<int> size) {
        return Tensor<T>(size, 1);
    }

    template<typename T>
    Tensor<T> full(std::vector<int> size, T value) {
        return Tensor<T>(size, value);
    }

    template<typename T>
    Tensor<T> eye(int n) {
        Tensor<T> tensor({n, n}, 0);
        for (int i = 0; i < n; ++i) {
            tensor.data[i * n + i] = 1;
        }
        return tensor;
    }

    template<typename T>
    std::ostream &operator<<(std::ostream &os, const Tensor<T> &tensor) {
        int dimension = 0;
        int spaces = -1;
        for (int i = 0; i < tensor.shape.size(); ++i) {
            os << "[";
            dimension = tensor.shape[i];
            spaces++;
        }

        for (int i = 0; i < tensor.data.size(); ++i) {
            os << tensor.data[i];
            if ((i + 1) % dimension == 0 && i != tensor.data.size() - 1) {
                os << "],\n";
                for (int j = 0; j < spaces; ++j) {
                    os << " ";
                }
                os << "[";
            } else if (i != tensor.data.size() - 1) {
                os << ", ";
            }
        }

        for (int i = 0; i < tensor.shape.size(); ++i) {
            os << "]";
        }
        return os;
    }
}

#endif // TENSOR_ACLR_IMPL_CUH

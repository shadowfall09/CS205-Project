#ifndef TENSOR_ACLR_IMPL_CUH
#define TENSOR_ACLR_IMPL_CUH

#include "tensor_aclr.cuh"
#include <random>
#include <algorithm>
#include <typeinfo>
#include <numeric>
#include <utility>

namespace ts {
    // ======= Random Part =======
    const double MIN_RANDOM = 0, MAX_RANDOM = 10;
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<> dis(MIN_RANDOM, MAX_RANDOM);

    double random() {
        return dis(gen);
    }
    // ======= Random Part End =======

    // Helper function to calculate total shape from shape
    int totalSize(const std::vector<int> &shape) {
        return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    }

    // Helper function to print tensor
    template<typename T>
    std::ostream &operator<<(std::ostream &os, const Tensor<T> &tensor) {
        int dimension = 0;
        int spaces = -1;
        int data_size = totalSize(tensor.shape);
        for (int i = 0; i < tensor.shape.size(); ++i) {
            os << "[";
            dimension = tensor.shape[i];
            spaces++;
        }

        for (int i = 0; i < data_size; ++i) {
            os << *(tensor.data + i);
            if ((i + 1) % dimension == 0 && i != data_size - 1) {
                os << "],\n";
                for (int j = 0; j < spaces; ++j) {
                    os << " ";
                }
                os << "[";
            } else if (i != data_size - 1) {
                os << ", ";
            }
        }

        for (int i = 0; i < tensor.shape.size(); ++i) {
            os << "]";
        }
        return os;
    }

    // ======= Class Constructor =======
    template<typename T>
    Tensor<T>::Tensor(const std::vector<int> &shape, T defaultValue)
            :  shape(shape), type(typeid(T).name()) {
        cudaMallocManaged(&data, totalSize(shape) * sizeof(T));
        std::fill(data, data + totalSize(shape), defaultValue);
    }

    template<typename T>
    Tensor<T>::Tensor(std::vector<int> shape, T *data)
            : data(data), shape(std::move(shape)), type(typeid(T).name()) {}
    // ======= Class Constructor End =======

    // ======= 1. Creation and Initialization =======
    template<typename T>
    Tensor<T> tensor(std::vector<int> shape, T *data) {
        return Tensor<T>(shape, data);
    }

    template<typename T>
    Tensor<T> tensor(std::vector<int> shape, T defaultValue) {
        return Tensor<T>(shape, defaultValue);
    }

    template<typename T>
    Tensor<T> rand(std::vector<int> shape) {
        Tensor<T> tensor(shape, nullptr);
        cudaMallocManaged(&tensor.data, totalSize(shape) * sizeof(T));
        for (int i = 0; i < totalSize(shape); ++i) {
            tensor.data[i] = random();
        }
        return tensor;
    }

    template<typename T>
    Tensor<T> zeros(std::vector<int> shape) {
        return Tensor<T>(shape, 0);
    }

    template<typename T>
    Tensor<T> ones(std::vector<int> shape) {
        return Tensor<T>(shape, 1);
    }

    template<typename T>
    Tensor<T> full(std::vector<int> shape, T value) {
        return Tensor<T>(shape, value);
    }

    template<typename T>
    Tensor<T> eye(int n) {
        Tensor<T> tensor({n, n}, 0);
        for (int i = 0; i < n; ++i) {
            tensor.data[i * n + i] = 1;
        }
        return tensor;
    }
    // ======= 1. Creation and Initialization End =======

    // ======= 2. Indexing and Slicing =======
    template<typename T>
    Tensor<T> Tensor<T>::operator()(int index) {
        std::vector<int> new_shape = shape;
        new_shape.erase(new_shape.begin());
        Tensor<T> tensor(new_shape, data + index * totalSize(new_shape));
        return tensor;
    }

    // example: ts::Tensor t2 = t(2,{2,4});
    // This slices the third to fifth (excluded) elements of
    //the third dimension of t.
    template<typename T>
    Tensor<T> Tensor<T>::operator()(int axis, std::vector<int> indices) {
        // indices must be {start, end}
        // include start, exclude end
        std::vector<int> new_shape = shape;
        new_shape[axis] = indices[1] - indices[0];
        Tensor<T> tensor(new_shape, data + indices[0] * totalSize(new_shape));
        return tensor;
    }

}

#endif // TENSOR_ACLR_IMPL_CUH

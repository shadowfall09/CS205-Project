#ifndef TENSOR_ACLR_IMPL_CUH
#define TENSOR_ACLR_IMPL_CUH

#include "tensor_aclr.cuh"
#include <random>
#include <algorithm>
#include <typeinfo>
#include <numeric>
#include <utility>
#include <cassert>
#include <cmath>

namespace ts {
    // ======= Random Part =======
    const double MIN_RANDOM = 0, MAX_RANDOM = 10;
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<> dis(MIN_RANDOM, MAX_RANDOM);

    /**
     * @brief Generate a random number between MIN_RANDOM and MAX_RANDOM
     * @return A random number between MIN_RANDOM and MAX_RANDOM
     */
    double random() {
        return dis(gen);
    }
    // ======= Random Part End =======

    // Helper function to calculate total shape from shape
    /**
     * @brief Calculate total size from shape
     * @param shape
     * @return Total size from shape
     */
    int totalSize(const std::vector<int> &shape) {
        return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    }

    // Helper function to print tensor
    /**
     * @brief Print tensor
     * @tparam T
     * @param os
     * @param tensor
     */
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
    /**
     * @brief Construct a new Tensor object, filled with the given value
     * @tparam T
     * @param shape
     * @param defaultValue
     * @return A new Tensor object, filled with the given value
     */
    template<typename T>
    Tensor<T>::Tensor(const std::vector<int> &shape, T defaultValue)
            :  shape(shape), type(typeid(T).name()) {
        cudaMallocManaged(&data, totalSize(shape) * sizeof(T));
        std::fill(data, data + totalSize(shape), defaultValue);
    }

    /**
     * @brief Construct a new Tensor object, filled with the given data
     * @tparam T
     * @param shape
     * @param data
     * @return A new Tensor object, filled with the given data
     */
    template<typename T>
    Tensor<T>::Tensor(std::vector<int> shape, T *data)
            : data(data), shape(std::move(shape)), type(typeid(T).name()) {}

    /**
     * @brief Construct a new Tensor object from another Tensor object
     * @tparam T
     * @param other
     * @return A new Tensor object from another Tensor object
     */
    template<typename T>
    Tensor<T>::Tensor(const Tensor<T> &other) {
        shape = other.shape;
        type = other.type;
        cudaMallocManaged(&data, totalSize(shape) * sizeof(T));
        cudaMemcpy(data, other.data, totalSize(shape) * sizeof(T), cudaMemcpyDeviceToDevice);
    }
    // ======= Class Constructor End =======

    // ======= 1. Creation and Initialization =======
    /**
     * @brief Create a tensor of given shape and fill it with the given data
     * @tparam T
     * @param shape
     * @param data
     * @return A tensor of given shape and filled with the given data
     */
    template<typename T>
    Tensor<T> tensor(std::vector<int> shape, T *data) {
        return Tensor<T>(shape, data);
    }

    /**
     * @brief Create a tensor of given shape and fill it with the given value
     * @tparam T
     * @param shape
     * @param defaultValue
     * @return A tensor of given shape and filled with the given value
     */
    template<typename T>
    Tensor<T> tensor(std::vector<int> shape, T defaultValue) {
        return Tensor<T>(shape, defaultValue);
    }

    /**
     * @brief Create a tensor of given shape and fill it with random values
     * @tparam T
     * @param shape
     * @return A tensor of given shape and filled with random values
     */
    template<typename T>
    Tensor<T> rand(std::vector<int> shape) {
        Tensor<T> tensor(shape, nullptr);
        cudaMallocManaged(&tensor.data, totalSize(shape) * sizeof(T));
        for (int i = 0; i < totalSize(shape); ++i) {
            tensor.data[i] = random();
        }
        return tensor;
    }

    /**
     * @brief Create a tensor of given shape and fill it with 0
     * @tparam T
     * @param shape
     * @return A tensor of given shape and filled with 0
     */
    template<typename T>
    Tensor<T> zeros(std::vector<int> shape) {
        return Tensor<T>(shape, 0);
    }

    /**
     * @brief Create a tensor of given shape and fill it with 1
     * @tparam T
     * @param shape
     * @return A tensor of given shape and filled with 1
     */
    template<typename T>
    Tensor<T> ones(std::vector<int> shape) {
        return Tensor<T>(shape, 1);
    }

    /**
     * @brief Create a tensor of given shape and fill it with the given value
     * @tparam T
     * @param shape
     * @param value
     * @return A tensor of given shape and filled with the given value
     */
    template<typename T>
    Tensor<T> full(std::vector<int> shape, T value) {
        return Tensor<T>(shape, value);
    }

    /**
     * @brief Create an identity matrix of size n
     * @tparam T
     * @param n
     * @return An identity matrix of size n
     */
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

    /**
     * @brief Get the element at the given index
     * @tparam T
     * @param index
     * @return The element at the given index
     */
    template<typename T>
    Tensor<T> Tensor<T>::operator()(int index) {
        std::vector<int> new_shape = shape;
        new_shape.erase(new_shape.begin());
        Tensor<T> tensor(new_shape, data + index * totalSize(new_shape));
        return tensor;
    }

    /**
     * @brief Get the slice of the tensor at the given axis and indices
     * @tparam T
     * @param axis
     * @param indices
     * @return The slice of the tensor at the given axis and indices
     */
    template<typename T>
    Tensor<T> Tensor<T>::operator()(int axis, std::vector<int> indices) {
        if (indices.size() < 2) {
            throw std::invalid_argument("indices size must be greater than 1");
        }

        std::vector<int> new_shape = shape;
        new_shape[1] = indices[1] - indices[0];
        Tensor<T> tensor(new_shape, data + indices[0] * totalSize(new_shape));
        return tensor;
    }

    // ======= CUDA FUNCTIONS =======
    // Add with scalar
    template<typename T>
    __global__ void addScalarKernel(T *data, T value, T *data3, int size) {
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        if (index < size) {
            data3[index] = data[index] + value;
        }
    }

    // Add
    template<typename T>
    __global__ void addKernel(T *data1, T *data2, T *data3, int size) {
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        if (index < size) {
            data3[index] = data1[index] + data2[index];
        }
    }

    // Subtract with scalar
    template<typename T>
    __global__ void subScalarKernel(T *data, T value, T *data3, int size) {
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        if (index < size) {
            data3[index] = data[index] - value;
        }
    }

    // Subtract
    template<typename T>
    __global__ void subKernel(T *data1, T *data2, T *data3, int size) {
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        if (index < size) {
            data3[index] = data1[index] - data2[index];
        }
    }

    // Multiply with scalar
    template<typename T>
    __global__ void mulScalarKernel(T *data, T value, T *result, int size) {
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        if (index < size) {
            result[index] = data[index] * value;
        }
    }

    // Multiply
    template<typename T>
    __global__ void mulKernel(T *data1, T *data2, T *result, int size) {
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        if (index < size) {
            result[index] = data1[index] * data2[index];
        }
    }

    // Divide with scalar
    template<typename T>
    __global__ void divScalarKernel(T *data, T value, T *result, int size) {
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        if (index < size) {
            result[index] = data[index] / value;
        }
    }

    // Divide
    template<typename T>
    __global__ void divKernel(T *data1, T *data2, T *result, int size) {
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        if (index < size) {
            result[index] = data1[index] / data2[index];
        }
    }

    // Log
    template<typename T>
    __global__ void logKernel(T *data, T *result, int size) {
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        if (index < size) {
            result[index] = log(data[index]);
        }
    }

    // EQ comparison
    template<typename T>
    __global__ void eqKernel(const T *data1, const T *data2, bool *result, int size) {
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        if (index < size) {
            result[index] = (data1[index] == data2[index]);
        }
    }

    // NE comparison
    template<typename T>
    __global__ void neKernel(const T *data1, const T *data2, bool *result, int size) {
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        if (index < size) {
            result[index] = (data1[index] != data2[index]);
        }
    }

    // GT comparison
    template<typename T>
    __global__ void gtKernel(const T *data1, const T *data2, bool *result, int size) {
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        if (index < size) {
            result[index] = (data1[index] > data2[index]);
        }
    }

    // GE comparison
    template<typename T>
    __global__ void geKernel(const T *data1, const T *data2, bool *result, int size) {
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        if (index < size) {
            result[index] = (data1[index] >= data2[index]);
        }
    }

    // LT comparison
    template<typename T>
    __global__ void ltKernel(const T *data1, const T *data2, bool *result, int size) {
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        if (index < size) {
            result[index] = (data1[index] < data2[index]);
        }
    }

    // LE comparison
    template<typename T>
    __global__ void leKernel(const T *data1, const T *data2, bool *result, int size) {
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        if (index < size) {
            result[index] = (data1[index] <= data2[index]);
        }
    }

    // ======= CUDA FUNCTIONS END =======

    // ======= 3.1 Pointwise operations =======

    // ======= 3.1.1 Add =======
    template<typename T>
    Tensor<T> Tensor<T>::add(const Tensor<T> &other) {
        assert(shape == other.shape);
        assert(type == other.type);
        Tensor<T> result(*this);
        int size = shape.size();
        addKernel<<<(size + 511) / 512, 512>>>(data, other.data, result.data, totalSize(shape));
        cudaDeviceSynchronize();
        return result;
    }


    template<typename T>
    Tensor<T> Tensor<T>::add(T value) {
        Tensor<T> result(*this);
        int size = shape.size();
        addScalarKernel<<<(size + 511) / 512, 512>>>(data, value, result.data, totalSize(shape));
        cudaDeviceSynchronize();
        return result;
    }

    template<typename T>
    Tensor<T> add(Tensor<T> &t1, Tensor<T> &t2) {
        return t1.add(t2);
    }

    template<typename T>
    Tensor<T> add(Tensor<T> &tensor, T value) {
        return tensor.add(value);
    }

    template<typename T>
    Tensor<T> Tensor<T>::operator+(const Tensor<T> &other) {
        return add(other);
    }

    template<typename T>
    Tensor<T> Tensor<T>::operator+(T value) {
        return add(value);
    }
    // ======= 3.1.1 Add End =======

    // ======= 3.1.2 Subtract =======
    template<typename T>
    Tensor<T> Tensor<T>::sub(const Tensor<T> &other) {
        assert(shape == other.shape);
        assert(type == other.type);
        Tensor<T> result(*this);
        int size = shape.size();
        subKernel<<<(size + 511) / 512, 512>>>(data, other.data, result.data, totalSize(shape));
        cudaDeviceSynchronize();
        return result;
    }

    template<typename T>
    Tensor<T> Tensor<T>::sub(T value) {
        Tensor<T> result(*this);
        int size = shape.size();
        subScalarKernel<<<(size + 511) / 512, 512>>>(data, value, result.data, totalSize(shape));
        cudaDeviceSynchronize();
        return result;
    }

    template<typename T>
    Tensor<T> sub(Tensor<T> &t1, Tensor<T> &t2) {
        return t1.sub(t2);
    }

    template<typename T>
    Tensor<T> sub(Tensor<T> &tensor, T value) {
        return tensor.sub(value);
    }

    template<typename T>
    Tensor<T> Tensor<T>::operator-(const Tensor<T> &other) {
        return sub(other);
    }

    template<typename T>
    Tensor<T> Tensor<T>::operator-(T value) {
        return sub(value);
    }
    // ======= 3.1.2 Subtract End =======

    // ======= 3.1.3 Multiply =======
    template<typename T>
    Tensor<T> Tensor<T>::mul(const Tensor<T> &other) {
        assert(shape == other.shape);
        assert(type == other.type);
        Tensor<T> result(*this);
        int size = shape.size();
        mulKernel<<<(size + 511) / 512, 512>>>(data, other.data, result.data, totalSize(shape));
        cudaDeviceSynchronize();
        return result;
    }

    template<typename T>
    Tensor<T> Tensor<T>::mul(T value) {
        Tensor<T> result(*this);
        int size = shape.size();
        mulScalarKernel<<<(size + 511) / 512, 512>>>(data, value, result.data, totalSize(shape));
        cudaDeviceSynchronize();
        return result;
    }

    template<typename T>
    Tensor<T> mul(Tensor<T> &t1, Tensor<T> &t2) {
        return t1.mul(t2);
    }

    template<typename T>
    Tensor<T> mul(Tensor<T> &tensor, T value) {
        return tensor.mul(value);
    }

    template<typename T>
    Tensor<T> Tensor<T>::operator*(const Tensor<T> &other) {
        return mul(other);
    }

    template<typename T>
    Tensor<T> Tensor<T>::operator*(T value) {
        return mul(value);
    }
    // ======= 3.1.3 Multiply End =======

    // ======= 3.1.4 Divide =======
    template<typename T>
    Tensor<T> Tensor<T>::div(const Tensor<T> &other) {
        assert(shape == other.shape);
        assert(type == other.type);
        Tensor<T> result(*this);
        int size = shape.size();
        divKernel<<<(size + 511) / 512, 512>>>(data, other.data, result.data, totalSize(shape));
        cudaDeviceSynchronize();
        return result;
    }

    template<typename T>
    Tensor<T> Tensor<T>::div(T value) {
        Tensor<T> result(*this);
        int size = shape.size();
        divScalarKernel<<<(size + 511) / 512, 512>>>(data, value, result.data, totalSize(shape));
        cudaDeviceSynchronize();
        return result;
    }

    template<typename T>
    Tensor<T> div(Tensor<T> &t1, Tensor<T> &t2) {
        return t1.div(t2);
    }

    template<typename T>
    Tensor<T> div(Tensor<T> &tensor, T value) {
        return tensor.div(value);
    }

    template<typename T>
    Tensor<T> Tensor<T>::operator/(const Tensor<T> &other) {
        return div(other);
    }

    template<typename T>
    Tensor<T> Tensor<T>::operator/(T value) {
        return div(value);
    }
    // ======= 3.1.4 Divide End =======

    // ======= 3.1.5 Log ======
    template<typename T>
    Tensor<T> Tensor<T>::log() {
        Tensor<T> result(*this);
        int size = shape.size();
        logKernel<<<(size + 511) / 512, 512>>>(data, result.data, totalSize(shape));
        cudaDeviceSynchronize();
        return result;
    }

    template<>
    Tensor<bool> Tensor<bool>::log() {
        throw std::invalid_argument("log operation is not supported for Tensor<bool>");
    }

    template<typename T>
    Tensor<T> log(Tensor<T> &tensor) {
        return tensor.log();
    }
    // ======= 3.1.5 Log End ======

    // ======= 3.1 Pointwise operations End =======

    // ======= 3.3 Comparison operations =======

    template<typename T>
    Tensor<bool> Tensor<T>::eq(const Tensor<T> &other) {
        assert(shape == other.shape);
        assert(type == other.type);
        Tensor<bool> result(shape, false);
        int size = shape.size();
        eqKernel<<<(size + 511) / 512, 512>>>(data, other.data, result.data, totalSize(shape));
        cudaDeviceSynchronize();
        return result;
    }

    template<typename T>
    Tensor<bool> Tensor<T>::ne(const Tensor<T> &other) {
        assert(shape == other.shape);
        assert(type == other.type);
        Tensor<bool> result(shape, false);
        int size = shape.size();
        neKernel<<<(size + 511) / 512, 512>>>(data, other.data, result.data, totalSize(shape));
        cudaDeviceSynchronize();
        return result;
    }

    template<typename T>
    Tensor<bool> Tensor<T>::gt(const Tensor<T> &other) {
        assert(shape == other.shape);
        assert(type == other.type);
        Tensor<bool> result(shape, false);
        int size = shape.size();
        gtKernel<<<(size + 511) / 512, 512>>>(data, other.data, result.data, totalSize(shape));
        cudaDeviceSynchronize();
        return result;
    }

    template<typename T>
    Tensor<bool> Tensor<T>::ge(const Tensor<T> &other) {
        assert(shape == other.shape);
        assert(type == other.type);
        Tensor<bool> result(shape, false);
        int size = shape.size();
        geKernel<<<(size + 511) / 512, 512>>>(data, other.data, result.data, totalSize(shape));
        cudaDeviceSynchronize();
        return result;
    }

    template<typename T>
    Tensor<bool> Tensor<T>::lt(const Tensor<T> &other) {
        assert(shape == other.shape);
        assert(type == other.type);
        Tensor<bool> result(shape, false);
        int size = shape.size();
        ltKernel<<<(size + 511) / 512, 512>>>(data, other.data, result.data, totalSize(shape));
        cudaDeviceSynchronize();
        return result;
    }

    template<typename T>
    Tensor<bool> Tensor<T>::le(const Tensor<T> &other) {
        assert(shape == other.shape);
        assert(type == other.type);
        Tensor<bool> result(shape, false);
        int size = shape.size();
        leKernel<<<(size + 511) / 512, 512>>>(data, other.data, result.data, totalSize(shape));
        cudaDeviceSynchronize();
        return result;
    }

    template<typename T>
    Tensor<bool> Tensor<T>::operator!=(const Tensor<T> &other) {
        return ne(other);
    }

    template<typename T>
    Tensor<bool> Tensor<T>::operator>(const Tensor<T> &other) {
        return gt(other);
    }

    template<typename T>
    Tensor<bool> Tensor<T>::operator>=(const Tensor<T> &other) {
        return ge(other);
    }

    template<typename T>
    Tensor<bool> Tensor<T>::operator<(const Tensor<T> &other) {
        return lt(other);
    }

    template<typename T>
    Tensor<bool> Tensor<T>::operator<=(const Tensor<T> &other) {
        return le(other);
    }

    template<typename T>
    Tensor<bool> Tensor<T>::operator==(const Tensor<T> &other) {
        return eq(other);
    }

    template<typename T>
    Tensor<bool> eq(Tensor<T> &t1, Tensor<T> &t2) {
        return t1.eq(t2);
    }

    template<typename T>
    Tensor<bool> ne(Tensor<T> &t1, Tensor<T> &t2) {
        return t1.ne(t2);
    }

    template<typename T>
    Tensor<bool> gt(Tensor<T> &t1, Tensor<T> &t2) {
        return t1.gt(t2);
    }

    template<typename T>
    Tensor<bool> ge(Tensor<T> &t1, Tensor<T> &t2) {
        return t1.ge(t2);
    }

    template<typename T>
    Tensor<bool> lt(Tensor<T> &t1, Tensor<T> &t2) {
        return t1.lt(t2);
    }

    template<typename T>
    Tensor<bool> le(Tensor<T> &t1, Tensor<T> &t2) {
        return t1.le(t2);
    }

    // ======= 3.3 Comparison operations End =======
}

#endif // TENSOR_ACLR_IMPL_CUH

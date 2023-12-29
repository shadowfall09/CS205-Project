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
    Tensor<T>::Tensor(const std::vector<int> &shape, T *data)
            : data(data), shape(shape), type(typeid(T).name()) {}

    /**
     * @brief Construct a new Tensor object from another Tensor object
     * @tparam T
     * @param other
     * @return A new Tensor object from another Tensor object
     */
    template<typename T>
    Tensor<T>::Tensor(const Tensor<T> &other)
            : data(other.data), shape(other.shape), type(other.type) {}

    // ======= Class Constructor End =======


    // ======= 0. Helper Functions =======

    // ------- Random Part -------
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
    // ------- Random Part End -------

    // Helper function to calculate total shape from shape
    /**
     * @brief Calculate total size from shape
     * @param shape
     * @return Total size from shape
     */
    int totalSize(const std::vector<int> &shape) {
        if (shape.empty()) return 1;
        return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    }

    /**
     * @brief Print tensor
     * @tparam T
     * @param os
     * @param tensor
     * @return
     */
    template<typename T>
    std::ostream &operator<<(std::ostream &os, const Tensor<T> &tensor) {
        printTensor(os, tensor, 0, 0, "");
        return os;
    }

    // Helper function to print tensor
    template<typename T>
    void printTensor(std::ostream &os, const Tensor<T> &tensor, int index, int dimension, const std::string &indent) {
        if (dimension == tensor.shape.size()) {
            os << tensor.data[index];
            return;
        }

        os << "[";
        int dimSize = tensor.shape[dimension];
        int nextIndexStep = (dimension < tensor.shape.size() - 1)
                            ? totalSize(std::vector<int>(tensor.shape.begin() + dimension + 1, tensor.shape.end()))
                            : 1;
        for (int i = 0; i < dimSize; ++i) {
            if (i > 0) {
                os << ", ";
                if (dimension != tensor.shape.size() - 1) {
                    os << "\n" << indent << std::string(dimension + 1, ' ');
                }
            }
            printTensor(os, tensor, index + i * nextIndexStep, dimension + 1, indent);
        }
        os << "]";
    }

    /**
     * @brief Get the index in flat array from index and shape
     * @param index
     * @param shape
     * @return The index in flat array from index and shape
     */
    int indexInFlatArray(const std::vector<int> &index, const std::vector<int> &shape) {
        int flatIndex = 0;
        int accum = 1;
        for (int i = shape.size() - 1; i >= 0; --i) {
            flatIndex += index[i] * accum;
            accum *= shape[i];
        }
        return flatIndex;
    }

    // deep copy
    /**
     * @brief Deep copy
     * @tparam T
     * @param other
     * @return A deep copy of the given tensor
     */
    template<typename T>
    Tensor<T> deepcopy(const Tensor<T> &other) {
        Tensor<T> result(other);
        cudaMallocManaged(&result.data, totalSize(result.shape) * sizeof(T));
        cudaMemcpy(result.data,
                   other.data,
                   totalSize(result.shape) * sizeof(T),
                   cudaMemcpyDeviceToDevice);
    }

    template<typename T>
    Tensor<T>::operator bool() const {
        return !shape.empty() && data != nullptr && !type.empty();
    }

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
    Tensor<T> eye(std::vector<int> shape) {
        Tensor<T> tensor(shape, 0);
        int n = *std::min_element(shape.begin(), shape.end());
        for (int i = 0; i < n; ++i) {
            std::vector<int> index(shape.size(), i);
            tensor.data[indexInFlatArray(index, shape)] = 1;
        }
        return tensor;
    }
    // ======= 1. Creation and Initialization End =======

    // ======= 2. Tensor Operations =======

    // ------- 2.1 Indexing and Slicing -------

    /**
     * @brief Get the element at the given index
     * @tparam T
     * @param index
     * @return The element at the given index
     */
    template<typename T>
    Tensor<T> Tensor<T>::operator()(int index) {
        assert(*this);

        std::vector<int> new_shape = shape;
        new_shape.erase(new_shape.begin());
        Tensor<T> tensor(new_shape, data + index * totalSize(new_shape));
        return tensor;
    }

    /**
     * @brief Get the slice of the tensor at the given index and indices
     * @tparam T
     * @param index The chosen index of the tensor to be sliced
     * @param indices The start (included) and end (excluded) indices of the slice
     * @return The slice of the tensor at the given index and indices
     */
    template<typename T>
    Tensor<T> Tensor<T>::operator()(int index, std::vector<int> indices) {
        assert(*this);
        assert(0 <= index && index < shape[0]);
        assert(indices.size() == 2 && 0 <= indices[0] && indices[0] < indices[1] && indices[1] <= shape[1]);

        std::vector<int> new_shape = shape;
        new_shape.erase(new_shape.begin());
        int old_element_count = totalSize(new_shape);
        new_shape[0] = indices[1] - indices[0];

        std::vector<int> temp_shape = new_shape;
        temp_shape.erase(temp_shape.begin());
        int new_element_count = totalSize(temp_shape);

        Tensor<T> tensor(new_shape,
                         data + index * old_element_count + indices[0] * new_element_count);
        return tensor;
    }

    // ------- 2.1 Indexing and Slicing End -------

    // ------- 2.2 Joining -------
    /**
     * @brief Join the given tensors along the given dimension
     * @tparam T
     * @param tensors
     * @param dim
     * @return The joined tensor
     */
    template<typename T>
    Tensor<T> cat(std::vector<Tensor<T>> &tensors, int dim) {
        for (Tensor<T> &tensor: tensors) {
            assert(tensor);
        }
        assert(!tensors.empty() && dim < tensors[0].shape.size());

        // 计算结果张量的形状
        std::vector<int> shape = tensors[0].shape;
        shape[dim] = std::accumulate(tensors.begin(), tensors.end(), 0,
                                     [dim](int sum, const Tensor<T> &tensor) { return sum + tensor.shape[dim]; });

        Tensor<T> result = ts::tensor(shape, (T) 0);

        int offset = 0; // 在连接维度上的偏移量
        for (const auto &tensor: tensors) {
            for (int i = 0; i < tensor.shape[dim]; ++i) {
                for (int j = 0; j < totalSize(tensor.shape) / tensor.shape[dim]; ++j) {
                    std::vector<int> index = tensor.shape;
                    int remaining = j;
                    for (int k = tensor.shape.size() - 1; k >= 0; --k) {
                        if (k != dim) {
                            index[k] = remaining % tensor.shape[k];
                            remaining /= tensor.shape[k];
                        } else {
                            index[k] = i;
                        }
                    }

                    int tensorFlatIndex = indexInFlatArray(index, tensor.shape);
                    index[dim] += offset;
                    int resultFlatIndex = indexInFlatArray(index, shape);
                    result.data[resultFlatIndex] = tensor.data[tensorFlatIndex];
                }
            }
            offset += tensor.shape[dim];
        }

        return result;
    }

    template<typename T>
    Tensor<T> tile(Tensor<T> &tensor, std::vector<int> shape) {
        assert(tensor);
        assert(!shape.empty());

        // 如果 shape 维度多于 tensor 的维度，扩展 tensor 的维度
        while (tensor.shape.size() < shape.size()) {
            tensor.shape.insert(tensor.shape.begin(), 1);
        }

        // 计算新张量的形状
        std::vector<int> new_shape;
        for (size_t i = 0; i < shape.size(); ++i) {
            new_shape.push_back(tensor.shape[i] * shape[i]);
        }

        // 创建新张量
        Tensor<T> tiled_tensor(new_shape, T());

        // 对新张量的每个元素进行复制
        for (int i = 0; i < totalSize(new_shape); ++i) {
            // 计算在原始张量中的索引
            std::vector<int> index = calculateIndex(i, new_shape, tensor.shape, shape);
            int linear_index = linearizeIndex(index, tensor.shape);

            // 复制数据
            tiled_tensor.data[i] = tensor.data[linear_index];
        }

        return tiled_tensor;
    }

    // 辅助函数：将线性索引转换为多维索引
    std::vector<int> calculateIndex(int linear_index, const std::vector<int> &new_shape,
                                    const std::vector<int> &original_shape, const std::vector<int> &shape) {
        std::vector<int> index(original_shape.size(), 0);
        for (int i = original_shape.size() - 1; i >= 0; --i) {
            index[i] = (linear_index /
                        std::accumulate(new_shape.begin() + i + 1, new_shape.end(), 1, std::multiplies<int>())) %
                       original_shape[i];
        }
        return index;
    }

    // 辅助函数：将多维索引转换为线性索引
    int linearizeIndex(const std::vector<int> &index, const std::vector<int> &shape) {
        int linear_index = 0;
        for (size_t i = 0; i < index.size(); ++i) {
            linear_index *= shape[i];
            linear_index += index[i];
        }
        return linear_index;
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
        Tensor<T> result = deepcopy(*this);
        int size = shape.size();
        addKernel<<<(size + 511) / 512, 512>>>(data, other.data, result.data, totalSize(shape));
        cudaDeviceSynchronize();
        return result;
    }


    template<typename T>
    Tensor<T> Tensor<T>::add(T value) {
        Tensor<T> result = deepcopy(*this);
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
        Tensor<T> result = deepcopy(*this);
        int size = shape.size();
        subKernel<<<(size + 511) / 512, 512>>>(data, other.data, result.data, totalSize(shape));
        cudaDeviceSynchronize();
        return result;
    }

    template<typename T>
    Tensor<T> Tensor<T>::sub(T value) {
        Tensor<T> result = deepcopy(*this);
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
        Tensor<T> result = deepcopy(*this);
        int size = shape.size();
        mulKernel<<<(size + 511) / 512, 512>>>(data, other.data, result.data, totalSize(shape));
        cudaDeviceSynchronize();
        return result;
    }

    template<typename T>
    Tensor<T> Tensor<T>::mul(T value) {
        Tensor<T> result = deepcopy(*this);
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
        Tensor<T> result = deepcopy(*this);
        int size = shape.size();
        divKernel<<<(size + 511) / 512, 512>>>(data, other.data, result.data, totalSize(shape));
        cudaDeviceSynchronize();
        return result;
    }

    template<typename T>
    Tensor<T> Tensor<T>::div(T value) {
        Tensor<T> result = deepcopy(*this);
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
        Tensor<T> result = deepcopy(*this);
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

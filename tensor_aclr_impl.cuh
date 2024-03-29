#ifndef TENSOR_ACLR_IMPL_CUH
#define TENSOR_ACLR_IMPL_CUH

#include "tensor_aclr.cuh"
#include <random>
#include <algorithm>
#include <typeinfo>
#include <numeric>
#include <utility>
#include <cassert>
#include <math.h>
#include <sstream>
#include <vector>
#include <string>
#include <cctype>
#include <iostream>
#include <unordered_set>
#include "omp.h"
#include <cereal/archives/binary.hpp>
#include <fstream>
#include <ostream>

namespace ts {
    int totalSize(const std::vector<int> &shape);

    template<typename T>
    void init_stride(Tensor<T> &tensor) {
        tensor.stride.resize(tensor.shape.size());
        int stride = 1;
        for (int i = tensor.shape.size() - 1; i >= 0; --i) {
            tensor.stride[i] = stride;
            stride *= tensor.shape[i];
        }
    }

    // ======= Tensor Constructor =======

    /**
     * @brief Construct a new Tensor object
     * @tparam T
     * @return A new Tensor object
     */
    template<typename T>
    Tensor<T>::Tensor() {}


    /**
     * @brief Construct a new Tensor object, filled with the given value
     * @tparam T
     * @param shape
     * @param defaultValue
     * @return A new Tensor object, filled with the given value
     */
    template<typename T>
    Tensor<T>::Tensor(const std::vector<int> &shape, T defaultValue)
            :shape(shape),
             stride(),
             type(typeid(T).name()) {
        T *data_ptr = new T[totalSize(shape)];
        std::fill(data_ptr, data_ptr + totalSize(shape), defaultValue);
        data = data_ptr;
        parent_data = std::shared_ptr<T>(data_ptr, std::default_delete<T[]>());
        init_stride(*this);
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
            :shape(shape),
             type(typeid(T).name()),
             stride() {
        this->data = data;
        this->parent_data = std::shared_ptr<T>(data, std::default_delete<T[]>());
        init_stride(*this);
    }

    /**
    * @brief Construct a new Tensor object, filled with the given data
    * @tparam T
    * @param shape
    * @param data
    * @param stride
    * @return A new Tensor object, filled with the given data
    */
    template<typename T>
    Tensor<T>::Tensor(const std::vector<int> &shape, const std::vector<int> &stride, T *data,
                      std::shared_ptr<T> parent_data)
            : shape(shape),
              type(typeid(T).name()),
              stride(stride) {
        this->data = data;
        this->parent_data = parent_data;
    }

    /**
     * @brief Construct a new Tensor object from another Tensor object
     * @tparam T
     * @param other
     * @return A new Tensor object from another Tensor object
     */
    template<typename T>
    Tensor<T>::Tensor(const Tensor<T> &other)
            : shape(other.shape),
              type(other.type),
              stride(other.stride) {
        this->data = other.data;
        this->parent_data = other.parent_data;
    }

    // ======= Class Constructor End =======


    // ======= 0. Helper Functions =======

    // ------- Random Part -------
    const double MIN_RANDOM = 1, MAX_RANDOM = 5000;
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
        std::string str;
        printTensor(str, tensor, 0, 0, "");
        if (str.size() > 3000) {
            int start = str.rfind('\n', 400);
            int end = str.find('\n', str.size() - 400);
            if (start == std::string::npos || end == std::string::npos) {
                start = 400;
                end = str.size() - 400;
            }
            std::string startPart = str.substr(0, start);
            std::string endPart = str.substr(end);
            str = startPart + "\n  ............\n" + endPart;
        }
        os << str;
        return os;
    }

    // Helper function to print tensor
    template<typename T>
    void printTensor(std::string &str, const Tensor<T> &tensor, int index, int dimension, const std::string &indent) {
        if (dimension == tensor.shape.size()) {
            str += std::to_string(tensor.data[index]);
            return;
        }

        str += "[";
        int dimSize = tensor.shape[dimension];
//        int nextIndexStep = (dimension < tensor.shape.size() - 1)
//                            ? totalSize(std::vector<int>(tensor.shape.begin() + dimension + 1, tensor.shape.end()))
//                            : 1;
        int nextIndexStep = tensor.stride[dimension];
        for (int i = 0; i < dimSize; ++i) {
            if (i > 0) {
                str += ", ";
                if (dimension != tensor.shape.size() - 1) {
                    str += "\n" + indent + std::string(dimension + 1, ' ');
                }
            }
            printTensor(str, tensor, index + i * nextIndexStep, dimension + 1, indent + " ");
        }
        str += "]";
    }

    // Helper function to get shape of tensor
    template<typename T>
    void Tensor<T>::printShape() {
        std::cout << "[ ";
        for (int i: shape) {
            std::cout << i << " ";
        }
        std::cout << "]" << std::endl;
    }


    // Helper function to get data
    template<typename T>
    void getData(std::vector<T> &V, const Tensor<T> &tensor, int index, int dimension) {
        if (dimension == tensor.shape.size()) {
            V.push_back(tensor.data[index]);
            return;
        }
        int dimSize = tensor.shape[dimension];
        int nextIndexStep = tensor.stride[dimension];
        for (int i = 0; i < dimSize; ++i) {
            getData(V, tensor, index + i * nextIndexStep, dimension + 1);
        }
    }

    template<typename T>
    T getT(const Tensor<T> &tensor) {
        T initial_type;
        if (tensor.type == typeid(int).name()) {
            initial_type = 0;
        } else if (tensor.type == typeid(double).name() || tensor.type == typeid(float).name()) {
            initial_type = 0.0f;
        } else if (tensor.type == typeid(bool).name()) {
            initial_type = false;
        }
        return initial_type;
    }

    // Helper function to construct a new tensor with given data vector
    template<typename T>
    Tensor<T> newTensor(const Tensor<T> &tensor, const std::vector<int> &shape) {
        ts::Tensor new_tensor(shape, getT(tensor));
        std::vector<T> V;
        getData(V, tensor, 0, 0);
        for (int i = 0; i < V.size(); i++) {
            new_tensor.data[i] = V[i];
        }
        return new_tensor;
    }


    // deep copy
    /**
     * @brief Deep copy
     * @tparam T
     * @param other
     * @return A deep copy of the given tensor
     */
    template<typename T>
    Tensor<T> deepcopy(const Tensor<T> other) {
        Tensor<T> result = newTensor(other, other.shape);
        return result;
    }

    template<typename T>
    Tensor<T>::operator bool() const {
        return !shape.empty() && data != nullptr && !type.empty();
    }

    //get a value from data
    template<typename T>
    T &get_value(Tensor<T> &t, const std::vector<int> &inds) {
        int offset = 0;
        for (int i = 0; i < t.shape.size(); ++i)
            offset += inds[i] * t.stride[i];
        return t.data[offset];
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
        T *data = new T[totalSize(shape)];
        for (int i = 0; i < totalSize(shape); ++i) {
            data[i] = (T) random();
        }
        tensor.data = data;
        tensor.parent_data = std::shared_ptr<T>(data, std::default_delete<T[]>());
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
        return Tensor<T>(shape, (T) 0);
    }

    /**
     * @brief Create a tensor of given shape and fill it with 1
     * @tparam T
     * @param shape
     * @return A tensor of given shape and filled with 1
     */
    template<typename T>
    Tensor<T> ones(std::vector<int> shape) {
        return Tensor<T>(shape, (T) 1);
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
            tensor.data[indexInFlatArray(index, shape)] = (T) 1;
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
        return this->operator()(index, 0);
    }

    /**
     * @brief Get the element at the given index from the given dimension
     * @tparam T
     * @param index
     * @param dim
     * @return The element at the given index and dimension
     */
    template<typename T>
    Tensor<T> Tensor<T>::operator()(int index, int dim) {
        assert(*this);
        std::vector<int> new_shape = shape;
        new_shape.erase(new_shape.begin() + dim);
        if (new_shape.empty()) new_shape.push_back(1);
        std::vector<int> new_stride = stride;
        new_stride.erase(new_stride.begin() + dim);
        if (new_stride.empty()) new_stride.push_back(1);
        Tensor<T> tensor(new_shape, new_stride, data + index * stride[dim], parent_data);
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
        return this->operator()(index, indices, 0);
    }

    /**
     * @brief Get the slice of the tensor at the given index and indices and dimension
     * @tparam T
     * @param index The chosen index of the tensor to be sliced
     * @param indices The start (included) and end (excluded) indices of the slice
     * @param dim
     * @return The slice of the tensor at the given index and indices
     */
    template<typename T>
    Tensor<T> Tensor<T>::operator()(int index, std::vector<int> indices, int dim) {
        assert(*this);
        assert(0 <= index && index < shape[dim]);
        assert(indices.size() == 2 && 0 <= indices[0] && indices[0] < indices[1] && indices[1] <= shape[1]);
        Tensor<T> tensor0 = this->operator()(index, dim);
        std::vector<int> new_shape = tensor0.shape;
        new_shape[0] = indices[1] - indices[0];
        Tensor<T> tensor(new_shape, tensor0.stride,
                         tensor0.data + indices[0] * tensor0.stride[0],
                         parent_data);
        return tensor;
    }


    // ------- 2.1 Indexing and Slicing End -------

    // ------- 2.2 Joining -------
    template<typename T>
    Tensor<T> cat(std::vector<Tensor<T>> &tensors, int dim) {
        // 检查维度一致性
        if (tensors.empty()) {
            throw std::invalid_argument("No tensors provided for concatenation.");
        }
        int num_dims = tensors[0].shape.size();
        for (const auto &tensor: tensors) {
            if (tensor.shape.size() != num_dims) {
                throw std::invalid_argument("All tensors must have the same number of dimensions.");
            }
        }

        // 计算新张量的形状
        std::vector<int> new_shape = tensors[0].shape;
        for (size_t i = 1; i < tensors.size(); ++i) {
            new_shape[dim] += tensors[i].shape[dim];
        }

        // 创建新张量并初始化为0
        Tensor<T> result = ts::zeros<T>(new_shape);

        // 拼接数据
        int offset = 0;
        for (const auto &tensor: tensors) {
            int tensor_dim_size = tensor.shape[dim];
            for (int i = 0; i < tensor_dim_size; ++i) {
                for (int j = 0; j < totalSize(tensor.shape) / tensor_dim_size; ++j) {
                    // 计算在原张量中的线性索引
                    std::vector<int> src_indices(num_dims, 0);
                    src_indices[dim] = i;
                    int src_linear_index = 0;
                    for (int k = 0; k < num_dims; ++k) {
                        int stride_index = j;
                        if (k != dim) {
                            stride_index %= tensor.shape[k];
                        } else {
                            stride_index = i;
                        }
                        src_linear_index += stride_index * tensor.stride[k];
                    }

                    // 计算在新张量中的线性索引
                    std::vector<int> dest_indices = src_indices;
                    dest_indices[dim] += offset;
                    int dest_linear_index = 0;
                    for (int k = 0; k < num_dims; ++k) {
                        int stride_index = j;
                        if (k != dim) {
                            stride_index %= new_shape[k];
                        } else {
                            stride_index = offset + i;
                        }
                        dest_linear_index += stride_index * result.stride[k];
                    }

                    // 复制数据
                    result.data[dest_linear_index] = tensor.data[src_linear_index];
                }
            }
            offset += tensor_dim_size;
        }

        return result;
    }

    std::vector<int> calculate_indices(int linear_index, const std::vector<int> &shape) {
        std::vector<int> indices(shape.size());
        for (int i = shape.size() - 1; i >= 0; --i) {
            indices[i] = linear_index % shape[i];
            linear_index /= shape[i];
        }
        return indices;
    }

    template<typename T>
    Tensor<T> tile(Tensor<T> &tensor, std::vector<int> shape) {
        // 检查维度一致性
        if (tensor.shape.size() != shape.size()) {
            throw std::invalid_argument("Shape of tensor and tile shape must have the same number of dimensions.");
        }

        // 创建新张量并初始化为0
        std::vector<int> new_shape;
        for (size_t i = 0; i < shape.size(); ++i) {
            new_shape.push_back(tensor.shape[i] * shape[i]);
        }
        Tensor<T> result = ts::zeros<T>(new_shape);

        // Tile 数据
        for (int i = 0; i < totalSize(new_shape); ++i) {
            std::vector<int> indices = calculate_indices(i, new_shape);
            std::vector<int> orig_indices(indices.size());
            for (size_t j = 0; j < indices.size(); ++j) {
                orig_indices[j] = indices[j] % tensor.shape[j];
            }
            int orig_linear_index = 0;
            for (size_t k = 0; k < orig_indices.size(); ++k) {
                orig_linear_index += orig_indices[k] * tensor.stride[k];
            }

            result.data[i] = tensor.data[orig_linear_index];
        }

        return result;
    }



    // ------- 2.2 Joining End -------

    // ------- 2.3 Mutating -------
//    template<typename T>
//    Tensor<T> &Tensor<T>::operator=(T value) {
//        assert(*this && this->shape.size() == 1);
//
//        int element_count = totalSize(shape);
//        std::fill(data, data + element_count, value);
//        return *this;
//    }
//
//    template<typename T>
//    Tensor<T> &Tensor<T>::operator=(std::initializer_list<T> values) {
//        assert(*this);
//        assert(values.size() == shape[0]);
//
//        std::vector<int> new_shape = shape;
//        new_shape.erase(new_shape.begin());
//        int element_count = totalSize(new_shape);
//        for (int i = 0; i < values.size(); ++i) {
//            std::fill(data + i * element_count, data + (i + 1) * element_count, *(values.begin() + i));
//        }
//        return *this;
//    }



    template<typename T>
    Tensor<T> &Tensor<T>::operator=(T value) {
        assert(!this->shape.empty());

        int element_count = std::accumulate(this->shape.begin(), this->shape.end(), 1, std::multiplies<int>());
        for (int i = 0; i < element_count; ++i) {
            std::vector<int> indices = calculate_indices(i, this->shape);
            get_value(*this, indices) = value;
        }
        return *this;
    }

    template<typename T>
    Tensor<T> &Tensor<T>::operator=(std::initializer_list<T> values) {
        assert(!this->shape.empty());
        assert(values.size() == this->shape[0]);

        std::vector<int> new_shape = this->shape;
        new_shape.erase(new_shape.begin());
        int element_count = std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<int>());
        auto it = values.begin();
        for (int i = 0; i < values.size(); ++i, ++it) {
            for (int j = 0; j < element_count; ++j) {
                std::vector<int> indices = calculate_indices(i * element_count + j, this->shape);
                get_value(*this, indices) = *it;
            }
        }
        return *this;
    }

    // ------- 2.4 Permutation -------
    // transpose
    template<typename T>
    Tensor<T> Tensor<T>::transpose(int dim0, int dim1) {
        std::swap(this->shape[dim0], this->shape[dim1]);
        std::swap(this->stride[dim0], this->stride[dim1]);
        return *this;
    }

    //permute
    template<typename T>
    Tensor<T> Tensor<T>::permute(const std::vector<int> &dims) {
        std::vector<int> origin_shape = shape;
        std::vector<int> origin_stride = stride;
        for (int i = 0; i < this->shape.size(); ++i) {
            this->shape[i] = origin_shape[dims[i]];
            this->stride[i] = origin_stride[dims[i]];
        }
        return *this;
    }

    template<typename T>
    Tensor<T> permute(Tensor<T> &tensor, const std::vector<int> &dims) {
        return tensor.permute(dims);
    }

    template<typename T>
    Tensor<T> transpose(Tensor<T> &tensor, int dim0, int dim1) {
        return tensor.transpose(dim0, dim1);
    }

    // ------- 2.5 View -------
    // 是否经历转置导致data不连续
    template<typename T>
    bool is_contiguous(const Tensor<T> &t) {
        int stride = 1;
        for (int i = t.shape.size() - 1; i >= 0; --i) {
            if (t.stride[i] != stride)
                return false;
            stride *= t.shape[i];
        }
        return true;
    }

    template<typename T>
    Tensor<T> Tensor<T>::view(const std::vector<int> &shape0) {
        if (!is_contiguous(*this)) {
            throw std::invalid_argument("input tensor is not contiguous");
        }
        int stride0 = 1;
        for (int i = shape.size() - 1; i >= 0; --i) {
            this->shape[i] = shape0[i];
            this->stride[i] = stride0;
            stride0 *= shape0[i];
        }
        return *this;
    }

    template<typename T>
    Tensor<T> view(Tensor<T> &tensor, const std::vector<int> &shape0) {
        return tensor.view(shape0);
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
    __global__ void addKernel(const T *data1, const T *data2, T *data3, int size) {
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
            result[index] = std::log((double) data[index]);
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
        std::vector<T> Va, Vb;
        getData(Va, *this, 0, 0);
        getData(Vb, other, 0, 0);
        if (acceleration) {
            size_t data_size = totalSize(shape) * sizeof(T);
            T *a;
            T *b;
            T *c;
            cudaMallocManaged(&a, data_size);
            cudaMallocManaged(&b, data_size);
            cudaMallocManaged(&c, data_size);
            for (size_t q = 0; q < Va.size(); q++) {
                a[q] = Va[q];
                b[q] = Vb[q];
            }
            addKernel<<<(totalSize(shape) + 511) / 512, 512>>>(a, b, c, totalSize(shape));
            cudaDeviceSynchronize();
            cudaMemcpy(result.data, c, data_size, cudaMemcpyDeviceToHost);
            cudaFree(a);
            cudaFree(b);
            cudaFree(c);
        } else {
            for (int i = 0; i < totalSize(shape); i++) {
                result.data[i] = Va[i] + Vb[i];
            }
        }
        return result;
    }


    template<typename T>
    Tensor<T> Tensor<T>::add(T value) {
        Tensor<T> result = deepcopy(*this);
        std::vector<T> Va;
        getData(Va, *this, 0, 0);
        if (acceleration) {
            T *a;
            T *c;
            size_t data_size = totalSize(shape) * sizeof(T);
            cudaMallocManaged(&a, data_size);
            cudaMallocManaged(&c, data_size);
            for (size_t q = 0; q < Va.size(); q++) {
                a[q] = Va[q];
            }
            addScalarKernel<<<(totalSize(shape) + 511) / 512, 512>>>(a, value, c, totalSize(shape));
            cudaDeviceSynchronize();
            cudaMemcpy(result.data, c, data_size, cudaMemcpyDeviceToHost);
            cudaFree(a);
            cudaFree(c);
        } else {
            for (int i = 0; i < totalSize(shape); i++) {
                result.data[i] = Va[i] + value;
            }
        }
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
        std::vector<T> Va, Vb;
        getData(Va, *this, 0, 0);
        getData(Vb, other, 0, 0);
        if (acceleration) {
            size_t data_size = totalSize(shape) * sizeof(T);
            T *a;
            T *b;
            T *c;
            cudaMallocManaged(&a, data_size);
            cudaMallocManaged(&b, data_size);
            cudaMallocManaged(&c, data_size);
            for (size_t q = 0; q < Va.size(); q++) {
                a[q] = Va[q];
                b[q] = Vb[q];
            }
            subKernel<<<(totalSize(shape) + 511) / 512, 512>>>(a, b, c, totalSize(shape));
            cudaDeviceSynchronize();
            cudaMemcpy(result.data, c, data_size, cudaMemcpyDeviceToHost);
            cudaFree(a);
            cudaFree(b);
            cudaFree(c);
        } else {
            for (int i = 0; i < totalSize(shape); i++) {
                result.data[i] = Va[i] - Vb[i];
            }
        }
        return result;
    }

    template<typename T>
    Tensor<T> Tensor<T>::sub(T value) {
        Tensor<T> result = deepcopy(*this);
        std::vector<T> Va;
        getData(Va, *this, 0, 0);
        if (acceleration) {
            T *a;
            T *c;
            size_t data_size = totalSize(shape) * sizeof(T);
            cudaMallocManaged(&a, data_size);
            cudaMallocManaged(&c, data_size);
            for (size_t q = 0; q < Va.size(); q++) {
                a[q] = Va[q];
            }
            subScalarKernel<<<(totalSize(shape) + 511) / 512, 512>>>(a, value, c, totalSize(shape));
            cudaDeviceSynchronize();
            cudaMemcpy(result.data, c, data_size, cudaMemcpyDeviceToHost);
            cudaFree(a);
            cudaFree(c);
        } else {
            for (int i = 0; i < totalSize(shape); i++) {
                result.data[i] = Va[i] - value;
            }
        }
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
        std::vector<T> Va, Vb;
        getData(Va, *this, 0, 0);
        getData(Vb, other, 0, 0);
        if (acceleration) {
            size_t data_size = totalSize(shape) * sizeof(T);
            T *a;
            T *b;
            T *c;
            cudaMallocManaged(&a, data_size);
            cudaMallocManaged(&b, data_size);
            cudaMallocManaged(&c, data_size);
            for (size_t q = 0; q < Va.size(); q++) {
                a[q] = Va[q];
                b[q] = Vb[q];
            }
            mulKernel<<<(totalSize(shape) + 511) / 512, 512>>>(a, b, c, totalSize(shape));
            cudaDeviceSynchronize();
            cudaMemcpy(result.data, c, data_size, cudaMemcpyDeviceToHost);
            cudaFree(a);
            cudaFree(b);
            cudaFree(c);
        } else {
            for (int i = 0; i < totalSize(shape); i++) {
                result.data[i] = Va[i] * Vb[i];
            }
        }
        return result;
    }

    template<typename T>
    Tensor<T> Tensor<T>::mul(T value) {
        Tensor<T> result = deepcopy(*this);
        std::vector<T> Va;
        getData(Va, *this, 0, 0);
        if (acceleration) {
            T *a;
            T *c;
            size_t data_size = totalSize(shape) * sizeof(T);
            cudaMallocManaged(&a, data_size);
            cudaMallocManaged(&c, data_size);
            for (size_t q = 0; q < Va.size(); q++) {
                a[q] = Va[q];
            }
            mulScalarKernel<<<(totalSize(shape) + 511) / 512, 512>>>(a, value, c, totalSize(shape));
            cudaDeviceSynchronize();
            cudaMemcpy(result.data, c, data_size, cudaMemcpyDeviceToHost);
            cudaFree(a);
            cudaFree(c);
        } else {
            for (int i = 0; i < totalSize(shape); i++) {
                result.data[i] = Va[i] * value;
            }
        }
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
        std::vector<T> Va, Vb;
        getData(Va, *this, 0, 0);
        getData(Vb, other, 0, 0);
        if (acceleration) {
            size_t data_size = totalSize(shape) * sizeof(T);
            T *a;
            T *b;
            T *c;
            cudaMallocManaged(&a, data_size);
            cudaMallocManaged(&b, data_size);
            cudaMallocManaged(&c, data_size);
            for (size_t q = 0; q < Va.size(); q++) {
                a[q] = Va[q];
                b[q] = Vb[q];
            }
            divKernel<<<(totalSize(shape) + 511) / 512, 512>>>(a, b, c, totalSize(shape));
            cudaDeviceSynchronize();
            cudaMemcpy(result.data, c, data_size, cudaMemcpyDeviceToHost);
            cudaFree(a);
            cudaFree(b);
            cudaFree(c);
        } else {
            for (int i = 0; i < totalSize(shape); i++) {
                result.data[i] = Va[i] / Vb[i];
            }
        }
        return result;
    }

    template<typename T>
    Tensor<T> Tensor<T>::div(T value) {
        Tensor<T> result = deepcopy(*this);
        std::vector<T> Va;
        getData(Va, *this, 0, 0);
        if (acceleration) {
            T *a;
            T *c;
            size_t data_size = totalSize(shape) * sizeof(T);
            cudaMallocManaged(&a, data_size);
            cudaMallocManaged(&c, data_size);
            for (size_t q = 0; q < Va.size(); q++) {
                a[q] = Va[q];
            }
            divScalarKernel<<<(totalSize(shape) + 511) / 512, 512>>>(a, value, c, totalSize(shape));
            cudaDeviceSynchronize();
            cudaMemcpy(result.data, c, data_size, cudaMemcpyDeviceToHost);
            cudaFree(a);
            cudaFree(c);
        } else {
            for (int i = 0; i < totalSize(shape); i++) {
                result.data[i] = Va[i] / value;
            }
        }
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
        std::vector<T> Va;
        getData(Va, *this, 0, 0);
        if (acceleration) {
            T *a;
            T *c;
            size_t data_size = totalSize(shape) * sizeof(T);
            cudaMallocManaged(&a, data_size);
            cudaMallocManaged(&c, data_size);
            for (size_t q = 0; q < Va.size(); q++) {
                a[q] = Va[q];
            }
            logKernel<<<(totalSize(shape) + 511) / 512, 512>>>(a, c, totalSize(shape));
            cudaDeviceSynchronize();
            cudaMemcpy(result.data, c, data_size, cudaMemcpyDeviceToHost);
            cudaFree(a);
            cudaFree(c);
        } else {
            for (int i = 0; i < totalSize(shape); i++) {
                result.data[i] = std::log((double) Va[i]);
            }
        }
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


    // ======= 3.2 Reduction operations =======

    // ======= 3.2.1 Sum ======
    template<typename T>
    Tensor<T> sum(Tensor<T> &tensor, int dim) {
        assert(dim <= tensor.shape.size());
        Tensor<T> result(tensor(0, dim).shape, getT(tensor));
        if (acceleration) {
            omp_set_num_threads(4);
            for (int i = 0; i < tensor.shape[dim]; i++) {
                Tensor<T> tmp(tensor(0, dim).shape, getT(tensor));
                tmp = newTensor(tensor(i, dim), tensor(i, dim).shape);
//#pragma omp parallel for reduction(+:result.data[:totalSize(result.shape)])
#pragma omp parallel
                {
                    for (int j = 0; j < totalSize(result.shape); j++) {
                        result.data[j] += tmp.data[j];
                    }
                }
            }
        } else {
            for (int i = 0; i < tensor.shape[dim]; i++) {
                Tensor<T> tmp(tensor(0, dim).shape, getT(tensor));
                tmp = newTensor(tensor(i, dim), tensor(i, dim).shape);
                for (int j = 0; j < totalSize(result.shape); j++) {
                    result.data[j] += tmp.data[j];
                }
            }
        }
        return result;
    }

    template<typename T>
    Tensor<T> Tensor<T>::sum(int dim) {
        return ts::sum(*this, dim);
    }

    // ======= 3.2.1 Sum End======

    // ======= 3.2.2 Mean ======
    template<typename T>
    Tensor<T> mean(Tensor<T> &tensor, int dim) {
        assert(dim <= tensor.shape.size());
        Tensor<T> result(tensor(0, dim).shape, getT(tensor));
        if (acceleration == true) {
            omp_set_num_threads(4);
            for (int i = 0; i < tensor.shape[dim]; i++) {
                Tensor<T> tmp(tensor(0, dim).shape, getT(tensor));
                tmp = newTensor(tensor(i, dim), tensor(i, dim).shape);
#pragma omp parallel
                {
                    for (int j = 0; j < totalSize(result.shape); j++) {
                        result.data[j] += tmp.data[j];
                    }
                }
            }
        } else {
            for (int i = 0; i < tensor.shape[dim]; i++) {
                Tensor<T> tmp(tensor(0, dim).shape, getT(tensor));
                tmp = newTensor(tensor(i, dim), tensor(i, dim).shape);
                for (int j = 0; j < totalSize(result.shape); j++) {
                    result.data[j] += tmp.data[j];
                }
            }
        }

        for (int i = 0; i < totalSize(result.shape); i++) {
            result.data[i] = result.data[i] / tensor.shape[dim];
        }
        return result;
    }

    template<typename T>
    Tensor<T> Tensor<T>::mean(int dim) {
        return ts::mean(*this, dim);
    }

    // ======= 3.2.2 Mean End======

    // ======= 3.2.3 Max ======
    template<typename T>
    Tensor<T> max(Tensor<T> &tensor1, Tensor<T> &tensor2) {
        assert(tensor1.shape == tensor2.shape);
        ts::Tensor result = ts::tensor(tensor1.shape, new T[totalSize(tensor1.shape)]);
        int size = totalSize(tensor1.shape);
        if (acceleration) {
            omp_set_num_threads(4);
#pragma omp parallel
            {
                for (int i = 0; i < size; i++) {
                    result.data[i] = std::max(tensor1.data[i], tensor2.data[i]);
                }
            }
        } else {
            for (int i = 0; i < size; i++) {
                result.data[i] = std::max(tensor1.data[i], tensor2.data[i]);
            }
        }
        return result;
    }

    template<typename T>
    Tensor<T> max(Tensor<T> &tensor, int dim) {
        assert(dim <= tensor.shape.size());
        Tensor<T> result(tensor(0, dim).shape, std::numeric_limits<T>::min());
        if (acceleration) {
            omp_set_num_threads(4);
#pragma omp parallel
            {
                for (int i = 0; i < tensor.shape[dim]; i++) {
                    Tensor<T> tmp(tensor(0, dim).shape, getT(tensor));
                    tmp = newTensor(tensor(i, dim), tensor(i, dim).shape);
                    result = max(result, tmp);
                }
            }
        } else {
            for (int i = 0; i < tensor.shape[dim]; i++) {
                Tensor<T> tmp(tensor(0, dim).shape, getT(tensor));
                tmp = newTensor(tensor(i, dim), tensor(i, dim).shape);
                result = max(result, tmp);
            }
        }
        return result;
    }

    template<typename T>
    Tensor<T> Tensor<T>::max(int dim) {
        return ts::max(*this, dim);
    }

    // ======= 3.2.3 Max End ======

    // ======= 3.2.4 Min ======
    template<typename T>
    Tensor<T> min(Tensor<T> &tensor1, Tensor<T> &tensor2) {
        assert(tensor1.shape == tensor2.shape);
        ts::Tensor result = ts::tensor(tensor1.shape, new T[totalSize(tensor1.shape)]);
        int size = totalSize(tensor1.shape);
        if (acceleration) {
            omp_set_num_threads(4);
#pragma omp parallel
            {
                for (int i = 0; i < size; i++) {
                    result.data[i] = std::min(tensor1.data[i], tensor2.data[i]);
                }
            }
        } else {
            for (int i = 0; i < size; i++) {
                result.data[i] = std::min(tensor1.data[i], tensor2.data[i]);
            }
        }
        return result;
    }

    template<typename T>
    Tensor<T> min(Tensor<T> &tensor, int dim) {
        assert(dim <= tensor.shape.size());
        Tensor<T> result(tensor(0, dim).shape, std::numeric_limits<T>::max());
        if (acceleration) {
            omp_set_num_threads(4);
#pragma omp parallel
            {
                for (int i = 0; i < tensor.shape[dim]; i++) {
                    Tensor<T> tmp(tensor(0, dim).shape, getT(tensor));
                    tmp = newTensor(tensor(i, dim), tensor(i, dim).shape);
                    result = min(result, tmp);
                }
            }
        } else {
            for (int i = 0; i < tensor.shape[dim]; i++) {
                Tensor<T> tmp(tensor(0, dim).shape, getT(tensor));
                tmp = newTensor(tensor(i, dim), tensor(i, dim).shape);
                result = min(result, tmp);
            }
        }
        return result;
    }

    template<typename T>
    Tensor<T> Tensor<T>::min(int dim) {
        return ts::min(*this, dim);
    }

    // ======= 3.2.4 Min End ======

    // ======= 3.2 Reduction operations End =======

    // ======= 3.3 Comparison operations =======

    template<typename T>
    Tensor<bool> Tensor<T>::eq(const Tensor<T> &other) {
        assert(data);
        assert(shape == other.shape);
        assert(type == other.type);
        Tensor<bool> result(shape, false);
        std::vector<T> Va, Vb;
        getData(Va, *this, 0, 0);
        getData(Vb, other, 0, 0);
        if (acceleration) {
            T *a;
            T *b;
            bool *c;
            size_t data_size = totalSize(shape) * sizeof(T);
            cudaMallocManaged(&a, data_size);
            cudaMallocManaged(&b, data_size);
            cudaMallocManaged(&c, totalSize(shape) * sizeof(bool));
            for (size_t q = 0; q < Va.size(); q++) {
                a[q] = Va[q];
                b[q] = Vb[q];
            }
            eqKernel<<<(totalSize(shape) + 511) / 512, 512>>>(a, b, c, totalSize(shape));
            cudaDeviceSynchronize();
            cudaMemcpy(result.data, c, totalSize(shape) * sizeof(bool), cudaMemcpyDeviceToHost);
            cudaFree(a);
            cudaFree(b);
            cudaFree(c);
        } else {
            for (int i = 0; i < totalSize(shape); i++) {
                result.data[i] = Va[i] == Vb[i];
            }
        }
        return result;
    }

    template<typename T>
    Tensor<bool> Tensor<T>::ne(const Tensor<T> &other) {
        assert(data);
        assert(shape == other.shape);
        assert(type == other.type);
        Tensor<bool> result(shape, false);
        std::vector<T> Va, Vb;
        getData(Va, *this, 0, 0);
        getData(Vb, other, 0, 0);
        if (acceleration) {
            T *a;
            T *b;
            bool *c;
            size_t data_size = totalSize(shape) * sizeof(T);
            cudaMallocManaged(&a, data_size);
            cudaMallocManaged(&b, data_size);
            cudaMallocManaged(&c, totalSize(shape) * sizeof(bool));
            for (size_t q = 0; q < Va.size(); q++) {
                a[q] = Va[q];
                b[q] = Vb[q];
            }
            neKernel<<<(totalSize(shape) + 511) / 512, 512>>>(a, b, c, totalSize(shape));
            cudaDeviceSynchronize();
            cudaMemcpy(result.data, c, totalSize(shape) * sizeof(bool), cudaMemcpyDeviceToHost);
            cudaFree(a);
            cudaFree(b);
            cudaFree(c);
        } else {
            for (int i = 0; i < totalSize(shape); i++) {
                result.data[i] = Va[i] != Vb[i];
            }
        }
        return result;
    }

    template<typename T>
    Tensor<bool> Tensor<T>::gt(const Tensor<T> &other) {
        assert(data);
        assert(shape == other.shape);
        assert(type == other.type);
        Tensor<bool> result(shape, false);
        std::vector<T> Va, Vb;
        getData(Va, *this, 0, 0);
        getData(Vb, other, 0, 0);
        if (acceleration) {
            T *a;
            T *b;
            bool *c;
            size_t data_size = totalSize(shape) * sizeof(T);
            cudaMallocManaged(&a, data_size);
            cudaMallocManaged(&b, data_size);
            cudaMallocManaged(&c, totalSize(shape) * sizeof(bool));
            for (size_t q = 0; q < Va.size(); q++) {
                a[q] = Va[q];
                b[q] = Vb[q];
            }
            gtKernel<<<(totalSize(shape) + 511) / 512, 512>>>(a, b, c, totalSize(shape));
            cudaDeviceSynchronize();
            cudaMemcpy(result.data, c, totalSize(shape) * sizeof(bool), cudaMemcpyDeviceToHost);
            cudaFree(a);
            cudaFree(b);
            cudaFree(c);
        } else {
            for (int i = 0; i < totalSize(shape); i++) {
                result.data[i] = Va[i] > Vb[i];
            }
        }
        return result;
    }

    template<typename T>
    Tensor<bool> Tensor<T>::ge(const Tensor<T> &other) {
        assert(data);
        assert(shape == other.shape);
        assert(type == other.type);
        Tensor<bool> result(shape, false);
        std::vector<T> Va, Vb;
        getData(Va, *this, 0, 0);
        getData(Vb, other, 0, 0);
        if (acceleration) {
            T *a;
            T *b;
            bool *c;
            size_t data_size = totalSize(shape) * sizeof(T);
            cudaMallocManaged(&a, data_size);
            cudaMallocManaged(&b, data_size);
            cudaMallocManaged(&c, totalSize(shape) * sizeof(bool));
            for (size_t q = 0; q < Va.size(); q++) {
                a[q] = Va[q];
                b[q] = Vb[q];
            }
            geKernel<<<(totalSize(shape) + 511) / 512, 512>>>(a, b, c, totalSize(shape));
            cudaDeviceSynchronize();
            cudaMemcpy(result.data, c, totalSize(shape) * sizeof(bool), cudaMemcpyDeviceToHost);
            cudaFree(a);
            cudaFree(b);
            cudaFree(c);
        } else {
            for (int i = 0; i < totalSize(shape); i++) {
                result.data[i] = Va[i] >= Vb[i];
            }
        }
        return result;
    }

    template<typename T>
    Tensor<bool> Tensor<T>::lt(const Tensor<T> &other) {
        assert(data);
        assert(shape == other.shape);
        assert(type == other.type);
        Tensor<bool> result(shape, false);
        std::vector<T> Va, Vb;
        getData(Va, *this, 0, 0);
        getData(Vb, other, 0, 0);
        if (acceleration) {
            T *a;
            T *b;
            bool *c;
            size_t data_size = totalSize(shape) * sizeof(T);
            cudaMallocManaged(&a, data_size);
            cudaMallocManaged(&b, data_size);
            cudaMallocManaged(&c, totalSize(shape) * sizeof(bool));
            for (size_t q = 0; q < Va.size(); q++) {
                a[q] = Va[q];
                b[q] = Vb[q];
            }
            ltKernel<<<(totalSize(shape) + 511) / 512, 512>>>(a, b, c, totalSize(shape));
            cudaDeviceSynchronize();
            cudaMemcpy(result.data, c, totalSize(shape) * sizeof(bool), cudaMemcpyDeviceToHost);
            cudaFree(a);
            cudaFree(b);
            cudaFree(c);
        } else {
            for (int i = 0; i < totalSize(shape); i++) {
                result.data[i] = Va[i] < Vb[i];
            }
        }
        return result;
    }

    template<typename T>
    Tensor<bool> Tensor<T>::le(const Tensor<T> &other) {
        assert(data);
        assert(shape == other.shape);
        assert(type == other.type);
        Tensor<bool> result(shape, false);
        std::vector<T> Va, Vb;
        getData(Va, *this, 0, 0);
        getData(Vb, other, 0, 0);
        if (acceleration) {
            T *a;
            T *b;
            bool *c;
            size_t data_size = totalSize(shape) * sizeof(T);
            cudaMallocManaged(&a, data_size);
            cudaMallocManaged(&b, data_size);
            cudaMallocManaged(&c, totalSize(shape) * sizeof(bool));
            for (size_t q = 0; q < Va.size(); q++) {
                a[q] = Va[q];
                b[q] = Vb[q];
            }
            leKernel<<<(totalSize(shape) + 511) / 512, 512>>>(a, b, c, totalSize(shape));
            cudaDeviceSynchronize();
            cudaMemcpy(result.data, c, totalSize(shape) * sizeof(bool), cudaMemcpyDeviceToHost);
            cudaFree(a);
            cudaFree(b);
            cudaFree(c);
        } else {
            for (int i = 0; i < totalSize(shape); i++) {
                result.data[i] = Va[i] <= Vb[i];
            }
        }
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

    // ====== EINSUM helper functions ======

    //trace
    template<typename T>
    Tensor<T> Tensor<T>::trace() {
        int shape_size = shape[0];
        for (int i: shape) {
            if (i != shape_size) {
                throw std::invalid_argument("input tensor is not a square tensor");
            }
        }

        T trace = (T) 0;
        std::vector<int> inds;
        inds.reserve(shape.size());
        for (int i = 0; i < shape.size(); i++) {
            inds.push_back(i);
        }

        if (acceleration) {
            omp_set_num_threads(4);
#pragma omp parallel reduction(+:trace)
            {
                for (int t = 0; t < shape_size; t++) {
                    for (int i = 0; i < shape.size(); i++) {
                        inds[i] = t;
                    }
                    trace += get_value(*this, inds);
                }
            }
        } else {
            for (int t = 0; t < shape_size; t++) {
                for (int i = 0; i < shape.size(); i++) {
                    inds[i] = t;
                }
                trace += get_value(*this, inds);
            }
        }

        return tensor({1}, trace);
    }

    // diagonal
    template<typename T>
    Tensor<T> Tensor<T>::diagonal() {
        int shape_size = shape[0];
        for (int i: shape) {
            if (i != shape_size) {
                throw std::invalid_argument("input tensor is not a square tensor");
            }
        }
        T *diagonal = new T[shape_size];
        std::vector<int> inds;
        inds.reserve(shape.size());
        for (int i = 0; i < shape.size(); i++) {
            inds.push_back(i);
        }
        if (acceleration) {
            omp_set_num_threads(4);
            for (int t = 0; t < shape_size; t++) {
#pragma omp parallel
                {
                    for (int i = 0; i < shape.size(); i++) {
                        inds[i] = t;
                    }
                }
                diagonal[t] = get_value(*this, inds);
            }
        } else {
            for (int t = 0; t < shape_size; t++) {
                for (int i = 0; i < shape.size(); i++) {
                    inds[i] = t;
                }
                diagonal[t] = get_value(*this, inds);
            }
        }
        return tensor({shape_size}, diagonal);
    }


    template<typename T>
    __global__ void outerProductKernel(T *a, T *b, T *c, size_t len_a, size_t len_b) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < len_a * len_b) {
            int i = index / len_b;
            int j = index % len_b;
            c[index] = a[i] * b[j];
        }
    }

    // outer product
    template<typename T>
    Tensor<T> outerProduct(Tensor<T> tensor1, Tensor<T> tensor2) {
        assert(tensor1.shape.size() == 1);
        assert(tensor2.shape.size() == 1);
        Tensor<T> result = tensor({tensor1.shape[0], tensor2.shape[0]}, T(0));
        std::vector<T> Va, Vb;
        getData(Va, tensor1, 0, 0);
        getData(Vb, tensor2, 0, 0);
        if (acceleration) {
            T *a;
            T *b;
            T *c;
            cudaMallocManaged(&a, tensor1.shape[0] * sizeof(T));
            cudaMallocManaged(&b, tensor2.shape[0] * sizeof(T));
            cudaMallocManaged(&c, tensor1.shape[0] * tensor2.shape[0] * sizeof(T));
            for (size_t q = 0; q < Va.size(); q++) {
                a[q] = Va[q];
            }
            for (size_t q = 0; q < Vb.size(); q++) {
                b[q] = Vb[q];
            }
            outerProductKernel<<<(tensor1.shape[0] * tensor2.shape[0] + 511) / 512, 512>>>(a, b, c, tensor1.shape[0],
                                                                                           tensor2.shape[0]);
            cudaDeviceSynchronize();
            cudaMemcpy(result.data, c, tensor1.shape[0] * tensor2.shape[0] * sizeof(T), cudaMemcpyDeviceToHost);
            cudaFree(a);
            cudaFree(b);
            cudaFree(c);
        } else {
            for (size_t i = 0; i < tensor1.shape[0]; ++i) {
                for (size_t j = 0; j < tensor2.shape[0]; ++j) {
                    result.data[i * tensor2.shape[0] + j] = Va[i] * Vb[j];
                }
            }
        }

        return result;
    }

    std::pair<std::vector<std::string>, std::string> splitCommand(const std::string &command) {
        std::pair<std::vector<std::string>, std::string> result;
        std::stringstream ss(command);
        std::string item;

        while (std::getline(ss, item, ',')) {
            size_t arrowPos = item.find("->");
            if (arrowPos != std::string::npos) {
                result.second = item.substr(arrowPos + 2);
                item = item.substr(0, arrowPos);
            }
            result.first.push_back(item);
        }

        return result;
    }

    bool isSingleChar(const std::string &str) {
        for (char c: str) {
            if (c != str[0]) {
                return false;
            }
        }
        return true;
    }

    std::vector<int> sort_indexes(const std::string &str) {
        std::vector<int> idx(str.size());
        for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;

        std::sort(idx.begin(), idx.end(),
                  [&str](size_t i1, size_t i2) { return str[i1] < str[i2]; });
        return idx;
    }

    std::vector<int> get_indexes(const std::string &str1, const std::string &str2) {
        std::unordered_map<char, int> char_to_index;
        for (int i = 0; i < str1.size(); ++i) {
            char_to_index[str1[i]] = i;
        }
        std::vector<int> indexes;
        for (char c: str2) {
            if (char_to_index.find(c) == char_to_index.end()) {
                throw std::runtime_error("Character not found in the first string");
            }
            indexes.push_back(char_to_index[c]);
        }
        return indexes;
    }

    bool isAllAlpha(const std::string &str) {
        return std::all_of(str.begin(), str.end(), [](char c) { return ::isalpha(c) || c == '.'; });
    }

    template<typename T>
    __global__ void block_sum_kernel(T *d_out, T *d_in, int size) {
        extern __shared__ T shared_data[];
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < size) {
            shared_data[threadIdx.x] = d_in[idx];
        } else {
            shared_data[threadIdx.x] = 0;
        }
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                shared_data[threadIdx.x] += shared_data[threadIdx.x + s];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) d_out[blockIdx.x] = shared_data[0];
    }

    template<typename T>
    Tensor<T> singleTensorSum(std::vector<T> &data) {
        T sum = 0;
        if (acceleration) {
            T *d_in;
            T *d_out;
            cudaMallocManaged((void **) &d_in, data.size() * sizeof(T));
            cudaMallocManaged((void **) &d_out, sizeof(T));
            cudaMemcpy(d_in, data.data(), data.size() * sizeof(T), cudaMemcpyHostToDevice);
            int blockSize = 1024;
            int numBlocks = (data.size() + blockSize - 1) / blockSize;
            block_sum_kernel<<<numBlocks, blockSize, blockSize * sizeof(T)>>>(d_out, d_in, data.size());
            cudaDeviceSynchronize();
            cudaMemcpy(&sum, d_out, sizeof(T), cudaMemcpyDeviceToHost);
            cudaFree(d_in);
            cudaFree(d_out);
        } else {
            for (T data0: data) {
                sum += data0;
            }
        }
        return tensor({1}, sum);
    }


    template<typename T>
    Tensor<T> checkOrder(std::vector<int> &idx, Tensor<T> tensor) {
        ts::Tensor<T> tensor_new = ts::deepcopy(tensor);
        for (int i = 0; i < idx.size(); i++) {
            for (int j = i; j < idx.size(); j++) {
                if (idx[i] > idx[j]) {
                    tensor_new.transpose(idx[j], idx[i]);
                    int tmp = idx[j];
                    idx[j] = idx[i];
                    idx[i] = tmp;
                }
            }
        }
        return tensor_new;
    }

    template<typename T>
    Tensor<T> matrix_mul(Tensor<T> t1, Tensor<T> t2) {
        assert(t1.shape.size() == 2 && t2.shape.size() == 2);
        assert(t1.shape[1] == t2.shape[0]);
        int resultRows = t1.shape[0];
        int resultCols = t2.shape[1];
        std::vector shape = {resultRows, resultCols};
        Tensor<T> result = ts::tensor(shape, new T[totalSize(shape)]);

        if (acceleration) {
            int num_threads = 4;
            omp_set_num_threads(num_threads);

#pragma omp parallel collapse(2)
            {
                for (int i = 0; i < resultRows; i++) {
                    for (int j = 0; j < resultCols; j++) {
                        result.data[i * resultCols + j] = 0;
                        for (int k = 0; k < t1.shape[1]; k++) {
                            result.data[i * resultCols + j] +=
                                    t1.data[i * t1.shape[1] + k] * t2.data[k * t2.shape[1] + j];
                        }
                    }
                }
            }
        } else {
            for (int i = 0; i < resultRows; i++) {
                for (int j = 0; j < resultCols; j++) {
                    result.data[i * resultCols + j] = 0;
                    for (int k = 0; k < t1.shape[1]; k++) {
                        result.data[i * resultCols + j] += t1.data[i * t1.shape[1] + k] * t2.data[k * t2.shape[1] + j];
                    }
                }
            }
        }

        return result;
    }

    template<typename T>
    Tensor<T> batch_matrix_mul(Tensor<T> t1, Tensor<T> t2) {
        assert(t1.shape.size() == 3 && t2.shape.size() == 3);
        assert(t1.shape[2] == t2.shape[1]);
        std::vector<int> shape{t1.shape[0], t1.shape[1], t2.shape[2]};
        Tensor<T> result = ts::tensor(shape, new T[totalSize(shape)]);
        int idx = 0;
        if (acceleration) {
            omp_set_num_threads(4);
#pragma omp parallel
            {
                for (int i = 0; i < t1.shape[0]; i++) {
                    Tensor<T> tmp1(t1(i, 0).shape, getT(t1));
                    tmp1 = newTensor(t1(i, 0), t1(i, 0).shape);
                    Tensor<T> tmp2(t2(i, 0).shape, getT(t2));
                    tmp2 = newTensor(t2(i, 0), t2(i, 0).shape);
                    Tensor<T> mul = matrix_mul(tmp1, tmp2);
                    for (int j = 0; j < totalSize(mul.shape); j++) {
                        result.data[idx] = mul.data[j];
                        idx++;
                    }
                }
            }
        } else {
            for (int i = 0; i < t1.shape[0]; i++) {
                Tensor<T> tmp1(t1(i, 0).shape, getT(t1));
                tmp1 = newTensor(t1(i, 0), t1(i, 0).shape);
                Tensor<T> tmp2(t2(i, 0).shape, getT(t2));
                tmp2 = newTensor(t2(i, 0), t2(i, 0).shape);
                Tensor<T> mul = matrix_mul(tmp1, tmp2);
                for (int j = 0; j < totalSize(mul.shape); j++) {
                    result.data[idx] = mul.data[j];
                    idx++;
                }
            }
        }
        return result;
    }

    bool has_no_duplicates(const std::vector<std::string> &vec) {
        std::unordered_set<std::string> seen;
        for (const auto &str: vec) {
            if (!seen.insert(str).second) {
                return false;
            }
        }
        return true;
    }

    template<typename T>
    Tensor<T> flatten(Tensor<T> t) {
        std::vector<T> V;
        getData(V, t, 0, 0);
        T *data = new T[V.size()];
        for (int i = 0; i < V.size(); i++) {
            data[i] = V[i];
        }
        std::vector<int> shape0{int(V.size())};
        return tensor<T>(shape0, data);
    }

    // ====== EINSUM helper functions END ======

    // ====== 3.4 EINSUM ======
    template<typename T>
    Tensor<T> einsum(std::string command, std::initializer_list<Tensor<T>> tensors_list) {
        auto commands = splitCommand(command);
        std::vector<std::string> input_tensor_index = commands.first;
        std::string output_tensor_index = commands.second;
        std::vector<Tensor<T>> tensors(tensors_list);
        if (input_tensor_index.size() != tensors.size()) {
            throw std::invalid_argument("input tensors does not match the instruction");
        }
        for (int i = 0; i < input_tensor_index.size(); i++) {
            if (!isAllAlpha(input_tensor_index[i])) {
                throw std::invalid_argument("input tensors does not match the instruction");
            }
            if ((int) input_tensor_index[i].length() != (int) tensors[i].shape.size()) {
                throw std::invalid_argument("input tensors does not match the instruction");
            }
        }
        if (!isAllAlpha(output_tensor_index)) {
            throw std::invalid_argument("input tensors does not match the instruction");
        }

        // element-wise product
        if (input_tensor_index.size() == 2 && output_tensor_index.size() == 1 &&
            input_tensor_index[0] == input_tensor_index[1] && input_tensor_index[0] == output_tensor_index) {
            return mul(tensors[0], tensors[1]);
        }

        // trace & diagonal
        if (input_tensor_index.size() == 1 && isSingleChar(input_tensor_index[0])) {
            if (output_tensor_index.empty()) {
                return tensors[0].trace();
            } else if (output_tensor_index.size() == 1 && output_tensor_index[0] == input_tensor_index[0][0]) {
                return tensors[0].diagonal();
            } else throw std::invalid_argument("input tensors does not match the instruction");
        }

        // sum of a tensor
        if (has_no_duplicates(input_tensor_index) && output_tensor_index.empty()) {
            std::vector<T> V;
            getData(V, tensors[0], 0, 0);
            return singleTensorSum(V);
        }

        // vector-matrix mul
        if (input_tensor_index.size() == 2 && tensors[0].shape.size() == 2 && tensors[1].shape.size() == 1) {
            std::string first_order = input_tensor_index[0].substr(0, 1);
            std::string second_order = input_tensor_index[0].substr(1, 1);
            std::vector<int> vec;
            vec.push_back(tensors[1].shape[0]);
            vec.push_back(1);
            tensors[1].shape = vec;
            init_stride(tensors[1]);
            if (first_order != second_order) {
                if (output_tensor_index.size() == 1) {
                    if (output_tensor_index == first_order && second_order == input_tensor_index[1]) {
                        return flatten<T>(matrix_mul(tensors[0], tensors[1]));
                    } else if (output_tensor_index == second_order && first_order == input_tensor_index[1]) {
                        return flatten<T>(matrix_mul(tensors[0].transpose(0, 1), tensors[1]));
                    }
                } else if (output_tensor_index.empty()) {
                    if (second_order == input_tensor_index[1]) {
                        return flatten<T>(matrix_mul(tensors[0], tensors[1]));
                    } else if (first_order == input_tensor_index[1]) {
                        return flatten<T>(matrix_mul(tensors[0].transpose(0, 1), tensors[1]));
                    }
                }
            }
        }
        if (input_tensor_index.size() == 2 && tensors[0].shape.size() == 1 && tensors[1].shape.size() == 2) {
            std::string first_order = input_tensor_index[1].substr(0, 1);
            std::string second_order = input_tensor_index[1].substr(1, 1);
            std::vector<int> vec;
            vec.push_back(tensors[0].shape[0]);
            vec.push_back(1);
            tensors[0].shape = vec;
            init_stride(tensors[0]);
            if (first_order != second_order) {
                if (output_tensor_index.size() == 1) {
                    if (output_tensor_index == first_order && second_order == input_tensor_index[0]) {
                        return flatten(matrix_mul(tensors[1], tensors[0]));
                    } else if (output_tensor_index == second_order && first_order == input_tensor_index[0]) {
                        return flatten(matrix_mul(tensors[1].transpose(0, 1), tensors[0]));
                    }
                } else if (output_tensor_index.empty()) {
                    if (second_order == input_tensor_index[0]) {
                        return flatten(matrix_mul(tensors[1], tensors[0]));
                    } else if (first_order == input_tensor_index[0]) {
                        return flatten(matrix_mul(tensors[1].transpose(0, 1), tensors[0]));
                    }
                }
            }
        }

        // permute
        if (input_tensor_index.size() == 1 && output_tensor_index.empty()) {
            std::vector<int> idx = sort_indexes(input_tensor_index[0]);
            return tensors[0].permute(idx);
        }
        if (input_tensor_index.size() == 1 && input_tensor_index[0].size() == output_tensor_index.size()) {
            std::vector<int> idx = get_indexes(input_tensor_index[0], output_tensor_index);
            return tensors[0].permute(idx);
        }

        // Vector inner products
        if (input_tensor_index.size() == 2 && input_tensor_index[0].size() == 1 &&
            input_tensor_index[0] == input_tensor_index[1] && output_tensor_index.empty()) {
            return tensors[0].mul(tensors[1]).sum(0);
        }

        // Vector outer products
        if (input_tensor_index.size() == 2 && input_tensor_index[0].size() == 1 && input_tensor_index[1].size() == 1 &&
            input_tensor_index[0] != input_tensor_index[1] && (output_tensor_index.empty() ||
                                                               (input_tensor_index[0][0] == output_tensor_index[0] &&
                                                                input_tensor_index[1][0] == output_tensor_index[1]))) {
            return outerProduct(tensors[0], tensors[1]);
        }

        //Sum over an axis
        if (tensors_list.size() == 1 && input_tensor_index.size() == 1 &&
            output_tensor_index.size() <= tensors[0].shape.size()) {
            std::vector<int> idx = get_indexes(input_tensor_index[0], output_tensor_index);
            ts::Tensor result = checkOrder(idx, tensors[0]);
            int cnt = 0;
            for (int i = 0; i < tensors[0].shape.size(); i++) {
                auto it = std::find(idx.begin(), idx.end(), i);
                if (it == idx.end()) {
                    // Value not found in the vector
                    result = ts::sum(result, i - cnt);
                    cnt++;
                }
            }
            return result;
        }

        //Matrix Multiplication (dim = 2)
        if (input_tensor_index.size() == 2 && input_tensor_index[0][1] == input_tensor_index[1][0] &&
            input_tensor_index[0][0] == output_tensor_index[0] && input_tensor_index[1][1] == output_tensor_index[1]) {
            return matrix_mul(tensors[0], tensors[1]);
        }

        //Matrix elements multiply and sum
        if (input_tensor_index.size() == 2 && input_tensor_index[0] == input_tensor_index[1]) {
            Tensor<T> tmp = tensors[0].mul(tensors[1]).sum(1);
            std::vector shape = {1};
            Tensor<T> result = ts::tensor(shape, new T[totalSize(shape)]);
            result.data[0] = 0;
            if (acceleration) {
                omp_set_num_threads(4);
#pragma omp parallel
                {
                    for (int i = 0; i < tmp.shape[0]; i++) {
                        result.data[0] += tmp.data[i];
                    }
                }
            } else {
                for (int i = 0; i < tmp.shape[0]; i++) {
                    result.data[0] += tmp.data[i];
                }
            }
            return result;
        }

        //Batch matrix multiplication (dim = 3)
        if (input_tensor_index.size() == 2 && input_tensor_index[0].size() == 3 &&
            input_tensor_index[0][0] == input_tensor_index[1][0] &&
            input_tensor_index[0][0] == output_tensor_index[0] &&
            input_tensor_index[0][1] == output_tensor_index[1] &&
            input_tensor_index[0][2] == input_tensor_index[1][1] &&
            input_tensor_index[1][2] == output_tensor_index[2]) {
            return batch_matrix_mul(tensors[0], tensors[1]);
        }

        throw std::invalid_argument("input tensors does not match the instruction");
    }
    // ====== 3.4 EINSUM END ======

//    template<typename T>
//    template<class Archive>
//    void Tensor<T>::serialize(Archive &ar, const unsigned int version) {
//        ar & shape;
////        ar & stride;
//        ar & type;
//        if(Archive::is_saving::value) {
//            int size = totalSize(shape);
//            std::vector<T> V;
//            getData(V,*this,0,0);
//            T * data0=new T[size];
//            for (int i = 0;i< size;i++){
//                data0[i]=V[i];
//            }
//            ar & boost::serialization::make_array(data0, size);
//            delete []data0;
//        } else {
//            int size = totalSize(shape);
//            data = new T[size];
//            ar & boost::serialization::make_array(data, size);
//            init_stride(*this);
//        }
//    };

    template<typename T>
    template<class Archive>
    void Tensor<T>::save(Archive &ar) const {
        int size = totalSize(shape);
        std::vector<T> V;
        getData(V, *this, 0, 0);
        T *data0 = new T[size];
        for (int i = 0; i < size; i++) {
            data0[i] = V[i];
        }
        ar(shape, type, cereal::binary_data(data0, size * sizeof(T)));
        delete[] data0;
    }

    template<typename T>
    template<class Archive>
    void Tensor<T>::load(Archive &ar) {
        ar(shape, type);
        data = new T[totalSize(shape)];
        ar(cereal::binary_data(data, totalSize(shape) * sizeof(T)));
        init_stride(*this);
    }

    template<typename T>
    void save(Tensor<T> &t, std::string filename) {
        std::cout << "serialization start" << std::endl;
        std::ofstream file(filename, std::ios::binary);
        cereal::BinaryOutputArchive oarchive(file);
        oarchive(t);
        std::cout << "serialization end" << std::endl;
    }

    template<typename T>
    Tensor<T> load(std::string filename) {
        ts::Tensor<T> tb_2;
        std::cout << "deserialization start" << std::endl;
        std::ifstream file("tensor.cereal", std::ios::binary);
        cereal::BinaryInputArchive iarchive(file);
        iarchive(tb_2);
        std::cout << "deserialization end" << std::endl;
        return tb_2;
    }

    template<typename T>
    Tensor<T> dial(Tensor<T> &t) {
        assert(t.shape.size() == 2 && t.shape[0] == t.shape[1] || t.shape.size() == 1);
        if (t.shape.size() == 1) {
            std::vector<int> dial_size{t.shape[0], t.shape[0]};
            ts::Tensor tensor = ts::tensor<T>(dial_size, 0);
            int n = *std::min_element(dial_size.begin(), dial_size.end());
            for (int i = 0; i < n; ++i) {
                std::vector<int> index(dial_size.size(), i);
                tensor.data[indexInFlatArray(index, dial_size)] = t.data[i];
            }
            return tensor;
        }
        if (t.shape.size() == 2) {
            return t.diagonal();
        }
    }

    template<typename T>
    Tensor<T> clone(Tensor<T> &t) {
        return deepcopy(t);
    }

}

#endif // TENSOR_ACLR_IMPL_CUH
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
#include <sstream>
#include <vector>
#include <string>
#include <cctype>
#include <iostream>

namespace ts {
    int totalSize(const std::vector<int> &shape);

    template<typename T>
    void init_stride(Tensor<T> &tensor) {
        tensor.stride.resize(tensor.shape.size());
        int stride = 1;
        for(int i = tensor.shape.size() - 1; i >= 0; --i) {
            tensor.stride[i] = stride;
            stride *= tensor.shape[i];
        }
    }

    // ======= Tensor Constructor =======
    /**
     * @brief Construct a new Tensor object, filled with the given value
     * @tparam T
     * @param shape
     * @param defaultValue
     * @return A new Tensor object, filled with the given value
     */
    template<typename T>
    Tensor<T>::Tensor(const std::vector<int> &shape, T defaultValue)
            :  shape(shape),
               stride(),
               type(typeid(T).name()){
        cudaMallocManaged(&data, totalSize(shape) * sizeof(T));
        std::fill(data, data + totalSize(shape), defaultValue);
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
            : shape(shape),
              type(typeid(T).name()),
              stride(){
        size_t size = totalSize(shape) * sizeof(T);
        cudaMallocManaged(&(this->data), size);
        cudaMemcpy(this->data, data, size, cudaMemcpyHostToHost);
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
    Tensor<T>::Tensor(const std::vector<int> &shape,const std::vector<int> &stride, T *data)
            : shape(shape),
              type(typeid(T).name()),
              stride(stride){
        size_t size = totalSize(shape) * sizeof(T);
        cudaMallocManaged(&(this->data), size);
        cudaMemcpy(this->data, data, size, cudaMemcpyHostToHost);
              }

    /**
     * @brief Construct a new Tensor object from another Tensor object
     * @tparam T
     * @param other
     * @return A new Tensor object from another Tensor object
     */
    template<typename T>
    Tensor<T>::Tensor(const Tensor<T> &other)
            : data(other.data),
              shape(other.shape),
              type(other.type),
              stride(other.stride){}

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
//        int nextIndexStep = (dimension < tensor.shape.size() - 1)
//                            ? totalSize(std::vector<int>(tensor.shape.begin() + dimension + 1, tensor.shape.end()))
//                            : 1;
        int nextIndexStep = tensor.stride[dimension];
        for (int i = 0; i < dimSize; ++i) {
            if (i > 0) {
                os << ", ";
                if (dimension != tensor.shape.size() - 1) {
                    os << "\n" << indent << std::string(dimension + 1, ' ');
                }
            }
            printTensor(os, tensor, index + i * nextIndexStep, dimension + 1, indent + " ");
        }
        os << "]";
    }

    /**
     * @brief Get tensor's data (continuous)
     * @tparam T
     * @param os
     * @param tensor
     * @return
     */
    template<typename T>
    T* getTensorData(const Tensor<T> &tensor) {
        T * data;
        size_t size = totalSize(tensor.shape) * sizeof(T);
        cudaMallocManaged(&data, size);
        int pointer = 0;
        getTData(tensor, 0, 0, data, pointer);
        return data;
    }

    // Helper function to get tensor's data (continuous)
    template<typename T>
    void getTData(const Tensor<T> &tensor, int index, int dimension, T* data, int& pointer) {
        if (dimension == tensor.shape.size()) {
            data[pointer++] = tensor.data[index];
            return;
        }
        int dimSize = tensor.shape[dimension];
        int nextIndexStep = tensor.stride[dimension];
        for (int i = 0; i < dimSize; ++i) {
            getTData(tensor, index + i * nextIndexStep, dimension + 1, data, pointer);
        }
    }

    /**
     * @brief Input tensor's data (from continuous to origin)
     * @tparam T
     * @param os
     * @param tensor
     * @return
     */
    template<typename T>
    void inputTensorData(Tensor<T> &tensor, T * data) {
        int pointer = 0;
        inputTData(tensor, 0, 0, data, pointer);
    }

    // Helper function to get tensor's data (continuous)
    template<typename T>
    void inputTData(Tensor<T> &tensor, int index, int dimension, T* data, int& pointer) {
        if (dimension == tensor.shape.size()) {
            tensor.data[index]=data[pointer++];
            return;
        }
        int dimSize = tensor.shape[dimension];
        int nextIndexStep = tensor.stride[dimension];
        for (int i = 0; i < dimSize; ++i) {
            inputTData(tensor, index + i * nextIndexStep, dimension + 1, data, pointer);
        }
    }

    // Helper function to get shape of tensor
    template<typename T>
    void Tensor<T>::printShape() {
        std::cout<<"[ ";
        for (int i:shape) {
            std::cout<<i<<" ";
        }
        std::cout<<"]"<<std::endl;
    }


    // Helper function to get new tensor
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
    T getT(const Tensor<T> &tensor){
        T initial_type;
        if (tensor.type == typeid(int).name()){
            initial_type = 0;
        }else if (tensor.type == typeid(double).name() || tensor.type == typeid(float).name()){
            initial_type = 0.0f;
        }else if (tensor.type == typeid(bool).name()){
            initial_type = false;
        }
        return initial_type;
    }

    // Helper function to construct a new tensor with given data vector
    template<typename T>
    Tensor<T> newTensor(const Tensor<T> &tensor, const std::vector<int> &shape) {
        ts::Tensor new_tensor(shape,getT(tensor));
        std::vector<T> V;
        getData(V,tensor,0,0);
        for(int i = 0; i < V.size(); i++){
            new_tensor.data[i] = V[i];
        }
        return new_tensor;
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
    Tensor<T> deepcopy(const Tensor<T> other) {
        Tensor<T> result=Tensor(other.shape,other.stride,other.data);
        return result;
    }

    template<typename T>
    Tensor<T>::operator bool() const {
        return !shape.empty() && data != nullptr && !type.empty();
    }

    //get a value from data
    template<typename T>
    T& get_value(Tensor<T>& t, const std::vector<int>& inds) {
        int offset = 0;
        for(int i = 0; i < t.shape.size(); ++i)
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
    Tensor<T> Tensor<T>::operator()(int index,int dim) {
        assert(*this);
        std::vector<int> new_shape = shape;
        new_shape.erase(new_shape.begin()+dim);
        std::vector<int> new_stride = stride;
        new_stride.erase(new_stride.begin()+dim);
        Tensor<T> tensor(new_shape,new_stride, data + index * stride[dim]);
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
        return this->operator()(index,indices,0);
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
    Tensor<T> Tensor<T>::operator()(int index, std::vector<int> indices,int dim) {
        assert(*this);
        assert(0 <= index && index < shape[dim]);
        assert(indices.size() == 2 && 0 <= indices[0] && indices[0] < indices[1] && indices[1] <= shape[1]);
        Tensor<T> tensor0 = this->operator()(index, dim);
        std::vector<int> new_shape = tensor0.shape;
        new_shape[0] = indices[1] - indices[0];
        Tensor<T> tensor(new_shape,tensor0.stride,
                         tensor0.data + indices[0] * tensor0.stride[0]);
        return tensor;
    }


    // ------- 2.1 Indexing and Slicing End -------

    // ------- 2.2 Joining -------
    /**
     * @brief Concatenate the given tensors along the given dimension
     * @tparam T
     * @param tensors
     * @param dim
     * @return The concatenated tensor
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

    /**
     * @brief Multiply the given tensors along the given dimension
     * @tparam T The type of the tensor
     * @param tensor The given tensor
     * @param shape The shape of the multiplier
     * @return The tiled tensor
     */
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
                        std::accumulate(new_shape.begin() + i + 1, new_shape.end(), 1, std::multiplies<>())) %
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

    std::vector<int> calculate_indices(int linear_index, const std::vector<int>& shape) {
        std::vector<int> indices(shape.size());
        for(int i = shape.size() - 1; i >= 0; --i) {
            indices[i] = linear_index % shape[i];
            linear_index /= shape[i];
        }
        return indices;
    }

    template<typename T>
    Tensor<T> &Tensor<T>::operator=(T value) {
        assert(!this->shape.empty());

        int element_count = std::accumulate(this->shape.begin(), this->shape.end(), 1, std::multiplies<int>());
        for(int i = 0; i < element_count; ++i) {
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
            for(int j = 0; j < element_count; ++j) {
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
    Tensor<T> Tensor<T>::permute(const std::vector<int>& dims) {
        std::vector<int> origin_shape = shape;
        std::vector<int> origin_stride = stride;
        for(int i = 0; i < this->shape.size(); ++i) {
            this->shape[i] = origin_shape[dims[i]];
            this->stride[i] = origin_stride[dims[i]];
        }
        return *this;
    }

    template<typename T>
    Tensor<T> permute(Tensor<T> &tensor,const std::vector<int>& dims){
        return tensor.permute(dims);
    }

    template<typename T>
    Tensor<T> transpose(Tensor<T> &tensor, int dim0, int dim1){
        return tensor.transpose(dim0,dim1);
    }

    // ------- 2.5 View -------
    // 是否经历转置导致data不连续
    template<typename T>
    bool is_contiguous(const Tensor<T>& t) {
        int stride = 1;
        for(int i = t.shape.size() - 1; i >= 0; --i) {
            if(t.stride[i] != stride)
                return false;
            stride *= t.shape[i];
        }
        return true;
    }

    template<typename T>
    Tensor<T> Tensor<T>::view(const std::vector<int>& shape0) {
        if (!is_contiguous(*this)){
            throw std::invalid_argument("input tensor is not contiguous");
        }
        int stride0 = 1;
        for(int i = shape.size() - 1; i >= 0; --i) {
            this->shape[i] = shape0[i];
            this->stride[i] = stride0;
            stride0 *= shape0[i];
        }
        return *this;
    }

    template<typename T>
    Tensor<T> view(Tensor<T> &tensor,const std::vector<int>& shape0){
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
    __global__ void addKernel(const T *data1,const T *data2, T *data3, int size) {
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



    // ======= 3.2 Reduction operations =======

    // ======= 3.2.1 Sum ======
    template<typename T>
    Tensor<T> sum(Tensor<T> &tensor, int dim) {
        assert(dim <= tensor.shape.size());
//        std::cout << tensor.getT() << std::endl;
        Tensor<T> result(tensor(0, dim).shape, getT(tensor));
        for (int i = 0; i < tensor.shape[dim]; i++){
            Tensor<T>tmp(tensor(0, dim).shape,getT(tensor));
            tmp = newTensor(tensor(i,dim),tensor(i,dim).shape);
            for(int j = 0; j < totalSize(result.shape); j++){
                result.data[j] += tmp.data[j];
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
        for (int i = 0; i < tensor.shape[dim]; i++){
            Tensor<T>tmp(tensor(0, dim).shape,getT(tensor));
            tmp = newTensor(tensor(i,dim),tensor(i,dim).shape);
            for(int j = 0; j < totalSize(result.shape); j++){
                result.data[j] += tmp.data[j];
            }
        }

        for(int i = 0; i < totalSize(result.shape); i++){
            result.data[i] = result.data[i]/tensor.shape[dim];
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
    Tensor<T> max(Tensor<T>& tensor1, Tensor<T>& tensor2){
        assert(tensor1.shape == tensor2.shape);
        ts::Tensor result = ts::tensor(tensor1.shape, new T[totalSize(tensor1.shape)]);
        int size = totalSize(tensor1.shape);
        for (int i = 0; i < size; i++)
        {
            result.data[i] = std::max(tensor1.data[i],tensor2.data[i]);
        }
        return result;
    }

    template<typename T>
    Tensor<T> max(Tensor<T> &tensor, int dim) {
        assert(dim <= tensor.shape.size());
        Tensor<T> result(tensor(0, dim).shape, std::numeric_limits<T>::min());
        for (int i = 0; i < tensor.shape[dim]; i++){
            Tensor<T>tmp(tensor(0, dim).shape,getT(tensor));
            tmp = newTensor(tensor(i,dim),tensor(i,dim).shape);
            result = max(result,tmp);
        }
        return result;
    }

    template<typename T>
    Tensor<T> Tensor<T>::max(int dim) {
        return ts::max(*this, dim);
    }

    // ======= 3.2.1 Max End ======

    // ======= 3.2.4 Min ======
    template<typename T>
    Tensor<T> min(Tensor<T>& tensor1, Tensor<T>& tensor2){
        assert(tensor1.shape == tensor2.shape);
        ts::Tensor result = ts::tensor(tensor1.shape, new T[totalSize(tensor1.shape)]);
        int size = totalSize(tensor1.shape);
        for (int i = 0; i < size; i++)
        {
            result.data[i] = std::min(tensor1.data[i],tensor2.data[i]);
        }
        return result;
    }

    template<typename T>
    Tensor<T> min(Tensor<T> &tensor, int dim) {
        assert(dim <= tensor.shape.size());
        Tensor<T> result(tensor(0, dim).shape, std::numeric_limits<T>::max());
        for (int i = 0; i < tensor.shape[dim]; i++){
            Tensor<T>tmp(tensor(0, dim).shape,getT(tensor));
            tmp = newTensor(tensor(i,dim),tensor(i,dim).shape);
            result = min(result,tmp);
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
        assert(data)
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
        assert(data);
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
        assert(data);
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
        assert(data);
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
        assert(data);
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
        assert(data);
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

    // ====== EINSUM helper functions ======

    // trace
    template<typename T>
    Tensor<T> Tensor<T>::trace() {
        int shape_size = shape[0];
        for (int i : shape){
            if (i!=shape_size){
                throw std::invalid_argument("input tensor is not a square tensor");
            }
        }
        T trace = (T)0;
        std::vector<int> inds;
        inds.reserve(shape.size());
        for (int i  = 0; i<shape.size();i++) {
            inds.push_back(i);
        }
        for (int t  = 0; t<shape_size;t++){
            for (int i  = 0; i<shape.size();i++) {
                inds[i]=t;
            }
            trace+=get_value(*this,inds);
        }
        return tensor({1},trace);
    }

    // diagonal
    template<typename T>
    Tensor<T> Tensor<T>::diagonal() {
        int shape_size = shape[0];
        for (int i : shape){
            if (i!=shape_size){
                throw std::invalid_argument("input tensor is not a square tensor");
            }
        }
        T* diagonal = new T[shape_size];
        std::vector<int> inds;
        inds.reserve(shape.size());
        for (int i  = 0; i<shape.size();i++) {
            inds.push_back(i);
        }
        for (int t  = 0; t<shape_size;t++){
            for (int i  = 0; i<shape.size();i++) {
                inds[i]=t;
            }
            diagonal[t]=get_value(*this,inds);
        }
        return tensor({shape_size},diagonal);
    }

    std::pair<std::vector<std::string>, std::string> splitCommand(const std::string& command) {
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

    bool isSingleChar(const std::string& str) {
        for (char c : str) {
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
                  [&str](size_t i1, size_t i2) {return str[i1] < str[i2];});
        return idx;
    }

    std::vector<int> get_indexes(const std::string &str1, const std::string &str2) {
        std::unordered_map<char, int> char_to_index;
        for (int i = 0; i < str1.size(); ++i) {
            char_to_index[str1[i]] = i;
        }
        std::vector<int> indexes;
        for (char c : str2) {
            if (char_to_index.find(c) == char_to_index.end()) {
                throw std::runtime_error("Character not found in the first string");
            }
            indexes.push_back(char_to_index[c]);
        }
        return indexes;
    }

    bool isAllAlpha(const std::string &str) {
        return std::all_of(str.begin(), str.end(), [](char c){ return ::isalpha(c) || c == '.'; });
    }

    // ====== EINSUM helper functions END ======

    // ====== 3.4 EINSUM ======
    template<typename T>
    Tensor<T> einsum(std::string command, std::initializer_list<Tensor<T>> tensors_list) {
        auto commands = splitCommand(command);
        std::vector<std::string> input_tensor_index = commands.first;
        std::string output_tensor_index = commands.second;
        std::vector<Tensor<T>> tensors(tensors_list);
        if (input_tensor_index.size()!=tensors.size()){
            throw std::invalid_argument("input tensors does not match the instruction");
        }
        for (int i = 0;i<input_tensor_index.size();i++){
            if (!isAllAlpha(input_tensor_index[i])){
                throw std::invalid_argument("input tensors does not match the instruction");
            }
            if ((int)input_tensor_index[i].length()!=(int)tensors[i].shape.size()){
                throw std::invalid_argument("input tensors does not match the instruction");
            }
        }
        if (!isAllAlpha(output_tensor_index)){
            throw std::invalid_argument("input tensors does not match the instruction");
        }

        // trace & diagonal
        if (input_tensor_index.size()==1&& isSingleChar(input_tensor_index[0])){
            if (output_tensor_index.empty()){
                return tensors[0].trace();
            }else if (output_tensor_index.size()==1&&output_tensor_index[0]==input_tensor_index[0][0]){
                return tensors[0].diagonal();
            }else throw std::invalid_argument("input tensors does not match the instruction");
        }

        // permute
        if (input_tensor_index.size()==1&&output_tensor_index.empty()){
            std::vector<int> idx = sort_indexes(input_tensor_index[0]);
            return tensors[0].permute(idx);
        }
        if (input_tensor_index.size()==1&&input_tensor_index[0].size()==output_tensor_index.size()){
            std::vector<int> idx = get_indexes(input_tensor_index[0],output_tensor_index);
            return tensors[0].permute(idx);
        }

        // Vector inner products
        if (input_tensor_index.size()==2&&input_tensor_index[0].size()==1&&input_tensor_index[0]==input_tensor_index[1]&&output_tensor_index.empty()){
            return tensors[0].mul(tensors[1]).sum(0);
        }

        //


        throw std::invalid_argument("input tensors does not match the instruction");
    }
    // ====== 3.4 EINSUM END ======

}

#endif // TENSOR_ACLR_IMPL_CUH
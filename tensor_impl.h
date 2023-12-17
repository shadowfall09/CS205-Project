#include <iostream>
#include <utility>
#include <vector>
#include <random>
#include <string>
#include <memory>
#include <iomanip>
#include "tensor.h"

namespace ts {
    template<typename T>
    Tensor<T>::Tensor(std::initializer_list<std::initializer_list<T>> list) {
        // Determine the shape
        shape.push_back(list.size());
        shape.push_back((*list.begin()).size());
        // Flatten the initializer list
        for (auto &sublist: list) {
            for (T elem: sublist) {
                data.push_back(elem);
            }
        }
        // Set the type
        type = typeid(T).name();
    }

    // Default constructor for tensor with data and shape
    template<typename T>
    Tensor<T>::Tensor(std::vector<T> data,
                      std::vector<int> shape
    ) :data(data), shape(std::move(shape)) {
        type = typeid(T).name();
    }

    // Static method to create a tensor with zeros
    template<typename T>
    Tensor<T> Tensor<T>::zeros(const std::vector<int> &shape) {
        auto shape_size = 1;
        for (int i: shape) {
            if (i <= 0) {
                throw std::invalid_argument("Shape must be positive");
            }
            shape_size *= i;
        }
        std::vector<T> data(shape_size, static_cast<T>(0));
        return Tensor(data, shape);
    }

    // Static method to create a tensor with ones
    template<typename T>
    Tensor<T> Tensor<T>::ones(const std::vector<int> &shape) {
        auto shape_size = 1;
        for (int i: shape) {
            if (i <= 0) {
                throw std::invalid_argument("Shape must be positive");
            }
            shape_size *= i;
        }
        std::vector<T> data(shape_size, static_cast<T>(1));
        return Tensor(data, shape);
    }

    // Static method to create a tensor with a specific value
    template<typename T>
    Tensor<T> Tensor<T>::full(const std::vector<int> &shape, T value) {
        auto shape_size = 1;
        for (int i: shape) {
            if (i <= 0) {
                throw std::invalid_argument("Shape must be positive");
            }
            shape_size *= i;
        }
        std::vector<T> data(shape_size, value);
        return Tensor(data, shape);
    }

    // Static method to create a tensor with random values
    template<typename T>
    Tensor<T> Tensor<T>::rand(std::vector<int> shape) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        auto shape_size = 1;
        for (int i: shape) {
            if (i <= 0) {
                throw std::invalid_argument("Shape must be positive");
            }
            shape_size *= i;
        }
        std::vector<T> random_data(shape_size);
        for (auto &elem: random_data) {
            elem = static_cast<T>(dis(gen));
        }
        return Tensor(random_data, shape);
    }

    // Static method to create an identity matrix tensor
    template<typename T>
    Tensor<T> Tensor<T>::eye(const std::vector<int> &shape) {
        if (shape.size() != 2) {
            throw std::invalid_argument("Shape must be 2D");
        }
        std::vector<T> data(shape[0] * shape[1], static_cast<T>(0));
        int n = std::min(shape[0], shape[1]);
        for (int i = 0; i < n; ++i) {
            data[i * shape[1] + i] = static_cast<T>(1);
        }
        return Tensor(data, shape);
    }

    // Method to get the size of the tensor
    template<typename T>
    std::vector<int> Tensor<T>::size() const {
        return shape;
    }

    // Method to get the type of the tensor
    template<typename T>
    std::string Tensor<T>::type_name() const {
        return type;
    }

    // Method to get the data pointer
    template<typename T>
    const void *Tensor<T>::data_ptr() const {
        return static_cast<const void *>(data.data());
    }

    // Helper function to create a tensor
    template<typename T>
    Tensor<T> tensor(std::initializer_list<std::initializer_list<T>> list) {
        return Tensor<T>(list);
    }

    //Helper function to create a tensor
    template<typename T>
    Tensor<T> tensor(std::vector<T> data, std::vector<int> shape) {
        return Tensor<T>(data, shape);
    }

    // Helper function to create a tensor with random values
    template<typename T>
    Tensor<T> rand(std::vector<int> shape) {
        return Tensor<T>::rand(shape);
    }

    template<typename T>
    Tensor<T> zeros(const std::vector<int> &shape) {
        return Tensor<T>::zeros(shape);
    }

    template<typename T>
    Tensor<T> ones(const std::vector<int> &shape) {
        return Tensor<T>::ones(shape);
    }

    template<typename T>
    Tensor<T> full(const std::vector<int> &shape, T value) {
        return Tensor<T>::full(shape, value);
    }

    template<typename T>
    Tensor<T> eye(const std::vector<int> &shape) {
        return Tensor<T>::eye(shape);
    }
}




#ifndef TENSOR_ACLR_CUH
#define TENSOR_ACLR_CUH

#include <vector>
#include <string>
#include <iostream>

namespace ts {
    template<typename T>
    class Tensor {
    public:
        T *data;
        std::vector<int> shape;
        std::string type;

        Tensor(const std::vector<int>& shape, T defaultValue);

        Tensor(std::vector<int> shape, T *data);


        template<typename U>
        friend std::ostream &operator<<(std::ostream &os, const Tensor<U> &tensor);

        // Other member functions...
    };

    template<typename T>
    Tensor<T> tensor(std::vector<int> shape, T *data);

    template<typename T>
    Tensor<T> tensor(std::vector<int> shape, T defaultValue);

    template<typename T>
    Tensor<T> rand(std::vector<int> shape);

    template<typename T>
    Tensor<T> zeros(std::vector<int> shape);

    template<typename T>
    Tensor<T> ones(std::vector<int> shape);

    template<typename T>
    Tensor<T> full(std::vector<int> shape, T value);

    template<typename T>
    Tensor<T> eye(int shape);
}

#endif // TENSOR_ACLR_CUH

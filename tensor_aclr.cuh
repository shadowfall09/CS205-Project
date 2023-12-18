#ifndef TENSOR_ACLR_CUH
#define TENSOR_ACLR_CUH

#include <vector>
#include <string>
#include <iostream>

namespace ts {
    template<typename T>
    class Tensor {
    public:
        std::vector<T> data;
        std::vector<int> shape;
        std::string type;

        Tensor(std::vector<int> shape, T defaultValue);

        explicit Tensor(std::vector<int> shape);

        template<typename U>
        friend std::ostream &operator<<(std::ostream &os, const Tensor<U> &tensor);

        // Other member functions...
    };

    template<typename T>
    Tensor<T> rand(std::vector<int> size);

    template<typename T>
    Tensor<T> zeros(std::vector<int> size);

    template<typename T>
    Tensor<T> ones(std::vector<int> size);

    template<typename T>
    Tensor<T> full(std::vector<int> size, T value);

    template<typename T>
    Tensor<T> eye(int size);
}

#endif // TENSOR_ACLR_CUH

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

        Tensor(const std::vector<int> &shape, T defaultValue);

        Tensor(std::vector<int> shape, T *data);


        template<typename U>
        friend std::ostream &operator<<(std::ostream &os, const Tensor<U> &tensor);

        // indexing: t(1)
        Tensor<T> operator()(int index);

        // slicing: t(2,{2,4})
        Tensor<T> operator()(int axis, std::vector<int> indices);

        // add
        Tensor<T> add(const Tensor<T>& other);
        Tensor<T> add(T value);
        Tensor<T> operator+(const Tensor<T>& other);
        Tensor<T> operator+(T value);

        // subtract
        Tensor<T> sub(const Tensor<T>& other);
        Tensor<T> sub(T value);
        Tensor<T> operator-(const Tensor<T>& other);
        Tensor<T> operator-(T value);

        // multiply
        Tensor<T> mul(const Tensor<T>& other);
        Tensor<T> mul(T value);
        Tensor<T> operator*(const Tensor<T>& other);
        Tensor<T> operator*(T value);

        //divide
        Tensor<T> div(const Tensor<T>& other);
        Tensor<T> div(T value);
        Tensor<T> operator/(const Tensor<T>& other);
        Tensor<T> operator/(T value);

        //log
        Tensor<T> log();

        // Other member functions...
        T getT();
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

    template<typename T>
    Tensor<T> add(Tensor<T>& t1, Tensor<T>& t2);

    template<typename T>
    Tensor<T> add(Tensor<T>& tensor, T value);

    template<typename T>
    Tensor<T> sub(Tensor<T>& t1, Tensor<T>& t2);

    template<typename T>
    Tensor<T> sub(Tensor<T>& tensor, T value);

    template<typename T>
    Tensor<T> mul(Tensor<T>& t1, Tensor<T>& t2);

    template<typename T>
    Tensor<T> mul(Tensor<T>& tensor, T value);

    template<typename T>
    Tensor<T> div(Tensor<T>& t1, Tensor<T>& t2);

    template<typename T>
    Tensor<T> div(Tensor<T>& tensor, T value);

    template<typename T>
    Tensor<T> log(Tensor<T>& tensor);
}

#endif // TENSOR_ACLR_CUH

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
        std::vector<int> stride;
        std::string type;

        Tensor(const std::vector<int> &shape, T defaultValue);

        Tensor(const std::vector<int> &shape, T *data);

        Tensor(const std::vector<int> &shape,const std::vector<int> &stride, T *data);

        Tensor(const Tensor<T> &other);

        template<typename U>
        friend std::ostream &operator<<(std::ostream &os, const Tensor<U> &tensor);

        explicit operator bool() const;

        // indexing
        Tensor<T> operator()(int index);

        // slicing
        Tensor<T> operator()(int index, std::vector<int> indices);

        // mutating
        Tensor<T> &operator=(T value);
        Tensor<T> &operator=(std::initializer_list<T> values);

        // transpose
        Tensor<T> transpose(int dim0, int dim1);
        Tensor<T> permute(const std::vector<int>& dims);

        // add
        Tensor<T> add(const Tensor<T> &other);

        Tensor<T> add(T value);

        Tensor<T> operator+(const Tensor<T> &other);

        Tensor<T> operator+(T value);

        // subtract
        Tensor<T> sub(const Tensor<T> &other);

        Tensor<T> sub(T value);

        Tensor<T> operator-(const Tensor<T> &other);

        Tensor<T> operator-(T value);

        // multiply
        Tensor<T> mul(const Tensor<T> &other);

        Tensor<T> mul(T value);

        Tensor<T> operator*(const Tensor<T> &other);

        Tensor<T> operator*(T value);

        //divide
        Tensor<T> div(const Tensor<T> &other);

        Tensor<T> div(T value);

        Tensor<T> operator/(const Tensor<T> &other);

        Tensor<T> operator/(T value);

        //log
        Tensor<T> log();

        //eq, ne, gt, ge, lt, le.
        Tensor<bool> eq(const Tensor<T> &other);

        Tensor<bool> operator==(const Tensor<T> &other);

        Tensor<bool> ne(const Tensor<T> &other);

        Tensor<bool> operator!=(const Tensor<T> &other);

        Tensor<bool> le(const Tensor<T> &other);

        Tensor<bool> operator<=(const Tensor<T> &other);

        Tensor<bool> lt(const Tensor<T> &other);

        Tensor<bool> operator<(const Tensor<T> &other);

        Tensor<bool> ge(const Tensor<T> &other);

        Tensor<bool> operator>=(const Tensor<T> &other);

        Tensor<bool> gt(const Tensor<T> &other);

        Tensor<bool> operator>(const Tensor<T> &other);

        // Other member functions...
        Tensor<T> trace();
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
    Tensor<T> eye(std::vector<int> shape);

    template<typename T>
    Tensor<T> cat(std::vector<Tensor<T>> &tensors, int dim);

    template<typename T>
    Tensor<T> tile(Tensor<T> &tensor, std::vector<int> shape);

    template<typename T>
    Tensor<T> permute(Tensor<T> &tensor,const std::vector<int>& dims);

    template<typename T>
    Tensor<T> transpose(Tensor<T> &tensor, int dim0, int dim1);

    template<typename T>
    Tensor<T> add(Tensor<T> &t1, Tensor<T> &t2);

    template<typename T>
    Tensor<T> add(Tensor<T> &tensor, T value);

    template<typename T>
    Tensor<T> sub(Tensor<T> &t1, Tensor<T> &t2);

    template<typename T>
    Tensor<T> sub(Tensor<T> &tensor, T value);

    template<typename T>
    Tensor<T> mul(Tensor<T> &t1, Tensor<T> &t2);

    template<typename T>
    Tensor<T> mul(Tensor<T> &tensor, T value);

    template<typename T>
    Tensor<T> div(Tensor<T> &t1, Tensor<T> &t2);

    template<typename T>
    Tensor<T> div(Tensor<T> &tensor, T value);

    template<typename T>
    Tensor<T> eq(Tensor<T> &t1, Tensor<T> &t2);

    template<typename T>
    Tensor<T> ne(Tensor<T> &t1, Tensor<T> &t2);

    template<typename T>
    Tensor<T> le(Tensor<T> &t1, Tensor<T> &t2);

    template<typename T>
    Tensor<T> lt(Tensor<T> &t1, Tensor<T> &t2);

    template<typename T>
    Tensor<T> ge(Tensor<T> &t1, Tensor<T> &t2);

    template<typename T>
    Tensor<T> gt(Tensor<T> &t1, Tensor<T> &t2);
}

#endif // TENSOR_ACLR_CUH

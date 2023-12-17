#ifndef CPP_PROJECT_TENSOR_H
#define CPP_PROJECT_TENSOR_H
#include <iostream>
#include <utility>
#include <vector>
#include <random>
#include <string>
#include <memory>
#include <iomanip>

namespace ts {

    template <typename T>
    class Tensor {
    private:
        std::vector<T> data;
        std::vector<int> shape;
        std::string type;

    public:
        Tensor(std::initializer_list<std::initializer_list<T>> list);
        Tensor(std::vector<T> data, std::vector<int> shape);
        static Tensor zeros(const std::vector<int> &shape);
        static Tensor ones(const std::vector<int> &shape);
        static Tensor full(const std::vector<int> &shape, T value);
        static Tensor rand(std::vector<int> shape);
        static Tensor eye(const std::vector<int> &shape);
        [[nodiscard]] std::vector<int> size() const;
        [[nodiscard]] std::string type_name() const;
        [[nodiscard]] const void *data_ptr() const;
        friend std::ostream &operator<<(std::ostream &os, const Tensor &tensor){
            // TODO: support more than 2D tensor
            os << "[";
            for (size_t i = 0; i < tensor.shape[0]; ++i) {
                if (i != 0) os << " ";
                os << "[";
                for (size_t j = 0; j < tensor.shape[1]; ++j) {
                    os << " " << std::fixed << std::setprecision(4) << tensor.data[i * tensor.shape[1] + j];
                    if (j != tensor.shape[1] - 1) os << ",";
                }
                os << "]";
                if (i != tensor.shape[0] - 1) os << "," << std::endl;
            }
            os << "]";
            return os;
        }
    };

    template <typename T>
    Tensor<T> tensor(std::initializer_list<std::initializer_list<T>> list);

    template <typename T>
    Tensor<T> tensor(std::vector<T> data, std::vector<int> shape);

    template <typename T>
    Tensor<T> rand(std::vector<int> shape);

    template <typename T>
    Tensor<T> zeros(const std::vector<int>& shape);

    template <typename T>
    Tensor<T> ones(const std::vector<int>& shape);

    template <typename T>
    Tensor<T> full(const std::vector<int>& shape, T value);

    template <typename T>
    Tensor<T> eye(const std::vector<int>& shape);
} // namespace ts

#endif //CPP_PROJECT_TENSOR_H

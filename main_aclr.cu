#include "tensor_aclr.cuh"
#include "tensor_aclr_impl.cuh"
#include <iostream>

int main() {
    // test all functions in main_aclr.cu
    std::vector<int> shape = {4, 5};
ts::Tensor<int> tensor = ts::tensor(shape, 1);
    std::cout << tensor << std::endl << std::endl;
    std::cout << ts::rand<int>(shape) << std::endl << std::endl;
    std::cout << ts::zeros<int>(shape) << std::endl << std::endl;
    std::cout << ts::ones<int>(shape) << std::endl << std::endl;
    std::cout << ts::full<int>(shape, 2) << std::endl << std::endl;
    std::cout << ts::eye<int>(3) << std::endl << std::endl;

    return 0;
}
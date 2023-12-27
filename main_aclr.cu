#include "tensor_aclr.cuh"
#include "tensor_aclr_impl.cuh"
#include <iostream>
#include <chrono>

int main() {
    ts::Tensor t = ts::tensor({3, 5},
                              new double[15]{
                                      0.1, 1.2, 3.4, 5.6, 7.8,
                                      2.2, 3.1, 4.5, 6.7, 8.9,
                                      4.9, 5.2, 6.3, 7.4, 8.5});
    std::cout << t(1) << std::endl;
    std::cout << t(2, {2, 4}) << std::endl;


    ts::Tensor<float> t1(std::vector<int>{3, 2},0.0f);
    ts::Tensor<float> t2(std::vector<int>{3, 2},0.0f);

    t1.data[0] = 0.1f; t1.data[1] = 1.2f;
    t1.data[2] = 2.2f; t1.data[3] = 3.1f;
    t1.data[4] = 4.9f; t1.data[5] = 5.2f;

    t2.data[0] = 0.2f; t2.data[1] = 1.3f;
    t2.data[2] = 2.2f; t2.data[3] = 3.2f;
    t2.data[4] = 4.8f; t2.data[5] = 5.2f;

    std::cout << (t1 == t2) << std::endl;

    return 0;
}

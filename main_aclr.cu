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
//    ts::Tensor t2 = ts::tensor({2,2,2},
//                              new double[8]{
//                                      1,2,3,4,5,6,7,8});
//    std::cout << t2(1, {0,2}) << std::endl;
    std::cout << t << std::endl;
    std::cout << ts::get_value(t,{1,0})<<std::endl;
    std::cout << t.transpose(0,1) << std::endl;
//    std::cout << t.transpose(0,1) << std::endl;
//    std::cout << t(2) << std::endl;
//    std::cout << t(2, {0,2}) << std::endl;
    std::cout << ts::get_value(t,{0,1});

    return 0;
}

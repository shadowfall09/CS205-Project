#include "tensor_aclr_impl.cuh"
#include <iostream>

using namespace std;
int main() {
    ts::Tensor t = ts::tensor({3,3},
                              new double[9]{
                                      1,2,3,
                                      4,5,6,
                                      7,8,9 });
    ts::Tensor t1 = ts::tensor({3,3},
                              new double[9]{
                                      2,3,
                                      4,5,6,
                                      7,8,9,10 });

    std::cout << (t-t1)<< std::endl;
//    std::cout << ts::einsum("iii",{t})<< std::endl;

//    std::cout << t.transpose(0,1) << std::endl;
//    std::cout << t.view({5,3})<<std::endl;

//    std::cout << t << std::endl;
//    std::cout << t.sum(0) << std::endl;
//    std::cout << sum(t,1) << std::endl;
//    std::cout << max(t,1) << std::endl;


    return 0;
}

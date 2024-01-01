#include "tensor_aclr_impl.cuh"
#include <iostream>

using namespace std;
int main() {
    ts::Tensor t = ts::tensor({3,2},
                              new double[6]{
                                      0.1, 1.2,
                                      2.2,3.1,
                                      4.9, 5.2 });


//    std::cout << t.trace() << std::endl;
//    std::cout << t.transpose(0,1) << std::endl;
//    std::cout << t.view({5,3})<<std::endl;

    std::cout << t << std::endl;
    std::cout << t.sum(0) << std::endl;
    std::cout << sum(t,1) << std::endl;
    std::cout << max(t,1) << std::endl;


    return 0;
}

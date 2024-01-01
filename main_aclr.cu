#include "tensor_aclr_impl.cuh"
#include <iostream>

using namespace std;
int main() {
    ts::Tensor t = ts::tensor({4,4},
                              new double[16]{
                                    0.1, 1.2, 3.4, 5.6,
                                      7.8,2.2, 3.1, 4.5,
                                      6.7, 8.9,4.9, 5.2,
                                      6.3, 7.4, 8.5,10});

    std::cout << t.trace() << std::endl;
//    std::cout << t.transpose(0,1) << std::endl;
//    std::cout << t.view({5,3})<<std::endl;
//    std::cout << t.transpose(0,1) << std::endl;


//    t(2)=1;
    std::cout << t(2) << std::endl;
    std::cout << t(2,1) << endl;
    cout<<t(2,{2,4},1)<<endl;
//    std::cout << t << std::endl;
    return 0;
}

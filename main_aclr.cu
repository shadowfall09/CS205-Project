#include "tensor_aclr_impl.cuh"
#include <iostream>
#include <chrono>
#include "omp.h"

using namespace std;
int main() {
    ts::acceleration= true;
//    ts::Tensor t = ts::tensor({3,3},
//                              new double[9]{
//                                      1,2,3,
//                                      4,5,6,
//                                      7,8,9 });
//    ts::Tensor t1 = ts::tensor({3,3},
//                              new double[9]{
//                                      2,3,
//                                      4,5,6,
//                                      7,8,9,10 });
    ts::Tensor test1 = ts::rand<int>({2, 3, 2});
    ts::Tensor test2 = ts::rand<int>({2, 2, 3});
//    cout << t(0)(0) << endl;
//    cout << test << endl;
//    cout << test.sum(1) << endl;
//    ts::einsum("ij->j",{test});
    cout << test1 << endl;
    cout << test2 << endl;
    cout << ts::einsum("ijk,ikl->ijl",{test1,test2}) << endl;
    return 0;
}
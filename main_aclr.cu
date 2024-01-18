#include "tensor_aclr_impl.cuh"
#include <iostream>
#include <chrono>
#include "omp.h"

using namespace std;
int main() {
    ts::acceleration= true;
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
//    ts::Tensor t = ts::tensor({1000000000},11);
//    ts::Tensor t1 = ts::tensor({1000000000},10);
//    auto start = std::chrono::high_resolution_clock::now();
//    t+t1;
////    std::cout << ()<< std::endl;
//    auto end = std::chrono::high_resolution_clock::now();
//    std::chrono::duration<double> diff = end-start;
//    std::cout << "Time to run the code: " << diff.count() << " s\n";
    ts::Tensor test = ts::rand<int>({3,2});
    cout << test << endl;
    cout << test.sum(1) << endl;
    return 0;
}
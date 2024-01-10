#include "tensor_aclr_impl.cuh"
#include <iostream>
#include <chrono>

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
    auto start = std::chrono::high_resolution_clock::now();

    std::cout << (t-t1)<< std::endl;
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end-start;
    std::cout << "Time to run the code: " << diff.count() << " s\n";

                               new double[9]{
                                       2,3,
                                       4,5,6,
                                       7,8,9,10 });
//    std::cout << (t-t1)<< std::endl;
//    std::cout << ts::einsum("iii",{t})<< std::endl;

//    std::cout << t.transpose(0,1) << std::endl;
//    std::cout << t.view({5,3})<<std::endl;

//    cout << t(0,1) << endl;
    ts::Tensor test = ts::rand<int>({3,3,3});
    cout << test << endl;
    cout << ts::einsum("ijk->j",{test}) << endl;
    cout << test.sum(0).sum(1) << endl;
    return 0;
}
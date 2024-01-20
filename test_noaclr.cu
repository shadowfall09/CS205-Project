#include <iostream>
#include <chrono> // 添加此行以包含 chrono 头文件
#include "tensor_aclr_impl.cuh"

using namespace std;

int main() {
    ts::acceleration = false;

//    ts::Tensor t1 = ts::ones<int>({500000000});
//    ts::Tensor t2 = ts::ones<int>({500000000});
    ts::Tensor t1 = ts::ones<int>({50000000});
    ts::Tensor t2 = ts::ones<int>({50000000});
    // 记录开始时间
    auto start_time = chrono::high_resolution_clock::now();

    //test cuda
//    ts::Tensor cuda_result = t1 * t2;

    //test openmp
    ts::Tensor mp_result = t1.sum(0);

    // 记录结束时间
    auto end_time = chrono::high_resolution_clock::now();

    // 计算时间差并以秒为单位输出
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);

    cout << "Time taken: " << duration.count() / 1000.0 << " seconds" << endl;

    return 0;
}

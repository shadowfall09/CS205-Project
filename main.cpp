#include "tensor_impl.h"
#include <iostream>

int main() {
    ts::Tensor<float> t = ts::tensor<float>({{0.1f, 1.2f}, {2.2f, 3.1f}, {4.9f, 5.2f}});
    std::cout << t.size()[0] << ", " << t.size()[1] << std::endl;
    std::cout << t.type_name() << std::endl;
    std::cout << t.data_ptr() << std::endl;
    std::cout << t << std::endl;
    ts::Tensor<float> random_tensor = ts::rand<float>({2,3,3});
    std::cout << random_tensor << std::endl;
    ts::Tensor<float> t1 = ts::zeros<float>({2, 3});
    ts::Tensor<float> t2 = ts::ones<float>({2, 3});
    ts::Tensor<float> t3 = ts::full<float>({2, 3}, 0.6f);
    ts::Tensor<float> t4 = ts::eye<float>({3, 3});
    std::cout << t1 << std::endl << t2 << std::endl << t3 << std::endl << t4 << std::endl;
    std::vector<int> shape = {2, 3};
    std::vector<int> data = {1, 2, 3, 4, 5, 6};
    ts::Tensor<int> t5 = ts::tensor(data, shape);
    std::cout << t5 << std::endl;

//    ts::Tensor t5 = ts::tensor({{0.1, 1.2, 3.4, 5.6, 7.8}, {2.2, 3.1, 4.5, 6.7, 8.9},{4.9, 5.2, 6.3, 7.4, 8.5}});
//    std::cout << t5(1) << std::endl << t5(2,{2,4}) << std::endl;
    return 0;
}
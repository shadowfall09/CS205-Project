#include <iostream>
#include <cereal/archives/binary.hpp>
#include <fstream>
#include "tensor_aclr_impl.cuh"

using namespace std;

int main() {
    {
        // 1.1 Create a tensor from a given array by copying data to your memory
        double *data1 = new double[12]{0, 1, 2, 3,
                                       4, 5, 6, 7,
                                       8, 9, 10, 11};
        vector<int> shape1{3, 4};
        ts::Tensor t1 = ts::tensor(shape1, data1);
        // 答辩时解释用的自己写的测试文件，所以构造方法没 copy data
        cout << "1.1: given data init" << endl;
        cout << t1 << endl << endl;
    }

    {
        // 1.2 Create a tensor with a given shape and data type and initialize it randomly
        vector<int> shape2{2, 3};
        ts::Tensor t2 = ts::rand<double>(shape2);
        cout << "1.2: random init (random bound is set to 1-5000)" << endl;
        cout << t2 << endl << endl;
    }

    {
        // 1.3 Create a tensor with a given shape and data type, and initialize it with a given value
        vector<int> shape3{2, 2},
                shape4{2, 3},
                shape5{3, 3};
        ts::Tensor t3 = ts::zeros<long>(shape3);
        ts::Tensor t4 = ts::ones<int>(shape4);
        ts::Tensor t5 = ts::full<float>(shape5, 3.14);
        cout << "1.3.1: zeros init" << endl << t3 << endl;
        cout << "1.3.2: ones init" << endl << t4 << endl;
        cout << "1.3.3: value init" << endl << t5 << endl << endl;
    }

    {
        // 1.4 Create a tensor with a given shape and data type, and initialize it to a specific pattern
        vector<int> shape6_1{3, 3},
                shape6_2{3, 4},
                shape6_3{4, 3};
        ts::Tensor t6_1 = ts::eye<int>(shape6_1),
                t6_2 = ts::eye<int>(shape6_2),
                t6_3 = ts::eye<int>(shape6_3);
        cout << "1.4: pattern init" << endl;
        cout << "eye({3,3}): " << endl << t6_1 << endl;
        cout << "eye({3,4}): " << endl << t6_2 << endl;
        cout << "eye({4,3}): " << endl << t6_3 << endl << endl;
    }

    {
        // 2.1 Indexing and slicing operations
        int *data7 = new int[12]{0, 1, 2, 3,
                                 4, 5, 6, 7,
                                 8, 9, 10, 11};
        vector<int> shape7{3, 4};
        ts::Tensor t7 = ts::tensor(shape7, data7),
                t7_1 = t7(1),
                t7_2 = t7(2, {2, 4});
        cout << "2.1.1: indexing: " << endl;
        cout << "tensor: " << endl << t7 << endl;
        cout << "tensor(1): " << endl << t7_1 << endl;
        cout << "tensor(2, {2,4}): " << endl << t7_2 << endl;

        t7_1(1) = -100;
        cout << "2.1.2: memory sharing:" << endl;
        cout << "after modifying indexed tensor, original tensor is: " << endl << t7 << endl << endl;
    }

    {
        // 2.1.3: test shared_ptr in indexing
        cout << "2.1.3: shared_ptr in indexing" << endl;
        ts::Tensor<double> t7_3({1, 3}, nullptr);
        {
            ts::Tensor t7_4 = ts::tensor({2, 3}, new double[6]{0, 1, 2, 3, 4, 5});
            t7_3 = t7_4(1);
            cout << "before outer tensor destruction: " << endl << t7_3 << endl;
        }
        cout << "after outer tensor destruction: " << endl << t7_3 << endl << endl;
    }

    {
        // 2.2 Joining operations
        cout << "2.2: ts::cat(tensors, dim), ts::tile(tensor, shape)" << endl;
        vector<int> shape8_1 = {2, 3},
                shape8_2 = {2, 3};
        double *data8_1 = new double[6]{0.1, 1.1, 2.1,
                                        3.1, 4.1, 5.1},
                *data8_2 = new double[6]{0.2, 1.2, 2.2,
                                         3.2, 4.2, 5.2};
        ts::Tensor t8_1 = ts::tensor(shape8_1, data8_1),
                t8_2 = ts::tensor(shape8_2, data8_2);
        vector<ts::Tensor<double>> t8s{t8_1, t8_2};
        ts::Tensor t8_cat1 = ts::cat(t8s, 0),
                t8_cat2 = ts::cat(t8s, 1),
                t8_tile = ts::tile(t8_1, {2, 3});
        cout << "tensor1: " << endl << t8_1 << endl;
        cout << "tensor2: " << endl << t8_2 << endl;
        cout << "2.2.1: cat(tensors, 0): " << endl << t8_cat1 << endl;
        cout << "2.2.1: cat(tensors, 1): " << endl << t8_cat2 << endl;
        cout << "2.2.2: tile(tensor, {2, 3}): " << endl << t8_tile << endl << endl;
    }

    {
        // 2.3 Mutating operations
        cout << "2.3: mutating + memory sharing" << endl;
        vector<int> shape9{3, 5};
        double *data9 = new double[15]{0, 1, 2, 3, 4,
                                       5, 6, 7, 8, 9,
                                       10, 11, 12, 13, 14};
        ts::Tensor t9 = ts::tensor(shape9, data9);
        cout << "tensor: " << endl << t9 << endl;
        t9(1) = 100;
        cout << "after tensor(1) = 100, original tensor is: " << endl << t9 << endl;
        t9(2, {2, 5}) = 200;
        cout << "after tensor(2, {2, 5}) = 200, original tensor is: " << endl << t9 << endl << endl;
    }

    {
        // 2.4 Transpose and permute operations
        cout << "2.4: transpose and permute both with memory sharing" << endl;
        vector<int> shape10{3, 5};
        double *data10 = new double[15]{0.24, 1.24, 2.24, 3.24, 4.24,
                                        5.24, 6.24, 7.24, 8.24, 9.24,
                                        10.24, 11.24, 12.24, 13.24, 14.24};
        ts::Tensor t10 = ts::tensor(shape10, data10),
                t10_trans1 = ts::transpose(t10, 0, 1),
                t10_trans2 = t10.transpose(0, 1),
                t10_permute1 = ts::permute(t10, {1, 0}),
                t10_permute2 = t10.permute({1, 0});
        cout << "tensor: " << endl << t10 << endl;
        cout << "2.4.1: after transpose(t, 0, 1): " << endl << t10_trans1 << endl;
        cout << "2.4.1: after t.transpose(0, 1): " << endl << t10_trans2 << endl;
        t10_trans1(0)(1) = 100;
        cout << "2.4.1: after modifying transposed tensor, original tensor is: " << endl << t10 << endl;

        cout << "2.4.2: after permute(t, {1, 0}): " << endl << t10_permute1 << endl;
        cout << "2.4.2: after t.permute({1, 0}): " << endl << t10_permute2 << endl << endl;
        t10_permute1(1)(0) = -100;
        cout << "2.4.2: after modifying permuted tensor, original tensor is: " << endl << t10 << endl << endl;

    }

    {
        // 2.5 View operations
        cout << "2.5: view + memory sharing" << endl;
        vector<int> shape11{3, 5};
        double *data11 = new double[15]{0.25, 1.25, 2.25, 3.25, 4.25,
                                        5.25, 6.25, 7.25, 8.25, 9.25,
                                        10.25, 11.25, 12.25, 13.25, 14.25};
        ts::Tensor t11 = ts::tensor(shape11, data11);
        cout << "tensor: " << endl << t11 << endl;
        ts::Tensor t11_view1 = ts::view(t11, {5, 3});
        cout << "2.5.1: view(tensor, {5, 3}): " << endl << t11_view1 << endl;
        t11_view1(1)(1) = 100;
        cout << "2.5.2: after modifying view tensor, original tensor is: " << endl << t11 << endl << endl;
    }

    {
        // bonus 1: serialization
        cout << "4.1: serialize to tensor.cereal" << endl;
        double *data1 = new double[12]{0, 1, 2, 3,
                                       4, 5, 6, 7,
                                       8, 9, 10, 11};
        vector<int> shape1{3, 4};
        ts::Tensor tb1_1 = ts::tensor(shape1, data1);
        cout << "tensor: " << tb1_1 << endl;
        ts::save(tb1_1,"tensor.cereal");
        cout << "serialization done, file name: tensor.cereal" << endl;
        ts::Tensor<double> tb_2=ts::load<double>("tensor.cereal");
        cout << "deserialized tensor: " << endl << tb_2 << endl << endl;
    }
    return 0;
}
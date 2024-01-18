#include <iostream>
#include "tensor_aclr_impl.cuh"

using namespace std;

int main() {
    // 3.1 Pointwise operations including add , sub , mul , div , log
    cout << "3.1: Pointwise operations" << endl;
    vector<int> shape1{3,2};
    double data1[3][2]{0.1, 1.2,
                       2.2, 3.1,
                       4.9, 5.2};
    ts::Tensor t1 = ts::tensor(shape1, &data1[0][0]);
    vector<int> shape2{3,2};
    double data2[3][2]{0.2, 1.3,
                       2.3, 3.2,
                       4.8, 5.1};
    ts::Tensor t2 = ts::tensor(shape2, &data2[0][0]);
    cout << "3.1.1: add" << endl;
    cout << "tensor1: " << endl << t1 << endl;
    cout << "tensor2: " << endl << t2 << endl;
    cout << "add(t1, t2) " << endl << ts::add(t1, t2) << endl;
    cout << "t1.add(t2): " << endl << t1.add(t2) << endl;
    cout << "t1 + t2" << endl << t1 + t2 << endl;
    cout << "add(t1, 1.0)" << endl << ts::add(t1, 1.0) << endl;
    cout << "t1.add(1.0)" << endl << t1.add(1.0) << endl;
    cout << endl;

    cout << "3.1.2: sub" << endl;
    cout << "sub(t1, t2) " << endl << ts::sub(t1, t2) << endl;
    cout << "t1.sub(t2): " << endl << t1.sub(t2) << endl;
    cout << "t1 - t2" << endl << t1 - t2 << endl;
    cout << "sub(t1, 1.0)" << endl << ts::sub(t1, 1.0) << endl;
    cout << "t1.sub(1.0)" << endl << t1.sub(1.0) << endl;
    cout << endl;

    cout << "3.1.3: mul" << endl;
    cout << "mul(t1, t2) " << endl << ts::mul(t1, t2) << endl;
    cout << "t1.mul(t2): " << endl << t1.mul(t2) << endl;
    cout << "t1 * t2" << endl << t1 * t2 << endl;
    cout << "mul(t1, 2.0)" << endl << ts::mul(t1, 2.0) << endl;
    cout << "t2.mul(3.0)" << endl << t1.mul(3.0) << endl;
    cout << endl;


    cout << "3.1.4: div" << endl;
    cout << "div(t1, t2) " << endl << ts::div(t1, t2) << endl;
    cout << "t1.div(t2): " << endl << t1.div(t2) << endl;
    cout << "t1 / t2" << endl << t1 / t2 << endl;
    cout << "div(t1, 2.0)" << endl << ts::div(t1, 2.0) << endl;
    cout << "t2.div(3.0)" << endl << t1.div(3.0) << endl;
    cout << endl;


    cout << "3.1.5: log" << endl;
    vector<int> shape3{3,2};
    int data3[3][2]{1, 2,
                       3, 4,
                       5, 6};
    ts::Tensor t3 = ts::tensor(shape3, &data3[0][0]);
    cout << "t3.log() " << endl << t3.log() << endl;
    cout << endl;

    // 3.2 Reduction operations including sum , mean , max , min
    //3.2.1 sum
    cout << "3.2: Reduction operations" << endl;

    vector<int> shape4{3,2};
    double data4[3][2]{0.1, 1.2,
                       2.2, 3.1,
                       4.9, 5.2};
    ts::Tensor t4 = ts::tensor(shape4, &data4[0][0]);
    cout << "3.2.1: sum" << endl;
    cout << "tensor4" << endl << t4 << endl;
    cout << "sum(t4, 0) " << endl << ts::sum(t4,0) << endl;
    cout << "sum(t4, 1) " << endl << ts::sum(t4,1) << endl;
    cout << "t4.sum(0)" << endl << t4.sum(0) << endl;
    cout << "t4.sum(1)" << endl << t4.sum(1) << endl;
    cout << endl;

    cout << "3.2.2: mean" << endl;
    cout << "mean(t4, 0) " << endl << ts::mean(t4,0) << endl;
    cout << "mean(t4, 1) " << endl << ts::mean(t4,1) << endl;
    cout << "t4.mean(0)" << endl << t4.mean(0) << endl;
    cout << "t4.mean(1)" << endl << t4.mean(1) << endl;
    cout << endl;

    cout << "3.2.3: max" << endl;
    cout << "max(t4, 0) " << endl << ts::max(t4,0) << endl;
    cout << "max(t4, 1) " << endl << ts::max(t4,1) << endl;
    cout << "t4.max(0)" << endl << t4.max(0) << endl;
    cout << "t4.max(1)" << endl << t4.max(1) << endl;
    cout << endl;

    cout << "3.2.4: min:" << endl;
    cout << "min(t4, 0) " << endl << ts::min(t4,0) << endl;
    cout << "min(t4, 1) " << endl << ts::min(t4,1) << endl;
    cout << "t4.min(0)" << endl << t4.min(0) << endl;
    cout << "t4.min(1)" << endl << t4.min(1) << endl;
    cout << endl;

    //3.3 Comparison operations including eq , ne , gt , ge , lt , le
    cout << "3.3: Comparison operations" << endl;
    vector<int> shape5{3,2};
    double data5[3][2]{0.1, 1.2,
                       2.2, 3.1,
                       4.9, 5.2};
    ts::Tensor t5 = ts::tensor(shape5, &data5[0][0]);
    vector<int> shape6{3,2};
    double data6[3][2]{0.2, 1.3,
                       2.2, 3.2,
                       4.8, 5.2};
    ts::Tensor t6 = ts::Tensor(shape6, &data6[0][0]);
    ts::Tensor<bool> t7 = t5.eq(t6);
    cout << "3.3.1: eq" << endl;
    cout << "tensor5: " << endl << t5 << endl;
    cout << "tensor6: " << endl << t6 << endl;
    cout << "eq(t5, t6)" << endl << ts::eq(t5, t6) << endl;
    cout << "t5.eq(t6)" << endl << t7 << endl;
    cout << "t5 == t6" << endl << (t5 == t6) << endl;
    cout << endl;

    cout << "3.3.2: ne" << endl;
    cout << "ne(t5, t6)" << endl << ts::ne(t5, t6) << endl;
    cout << "t5.ne(t6)" << endl << t5.ne(t6) << endl;
    cout << "t5 != t6" << endl << (t5 != t6) << endl;
    cout << endl;

    cout << "3.3.3: gt" << endl;
    cout << "gt(t5, t6)" << endl << ts::gt(t5, t6) << endl;
    cout << "t5.gt(t6)" << endl << t5.gt(t6) << endl;
    cout << "t5 > t6" << endl << (t5 > t6) << endl;
    cout << endl;

    cout << "3.3.4: ge" << endl;
    cout << "ge(t5, t6)" << endl << ts::ge(t5, t6) << endl;
    cout << "t5.ge(t6)" << endl << t5.ge(t6) << endl;
    cout << "t5 >= t6" << endl << (t5 >= t6) << endl;
    cout << endl;

    cout << "3.3.5: lt" << endl;
    cout << "lt(t5, t6)" << endl << ts::lt(t5, t6) << endl;
    cout << "t5.lt(t6)" << endl << t5.lt(t6) << endl;
    cout << "t5 < t6" << endl << (t5 < t6) << endl;
    cout << endl;

    cout << "3.3.6: le" << endl;
    cout << "le(t5, t6)" << endl << ts::le(t5, t6) << endl;
    cout << "t5.le(t6)" << endl << t5.le(t6) << endl;
    cout << "t5 <= t6" << endl << (t5 <= t6) << endl;
    cout << endl;

    //3.4 einsum
    //3.4.1 dot product
    cout << "3.4: einsum" << endl;
    cout <<"3.4.1 dot product " << endl;
    cout << "ts::einsum(\"i,i->\", t1, t2);" << endl;

    return 0;
}
#include <iostream>
#include "tensor_aclr_impl.cuh"

using namespace std;

int main() {
    // 3.1 Pointwise operations including add , sub , mul , div , log
    cout << "3.1: Pointwise operations" << endl;
    vector<int> shape1{3,2};
    double *data1 = new double[6]{0.1, 1.2,
                                       2.2, 3.1,
                                       4.9, 5.2};
    ts::Tensor t1 = ts::tensor(shape1, data1);
    vector<int> shape2{3,2};
    double *data2 = new double[6]{0.2, 1.3,
                                   2.3, 3.2,
                                   4.8, 5.1};
    ts::Tensor t2 = ts::tensor(shape2, data2);
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
    cout << "tensor1: " << endl << t1 << endl;
    cout << "tensor2: " << endl << t2 << endl;
    cout << "sub(t1, t2) " << endl << ts::sub(t1, t2) << endl;
    cout << "t1.sub(t2): " << endl << t1.sub(t2) << endl;
    cout << "t1 - t2" << endl << t1 - t2 << endl;
    cout << "sub(t1, 1.0)" << endl << ts::sub(t1, 1.0) << endl;
    cout << "t1.sub(1.0)" << endl << t1.sub(1.0) << endl;
    cout << endl;

    cout << "3.1.3: mul" << endl;
    cout << "tensor1: " << endl << t1 << endl;
    cout << "tensor2: " << endl << t2 << endl;
    cout << "mul(t1, t2) " << endl << ts::mul(t1, t2) << endl;
    cout << "t1.mul(t2): " << endl << t1.mul(t2) << endl;
    cout << "t1 * t2" << endl << t1 * t2 << endl;
    cout << "mul(t1, 2.0)" << endl << ts::mul(t1, 2.0) << endl;
    cout << "t2.mul(3.0)" << endl << t1.mul(3.0) << endl;
    cout << endl;


    cout << "3.1.4: div" << endl;
    cout << "tensor1: " << endl << t1 << endl;
    cout << "tensor2: " << endl << t2 << endl;
    cout << "div(t1, t2) " << endl << ts::div(t1, t2) << endl;
    cout << "t1.div(t2): " << endl << t1.div(t2) << endl;
    cout << "t1 / t2" << endl << t1 / t2 << endl;
    cout << "div(t1, 2.0)" << endl << ts::div(t1, 2.0) << endl;
    cout << "t2.div(3.0)" << endl << t1.div(3.0) << endl;
    cout << endl;


    cout << "3.1.5: log" << endl;
    vector<int> shape3{3,2};
    int *data3 = new int[6]{1, 2,
                       3, 4,
                       5, 6};
    ts::Tensor t3 = ts::tensor(shape3, data3);
    cout << "tensor3: " << endl << t3 << endl;
    cout << "t3.log() " << endl << t3.log() << endl;
    cout << endl;

    // 3.2 Reduction operations including sum , mean , max , min
    //3.2.1 sum
    cout << "3.2: Reduction operations" << endl;

    vector<int> shape4{3,2};
    double *data4 = new double[6]{0.1, 1.2,
                       2.2, 3.1,
                       4.9, 5.2};
    ts::Tensor t4 = ts::tensor(shape4, data4);
    cout << "3.2.1: sum" << endl;
    cout << "tensor4" << endl << t4 << endl;
    cout << "sum(t4, 0) " << endl << ts::sum(t4,0) << endl;
    cout << "sum(t4, 1) " << endl << ts::sum(t4,1) << endl;
    cout << "t4.sum(0)" << endl << t4.sum(0) << endl;
    cout << "t4.sum(1)" << endl << t4.sum(1) << endl;
    cout << endl;

    cout << "3.2.2: mean" << endl;
    cout << "tensor4" << endl << t4 << endl;
    cout << "mean(t4, 0) " << endl << ts::mean(t4,0) << endl;
    cout << "mean(t4, 1) " << endl << ts::mean(t4,1) << endl;
    cout << "t4.mean(0)" << endl << t4.mean(0) << endl;
    cout << "t4.mean(1)" << endl << t4.mean(1) << endl;
    cout << endl;

    cout << "3.2.3: max" << endl;
    cout << "tensor4" << endl << t4 << endl;
    cout << "max(t4, 0) " << endl << ts::max(t4,0) << endl;
    cout << "max(t4, 1) " << endl << ts::max(t4,1) << endl;
    cout << "t4.max(0)" << endl << t4.max(0) << endl;
    cout << "t4.max(1)" << endl << t4.max(1) << endl;
    cout << endl;

    cout << "3.2.4: min:" << endl;
    cout << "tensor4" << endl << t4 << endl;
    cout << "min(t4, 0) " << endl << ts::min(t4,0) << endl;
    cout << "min(t4, 1) " << endl << ts::min(t4,1) << endl;
    cout << "t4.min(0)" << endl << t4.min(0) << endl;
    cout << "t4.min(1)" << endl << t4.min(1) << endl;
    cout << endl;

    //3.3 Comparison operations including eq , ne , gt , ge , lt , le
    cout << "3.3: Comparison operations" << endl;
    vector<int> shape5{3,2};
    double *data5 = new double[6]{0.1, 1.2,
                       2.2, 3.1,
                       4.9, 5.2};
    ts::Tensor t5 = ts::tensor(shape5, data5);
    vector<int> shape6{3,2};
    double *data6 = new double[6]{0.2, 1.3,
                       2.2, 3.2,
                       4.8, 5.2};
    ts::Tensor t6 = ts::Tensor(shape6, data6);
    ts::Tensor<bool> t7 = t5.eq(t6);
    cout << "3.3.1: eq" << endl;
    cout << "tensor5: " << endl << t5 << endl;
    cout << "tensor6: " << endl << t6 << endl;
    cout << "eq(t5, t6)" << endl << ts::eq(t5, t6) << endl;
    cout << "t5.eq(t6)" << endl << t7 << endl;
    cout << "t5 == t6" << endl << (t5 == t6) << endl;
    cout << endl;

    cout << "3.3.2: ne" << endl;
    cout << "tensor5: " << endl << t5 << endl;
    cout << "tensor6: " << endl << t6 << endl;
    cout << "ne(t5, t6)" << endl << ts::ne(t5, t6) << endl;
    cout << "t5.ne(t6)" << endl << t5.ne(t6) << endl;
    cout << "t5 != t6" << endl << (t5 != t6) << endl;
    cout << endl;

    cout << "3.3.3: gt" << endl;
    cout << "tensor5: " << endl << t5 << endl;
    cout << "tensor6: " << endl << t6 << endl;
    cout << "gt(t5, t6)" << endl << ts::gt(t5, t6) << endl;
    cout << "t5.gt(t6)" << endl << t5.gt(t6) << endl;
    cout << "t5 > t6" << endl << (t5 > t6) << endl;
    cout << endl;

    cout << "3.3.4: ge" << endl;
    cout << "tensor5: " << endl << t5 << endl;
    cout << "tensor6: " << endl << t6 << endl;
    cout << "ge(t5, t6)" << endl << ts::ge(t5, t6) << endl;
    cout << "t5.ge(t6)" << endl << t5.ge(t6) << endl;
    cout << "t5 >= t6" << endl << (t5 >= t6) << endl;
    cout << endl;

    cout << "3.3.5: lt" << endl;
    cout << "tensor5: " << endl << t5 << endl;
    cout << "tensor6: " << endl << t6 << endl;
    cout << "lt(t5, t6)" << endl << ts::lt(t5, t6) << endl;
    cout << "t5.lt(t6)" << endl << t5.lt(t6) << endl;
    cout << "t5 < t6" << endl << (t5 < t6) << endl;
    cout << endl;

    cout << "3.3.6: le" << endl;
    cout << "tensor5: " << endl << t5 << endl;
    cout << "tensor6: " << endl << t6 << endl;
    cout << "le(t5, t6)" << endl << ts::le(t5, t6) << endl;
    cout << "t5.le(t6)" << endl << t5.le(t6) << endl;
    cout << "t5 <= t6" << endl << (t5 <= t6) << endl;
    cout << endl;

    //3.4 einsum
    cout << "3.4: einsum" << endl;

    vector<int> shape8{3};
    int *data8 = new int[3]{1, 2, 3};
    ts::Tensor t8 = ts::Tensor(shape8, data8);
    vector<int> shape9{3};
    int *data9 = new int[3]{4, 5, 6};
    ts::Tensor t9 = ts::Tensor(shape9, data9);
    cout << endl;

    //3.4.1 dot product (inner product)
    cout <<"3.4.1: dot product " << endl;
    cout << "tensor8: " << endl << t8 << endl;
    cout << "tensor9: " << endl << t9 << endl;
    cout << "ts::einsum(\"i,i->\", t8, t9);" << endl;
    cout << ts::einsum("i,i->",{t8, t9}) << endl;
    cout << endl;

    //3.4.2 element-wise product
    cout << "3.4.2: computes the element-wise product " << endl;
    cout << "tensor8: " << endl << t8 << endl;
    cout << "tensor9: " << endl << t9 << endl;
    cout << "ts::einsum(\"i,i->i\", t8, t9);" << endl;
    cout << ts::einsum("i,i->i",{t8, t9}) << endl;
    cout << endl;

    //3.4.3 diagonal
    cout << "3.4.3:  diagonal" << endl;
    vector<int> shape10{3,3};
    int *data10 = new int[9]{1, 2, 3,
                             4, 5, 6,
                             7, 8, 9};
    ts::Tensor t10 = ts::Tensor(shape10, data10);
    cout << "tensor10: " << endl << t10 << endl;
    cout <<"ts::einsum(\"ii->i\", t10);" << endl;
    cout << ts::einsum("ii->i", {t10}) << endl;
    cout << endl;

    //3.4.4 permute
    cout << "3.4.4: permute" << endl;
    cout << "tensor10: " << endl << t10 << endl;
    cout << "ts::einsum(\"ij->ji\",t10)" << endl;
    cout << ts::einsum("ij->ji",{t10}) << endl;
    cout << endl;


    //3.4.5 outer product
    cout << "3.4.5: outer product" << endl;
    cout << "tensor8: " << endl << t8 << endl;
    cout << "tensor9: " << endl << t9 << endl;
    cout << "ts::einsum(\"i,j->ij\", t8, t9);" << endl;
    cout << ts::einsum("i,j->ij",{t8, t9}) << endl;
    cout << endl;

    //3.4.6 batch matrix mul
    cout << "3.4.6: Batch Matrix Multiplication" << endl;
    ts::Tensor t11 = ts::rand<int>({2, 3, 2});
    ts::Tensor t12 = ts::rand<int>({2, 2, 3});
    cout << "tensor11: " << endl << t11 << endl;
    cout << "tensor12: " << endl << t12 << endl;
    cout << "ts::einsum(\"bij,bjk->bik\", t11, t12);" << endl;
    cout << ts::einsum("bij,bjk->bik", {t11, t12}) << endl;
    cout << endl;


    //3.4.7 trace
    cout << "3.4.7: Trace" << endl;
    vector<int> shape13{3,3};
    int *data13 = new int[9]{1, 2, 3,
                             4, 5, 6,
                             7, 8, 9};
    ts::Tensor t13 = ts::Tensor(shape13, data13);
    cout << "tensor13: " << endl << t13 << endl;
    cout << "ts::einsum(\"ii\",t13);" << endl;
    cout << ts::einsum("ii",{t13}) << endl;
    cout << endl;

    //3.4.8 sum over an axis
    cout << "3.4.8: sum over an axis" << endl;
    ts::Tensor t14 = ts::rand<int>({2,3,4});
    cout << "tensor14: " << endl << t14 << endl;
    cout << "ts::einsum(\"ijk->jk\",t14)" << endl;
    cout << ts::einsum("ijk->jk",{t14}) << endl;
    cout << endl;


    //3.4.9 Matrix elements multiply and sum
    cout << "3.4.9 matrix elements mul and sum" << endl;
    ts::Tensor t15 = ts::rand<int>({3,3});
    ts::Tensor t16 = ts::rand<int>({3,3});
    cout << "tensor15: " << endl << t15 << endl;
    cout << "tensor16: " << endl << t16 << endl;
    cout << "ts::einsum(\"ij,ij->\",t15,t16)" << endl;
    cout << ts::einsum("ij,ij->",{t15,t16}) << endl;
    cout << endl;


    //3.4.10 Sum of a tensor
    cout << "3.4.10 Sum of a tensor" << endl;
    vector<int> shape17{3,3};
    int *data17 = new int[9]{1, 2, 3,
                             4, 5, 6,
                             7, 8, 9};
    ts::Tensor t17 = ts::Tensor(shape17, data17);
    cout << "tensor17: " << endl << t17 << endl;
    cout << "ts::einsum(\"ij->\",t17)" << endl;
    cout << ts::einsum("ij->",{t17}) << endl;
    cout << endl;

    //3.4.11 Vector Matrix multiplication
    cout << "3.4.11 Vector Matrix multiplication" << endl;
    vector<int> shape18{2,3};
    int *data18 = new int[6]{0,1,2,3,4,5};
    ts::Tensor t18 = ts::Tensor(shape18, data18);
    vector<int> shape19{3};
    int *data19 = new int[3]{0,1,2};
    ts::Tensor t19 = ts::Tensor(shape19, data19);
    cout << "tensor18: " << endl << t18 << endl;
    cout << "tensor19: " << endl << t19 << endl;
    cout << "ts::einsum(\"ik,k->i\",t18,t19)" << endl;
    cout << ts::einsum("ik,k->i",{t18,t19}) << endl;
    cout << endl;


    //3.4.12 Matrix Multiplication
    cout << "3.4.12 Matrix Multiplication" << endl;
    cout << "tensor15: " << endl << t15 << endl;
    cout << "tensor16: " << endl << t16 << endl;
    cout << "ts::einsum(\"ik,kj->ij\",t15,t16)" << endl;
    cout << ts::einsum("ik,kj->ij",{t15,t16}) << endl;
    cout << endl;


    {
        // 3.4 other functions -- dial()
        cout << "3.4 other functions -- dial()" <<endl;
        ts::Tensor test0 = ts::rand<int>({5});
        cout << "input vector: " << endl << test0<<endl;
        cout << "output: " << endl;
        cout << ts::dial(test0) << endl;
        auto *data1 = new double[9]{0, 1, 2, 3,
                                    4, 5, 6, 7,
                                    8};
        vector<int> shape1{3, 3};
        ts::Tensor<double>tb1_1 = ts::tensor(shape1, data1);
        cout << "input n*n matrix: " << endl << tb1_1<<endl;
        cout << "output: " << endl;
        cout << ts::dial(tb1_1);
    }
    {
        // 3.4 other functions -- clone()
        cout << "3.4 other functions -- clone()" <<endl;
        ts::Tensor test0 = ts::rand<int>({5});
        cout << "input vector: " << endl << test0<<endl;
        cout << "clone output: " << endl;
        ts::Tensor test1 = ts::clone(test0);
        cout << test1 << endl;
        test0.data[0]=999;
        cout << "change origin: " << endl << test0<<endl;
        cout << "clone output: " << endl;
        cout << test1 << endl;
    }
    {
        // 3.4 other functions -- flatten()
        cout << "3.4 other functions -- flatten()" <<endl;
        auto *data0 = new double[9]{0, 1, 2, 3,
                                    4, 5, 6, 7,
                                    8};
        vector<int> shape0{3, 3};
        ts::Tensor<double>tb1_1 = ts::tensor(shape0, data0);
        cout << "input tensor: " << endl << tb1_1<<endl;
        cout << "output: " << endl;
        cout << ts::flatten<double>(tb1_1);
    }

    return 0;
}
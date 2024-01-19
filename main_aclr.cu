#include "tensor_aclr_impl.cuh"
//#include <iostream>
//#include <fstream>
//#include <boost/archive/text_oarchive.hpp>
//#include <boost/archive/text_iarchive.hpp>
//
using namespace std;
using namespace ts;
//int main() {
//#include "tensor_aclr_impl.cuh"
//#include <iostream>
//#include <fstream>
//#include <boost/archive/text_oarchive.hpp>
//#include <boost/archive/text_iarchive.hpp>
//
//using namespace std;
//using namespace ts;
//int main() {
//
//
//// 创建一个Tensor对象
//    double *data1 = new double[12]{0, 1, 2, 3,
//                                   4, 5, 6, 7,
//                                   8, 9, 10, 11};
//    vector<int> shape1{3, 4};
//    ts::Tensor t1 = ts::tensor(shape1, data1);
//
//// 序列化Tensor对象
//    {
//        std::ofstream ofs("tensor.txt");
//        boost::archive::text_oarchive oa(ofs);
//        oa << t1;
//    }
//    double *data2 = new double[1]{1};
//    vector<int> shape2{1};
//    ts::Tensor t2 = ts::tensor(shape2, data2);
//    {
//        std::ifstream ifs("tensor.txt");
//        boost::archive::text_iarchive ia(ifs);
//        ia >> t2;
//    }
//    cout<<t1<<endl;
//    cout<<t2;
//    return 0;
//}
//
//// 创建一个Tensor对象
//    double *data1 = new double[12]{0, 1, 2, 3,
//                                   4, 5, 6, 7,
//                                   8, 9, 10, 11};
//    vector<int> shape1{3, 4};
//    ts::Tensor t1 = ts::tensor(shape1, data1);
//
//// 序列化Tensor对象
//    {
//        std::ofstream ofs("tensor.txt");
//        boost::archive::text_oarchive oa(ofs);
//        oa << t1;
//    }
//    double *data2 = new double[1]{1};
//    vector<int> shape2{1};
//    ts::Tensor t2 = ts::tensor(shape2, data2);
//    {
//        std::ifstream ifs("tensor.txt");
//        boost::archive::text_iarchive ia(ifs);
//        ia >> t2;
//    }
//    cout<<t1<<endl;
//    cout<<t2;
//    return 0;
//}
#include <cereal/archives/binary.hpp>
#include <fstream>


int main() {
    // 创建一个Tensor对象
    double *data1 = new double[12]{0, 1, 2, 3,
                                   4, 5, 6, 7,
                                   8, 9, 10, 11};
    vector<int> shape1{3, 4};
    ts::Tensor t1 = ts::tensor(shape1, data1);
    {
        std::ofstream file("tensor.cereal", std::ios::binary);
        cereal::BinaryOutputArchive oarchive(file);
        oarchive(t1);
    }
    ts::Tensor<double> t2;
    {
        std::ifstream file("tensor.cereal", std::ios::binary);
        cereal::BinaryInputArchive iarchive(file);
        iarchive(t2);
    }

    cout<<t2<<endl;
    return 0;
}
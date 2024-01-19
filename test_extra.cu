#include <iostream>
#include "tensor_aclr_impl.cuh"

using namespace std;

int main() {
    ts::acceleration = true;
    // EXTRA 1 printing
    cout << "EXTRA 1 printing" << endl;
    vector<int> shape{20,10,10};
    ts::Tensor t1 = ts::ones<int>(shape);
    ts::Tensor t2 = ts::ones<int>(shape);
    cout<<t1.transpose(0,1)+t2.transpose(0,1)<<endl;

    return 0;
}

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <iostream>

int main() {
    // Initialize host vector with 10 elements
    thrust::host_vector<int> h_vec(10);
    for (int i = 0; i < 10; ++i) {
        h_vec[i] = i + 1;
    }

    // Transfer data to the device
    thrust::device_vector<int> d_vec = h_vec;

    // Perform parallel reduction (sum) on the device
    int sum = thrust::reduce(d_vec.begin(), d_vec.end(), 0, thrust::plus<int>());

    // Print the result
    std::cout << "Sum: " << sum << std::endl;

    return 0;
}

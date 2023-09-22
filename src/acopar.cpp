#include <acopar.hpp>
#include <hip_error.hpp>

#include <vector>

#include <hip/hip_runtime.h>

AntColony::AntColony(ParseCSV *csv) {
    this->csv = csv;
}

void AntColony::run() {
    this->createColony();
}

__global__ void test_mult_numbers(float *d_matrix, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    while(i < size) {
        d_matrix[i] *= 2;
        i += 16383; // GPU block size
    }
}

void test_cpu_mult_numbers(float *matrix, int size) {
    for(int i = 0; i < size; i++)
        matrix[i] *= 2;
}

void AntColony::createColony() {
    Matrix *matrix_ptr = this->getMatrix();
    int matrix_size = matrix_ptr->size();

    test_cpu_mult_numbers(matrix_ptr->data_.data(), matrix_size);
    for(int i = 0; i < matrix_size; i++)
        std::cout << matrix_ptr->data_[i] << " ";

    // Allocating necessary memory on GPU
    float *d_matrix{};
    hipMalloc(&d_matrix, matrix_size * sizeof(float));

    // Copying data from host to device
    float *matrix_elements = matrix_ptr->data_.data();
    HIP_CHECK(hipMemcpy(d_matrix, matrix_elements, matrix_size*sizeof(float), hipMemcpyHostToDevice));

    // Process the information
    test_mult_numbers<<<16, 1024>>>(d_matrix, matrix_size);

    // Getting data back from device
    std::vector<float> d_result(matrix_size);
    HIP_CHECK(hipMemcpy(d_result.data(), d_matrix, matrix_size*sizeof(float), hipMemcpyDeviceToHost));

    // for(int i = 0; i < matrix_size; i++)
    //     std::cout << d_result[i] << " ";

    // Freeing the memory on the GPU
    hipFree(d_matrix);
}

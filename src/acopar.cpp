#include <acopar.hpp>
#include <hip_error.hpp>

#include <vector>

#include <hip/hip_runtime.h>

AntColony::AntColony(ParseCSV *csv) {
    this->csv = csv;
}

void AntColony::run() {
    // Creating pointer thats gonna be used on all GPU computations
    float *d_matrix{};
    int *d_colony_matrix{};

    // Getting the pointer to the matrix that has all the csv data
    Matrix *matrix_ptr = this->getMatrix();

    int WIDTH = matrix_ptr->cols();
    int HEIGHT = matrix_ptr->rows();

    // Getting default values
    this->createColony(matrix_ptr, d_matrix, d_colony_matrix);

    // Getting data back from device
    std::vector<float> d_result(WIDTH*HEIGHT);
    HIP_CHECK(hipMemcpy(d_result.data(), d_matrix, WIDTH*HEIGHT*sizeof(float), hipMemcpyDeviceToHost));

    std::vector<int> d_colony_result(HEIGHT*HEIGHT);
    HIP_CHECK(hipMemcpy(d_colony_result.data(), d_colony_matrix, HEIGHT*HEIGHT*sizeof(int), hipMemcpyDeviceToHost));

    matrix_ptr->data_.clear();
    matrix_ptr->data_ = d_result;

    // Freeing the memory on the GPU
    hipFree(d_matrix);
    hipFree(d_colony_matrix);
}

__global__ void create_colony_matrix(int *d_matrix, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        if (x == y)
            d_matrix[y*width + x] = 1;

        else
            d_matrix[y*width + x] = -1;
    }
}

void AntColony::createColony(Matrix *matrix_ptr, float *d_matrix, int *d_colony_matrix) {
    int WIDTH = matrix_ptr->cols();
    int HEIGHT = matrix_ptr->rows();
    int matrix_size = WIDTH * HEIGHT;

    // Allocating necessary memory on GPU
    HIP_CHECK(hipMalloc(&d_matrix, matrix_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_colony_matrix, HEIGHT * HEIGHT * sizeof(int)));

    // Copying data from host to device
    float *matrix_elements = matrix_ptr->data_.data();
    HIP_CHECK(hipMemcpy(d_matrix, matrix_elements, matrix_size*sizeof(float), hipMemcpyHostToDevice));

    // Process the information
    dim3 block(16, 64);
    dim3 grid((HEIGHT + block.x - 1) / block.x, (HEIGHT + block.y - 1) / block.y);
    hipLaunchKernelGGL(create_colony_matrix, grid, block, 0, 0, d_colony_matrix, HEIGHT, HEIGHT);
    hipDeviceSynchronize();
}

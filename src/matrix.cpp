#include <matrix.hpp>

#include <iostream>

// __global__ void Matrix {
//     // Sets all float numbers inside a matrix to 0.
//   int row = blockIdx.y * blockDim.y + threadIdx.y;
//   int col = blockIdx.x * blockDim.x + threadIdx.x;

//   if (row < rows && col < cols) {
//     this->matrix[row * cols + col] = 0.0f;
//   }
// }

void Matrix::set(int row, int col, float value) {
    if (row < 0 || row >= this->rows() || col < 0 || col >= this->cols())
        std::cerr << "Error: impossible to insert value in this position!\n";

    this->data_[row * cols() + col] = value;
}

void Matrix::print() {
  for (int row = 0; row < this->rows(); row++) {
    for (int col = 0; col < this->cols(); col++) {
      std::cout << (*this)(row, col) << " ";
    }
    std::cout << std::endl;
  }
}

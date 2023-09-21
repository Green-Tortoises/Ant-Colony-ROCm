#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include <hip/hip_runtime.h>

// Efficient in memory Matrix implementation
class Matrix {
    int rows_;
    int cols_;
    std::vector<float> data_;

public:
    Matrix(int rows, int cols) : rows_(rows), cols_(cols), data_(rows * cols) {}
    ~Matrix() {
        this->data_.clear();
        this->data_.shrink_to_fit();
    }

    float& operator()(int row, int col) {
        return data_[row * cols_ + col];
    }

    void set(int row, int col, float value);
    void print();

    int rows() const { return rows_; }
    int cols() const { return cols_; }
};

#endif

#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include <hip/hip_runtime.h>

// Efficient in memory Matrix implementation
class Matrix {
public:
    int rows_;
    int cols_;
    std::vector<float> data_;

    Matrix(int rows, int cols) : rows_(rows), cols_(cols), data_(rows * cols) {}
    ~Matrix() {
        this->data_.clear();
        this->data_.shrink_to_fit();
    }

    float& operator()(int row, int col) {
        return data_[row * cols_ + col];
    }

    void set(int row, int col, float value);
    float get(int row, int col) { return data_[row * cols_ + col]; }
    float get(int pos) { return data_[pos]; }
    void print();

    int rows() const { return rows_; }
    int cols() const { return cols_; }
    int size() const { return rows_ * cols_; }
};

#endif

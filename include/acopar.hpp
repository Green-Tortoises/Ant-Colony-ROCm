#ifndef ACOPAR_HPP
#define ACOPAR_HPP

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <ctime>
#include <vector>
#include <cstdlib>

#include <matrix.hpp>

// Parse and get all the usefull data from the CSV
class ParseCSV {
    std::vector<std::string> header;
    Matrix *matrix;

public:
    ParseCSV(const char *filename);
    ~ParseCSV();

    Matrix *getMatrix() { return this->matrix; }

private:
    void parseHeader(const char *header);
};

// Ant Colony Optmizations code for GPU
class AntColony {
    ParseCSV *csv;

public:
    // Creating an AntColony using a CSV file
    AntColony(ParseCSV *csv);
    void run();

    // Access to the matrix for debugging purposes
    Matrix *getMatrix() { return this->csv->getMatrix(); }

private:
    // Creating colony in-memory
    void createColony();
};

#endif

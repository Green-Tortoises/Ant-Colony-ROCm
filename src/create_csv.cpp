#include <acopar.hpp>

#include <string>
#include <vector>
#include <iostream>

void ParseCSV::DumpCSV(const char *filename) {
    std::ofstream file;
    file.open(filename);

    // Writing the header back to the CSV file
    int i;
    for(i = 0; i < this->header.size()-1; i++)
        file << this->header.at(i) << ",";
    file << this->header.at(i) << "\n";

    // Dumping all the matrix data back to the file
    Matrix *m = this->matrix;
    for(i = 0; i < m->rows(); i++) {
        int j;
        for(j = 0; j < m->cols()-1; j++)
            file << m->get(i, j) << ",";
        file << m->get(i, j) << "\n";
    }
    file << "\n";

    file.close();
}

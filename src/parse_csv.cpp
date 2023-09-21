#include <acopar.hpp>

#include <string>
#include <vector>
#include <iostream>

static std::vector<std::string> parseLine(const char *header) {
    std::istringstream str_header(header); // Converting char * to str buffer
    std::string str;
    std::vector<std::string> elements;

    while(std::getline(str_header, str, ','))
        elements.push_back(str);

    return elements;
}

static int getNumLines(const char *filename) {
    std::ifstream file;
    file.open(filename);

    // Checking if the file opened correctly
    if(!file.is_open()) {
        std::cerr << "Error opening file: " << filename <<"\n";
        exit(1);
    }

    int rows = 0;
    std::string line;
    while(std::getline(file, line))
        rows++;

    file.close();
    return rows;
}

void ParseCSV::parseHeader(const char *header) {
    this->header = parseLine(header);
}

ParseCSV::ParseCSV(const char *filename) {
    // Getting number of lines to generate a static matrix later
    int rows = getNumLines(filename)-1; // do not count header line

    std::ifstream file;
    file.open(filename);

    // Checking if the file opened correctly
    if(!file.is_open()) {
        std::cerr << "Error opening file: " << filename <<"\n";
        exit(1);
    }

    const int line_size = 4096;
    char line[line_size];

    // Saving CSV header
    file.getline(line, line_size);
    this->parseHeader(line);

    int cols = this->header.size();
    this->matrix = new Matrix(rows, cols);

    for(int i = 0; i < rows; i++) {
        // Getting all lines from a file
        std::vector<std::string> str_line;
        file.getline(line, line_size);
        str_line = parseLine(line);

        int j = 0;
        for(std::string str : str_line) {
            this->matrix->set(i, j, std::stof(str));
            j++;
        }
    }

    file.close();
}

ParseCSV::~ParseCSV() {
    this->header.clear();
    delete this->matrix;
}

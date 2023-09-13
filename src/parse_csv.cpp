#include "../acopar.hpp"

#include <string>
#include <vector>
#include <iostream>


ParseCSV::ParseCSV(const char *filename) {
    this->rows = 0;
    std::ifstream file;
    file.open(filename, std::ios_base::in);
    char line[1024];
    char *token;

    // Saving CSV header
    file.getline(line, 1024);
    this->parseHeader(line);

    while(!file.eof()) {
        this->rows++;

        file.getline(line, 1024);
    }

    file.close();
}

void ParseCSV::parseHeader(const char *header) {
    std::istringstream str_header(header); // Converting char * to str buffer
    std::string str;
    std::vector<std::string> elements;

    while(std::getline(str_header, str, ','))
        elements.push_back(str);

    this->header = elements;
}

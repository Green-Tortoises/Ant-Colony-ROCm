#include "../acopar.hpp"

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

std::vector<std::vector<float>> ParseCSV::getLines() {
    return this->lines;
}

void ParseCSV::parseHeader(const char *header) {
    this->header = parseLine(header);
    this->cols = this->header.size();
}

ParseCSV::ParseCSV(const char *filename) {
    this->rows = 0;
    std::ifstream file;
    file.open(filename, std::ios_base::in);
    char line[1024];

    // Saving CSV header
    file.getline(line, 1024);
    this->parseHeader(line);

    while(!file.eof()) {
        // Getting all lines from a file
        std::vector<std::string> str_line;
        file.getline(line, 1024);
        str_line = parseLine(line);

        // Converting all strings in a line to float
        std::vector<float> csvFloats;
        for(std::string i : str_line)
            csvFloats.push_back(std::stof(i));

        this->lines.push_back(csvFloats);
        this->rows++;
    }

    file.close();
}

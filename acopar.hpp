#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <ctime>
#include <vector>
#include <cstdlib>
#include <hip/hip_runtime_api.h>

// Parse and get all the usefull data from the CSV
class ParseCSV {
    int rows;
    int cols;
    std::vector<std::string> header;
    // Lines are made of a line that contains all
    // the float numbers in that CSV line
    std::vector<std::vector<float>> lines;

public:
    ParseCSV(const char *filename);
    std::vector<std::vector<float>> getLines();

private:
    void parseHeader(const char *header);
};

// Ant Colony Optmizations code for GPU
class AntColony {
    ParseCSV *train;
    ParseCSV *test;

public:
    // Creating an AntColony using a CSV file
    AntColony(ParseCSV *train, ParseCSV *test);

};

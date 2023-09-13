#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <ctime>
#include <vector>
// #include <hip/hip_runtime_api.h>

// Parse and get all the usefull data from the CSV
class ParseCSV {
    int rows;
    int cols;
    std::vector<std::string> header;

public:
    ParseCSV(const char *filename);

private:
    void parseHeader(const char *header);
};

// Ant Colony Optmizations code for GPU
class AntColony {
    ParseCSV *parser;

public:
    // Creating an AntColony using a CSV file
    AntColony(const char *filename);

};

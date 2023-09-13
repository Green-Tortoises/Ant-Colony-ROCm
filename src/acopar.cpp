#include "../acopar.hpp"

AntColony::AntColony(const char *filename) {
    this->parser = new ParseCSV(filename);
}

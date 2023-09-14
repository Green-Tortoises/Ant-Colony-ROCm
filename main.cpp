#include "acopar.hpp"

int main(int argc, char **argv) {
    if(argc == 1) {
        std::cout << "Please run this program as " << argv[argc-1] << " path/to/train/dataset path/to/test/dataset!\n";
        return 1;
    }

    if (argc < 3) {
        std::cout << "Please provide path to the train dataset and test dataset.\n";
        return 2;
    }

    // Starting a random seed
    srand(time(NULL));

    ParseCSV *csv_train_parse = new ParseCSV(argv[1]);
    ParseCSV *csv_test_parser = new ParseCSV(argv[2]);
    AntColony *ant = new AntColony(csv_train_parse, csv_test_parser);

    delete csv_train_parse;
    delete csv_test_parser;
    delete ant;
    return 0;
}

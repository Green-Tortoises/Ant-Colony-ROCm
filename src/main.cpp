#include <acopar.hpp>

int main(int argc, char **argv) {
    if(argc == 2) {
        std::cout << "Please run this program as " << argv[argc-1] << " path/to/dataset path/to/resulting_dataset!\n";
        return 1;
    }

    // Starting a random seed
    srand(time(NULL));

    ParseCSV *csv_parse = new ParseCSV(argv[1]);
    AntColony *ant = new AntColony(csv_parse);

    // Running the Colony
    ant->run();

    // Dumping the resulting data to disk
    csv_parse->DumpCSV(argv[2]);

    delete csv_parse;
    delete ant;
    return 0;
}

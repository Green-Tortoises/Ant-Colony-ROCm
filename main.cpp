#include "acopar.hpp"

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("It's missing dataset parameters.\n");
        exit(EXIT_FAILURE);
    }

    clock_t start = time(NULL);

    AntColony *ant = new AntColony(argv[1]);

    clock_t stop = time(NULL);

    std::cout << "This execution took " << stop - start << " seconds!\n";

    return 0;
}

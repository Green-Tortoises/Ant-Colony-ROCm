/*
    Esse algoritmo de Colonia de Formiga implementa a busca
    estocástica sem a vantagem heurística do melhor caminho a ser escolhido.
    Ele foi reestruturado para beneficiar a paralelização. Assim, cada formiga
    escolhe seu conjunto e já roda o knn sobre sua solução. ESTOU TENTANDO ALOCAR
    UM POUCO DE MEMÓRIA DINAMICAMENTE E UM POUCO ESTÁTICA PARA VER SE CONSIGO RODAR
    BASES MAIORES QUE 14000 INSTÂNCIAS.
    A base maior que rodei foi a dos presos que forneceu a saida em 23hs mais ou menos.
*/
#define __HIP_PLATFORM_AMD__

#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <float.h>
#include <time.h>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <rocrand/rocrand_kernel.h>

#define MAX_INSTANCES 15000
#define MAX_ATTR 100
#define NUM_THREADS 12

int NUM_INSTANCES = 0;
int NUM_ATTR = 0;
int NUM_INSTANCES_TST = 0;
int NUM_ATTR_TST = 0;

int *d_the_colony;
int *d_ant_choices;
int *d_last_choices;
int *d_ultimo;
float *d_pheromone_trails;


class Matrix {
public:
    std::vector<std::string> headers;
    size_t num_columns;
    size_t num_rows;
    float *matrix; // Pointer to the GPU matrix

    Matrix() {
        num_columns = 0;
        num_rows = 0;
        matrix = nullptr;
    }

    ~Matrix() {
        hipFree(this->matrix);
        this->headers.clear();
        this->headers.shrink_to_fit();
    }

    inline size_t matrix_size() {
        return this->num_columns*this->num_rows;
    }

    void print() {
        size_t size = this->num_rows*this->num_columns;
        float mtr[size];
        hipMemcpy(mtr, this->matrix, size*sizeof(float), hipMemcpyDeviceToHost);

        std::cout << "CSV Header:\n[ ";
        for (size_t i = 0; i < this->headers.size(); ++i) {
            std::cout << this->headers[i];
            if (i != this->headers.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << " ]\n";

        std::cout << "Values:\n";
        for (size_t i = 0; i < this->num_rows; i++) {
            for (size_t j = 0; j < this->num_columns; j++)
                std::cout << mtr[i*this->num_columns + j] << " ";
            std::cout << "\n";
        }
    }
};

__global__ void _set_colony_diagonal(int *d_the_colony, int size) {
    int x = blockIdx.x*blockDim.x+threadIdx.x;
    int y = blockIdx.y*blockDim.y+threadIdx.y;

    if (x < size && y < size) {
        if (x == y)
            d_the_colony[y*size+x] = 1;
        else
            d_the_colony[y*size+x] = -1;
    }
}

//Cria a colonia n x n
void create_colony() {
    size_t size = NUM_INSTANCES*NUM_INSTANCES;

    hipError_t err;

    err = hipMalloc(&d_the_colony, size*sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Error after hipMalloc: " << hipGetErrorString(err) << std::endl;
        exit(1);
    }

    //Preenche toda a matriz colonia com -1 e
    //posiciona cada formiga em uma instância diferente
    dim3 dimBlock(1);  // e.g., BLOCK_WIDTH and BLOCK_HEIGHT might be 16, 32, etc.
    dim3 dimGrid((NUM_INSTANCES + dimBlock.x - 1) / dimBlock.x,
                 (NUM_INSTANCES + dimBlock.y - 1) / dimBlock.y);

    printf("Pointer to the colony on GPU: %p\nNumber of instances: %d\n", d_the_colony, NUM_INSTANCES);
    hipLaunchKernelGGL(_set_colony_diagonal, dimGrid, dimBlock, 0, 0, d_the_colony, NUM_INSTANCES);

    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error after hipLaunchKernelGGL: " << hipGetErrorString(err) << std::endl;
        exit(1);
    }

    // Sync all GPU threads
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        std::cerr << "Error after hipDeviceSynchronize: " << hipGetErrorString(err) << std::endl;
        exit(1);
    }
}

void read_csv(const char *filename, Matrix *csv) {
    std::ifstream file(filename);
    std::string tokens, token;

     // Check if file opened successfully
    if (!file.is_open()) {
        std::cerr << "Failed to open the file: " << filename << std::endl;
        return;
    }

    // Reading header
    std::getline(file, tokens);
    std::stringstream header(tokens);
    while (std::getline(header, token, ',')) {
        csv->headers.push_back(token);
        csv->num_columns++;  // Renamed from row_size
    }

    // Reading CSV float values
    std::vector<float> values;
    while (std::getline(file, tokens)) {
        // Parsing line into tokens
        std::stringstream line(tokens);
        while (std::getline(line, token, ','))
            values.push_back(std::stof(token));
        csv->num_rows++;
    }

    file.close();

    // Copying the matrix to GPU
    hipMalloc(&csv->matrix, values.size()*sizeof(float));
    hipMemcpy(csv->matrix, values.data(), values.size()*sizeof(float), hipMemcpyHostToDevice);

    std::cout << csv->num_rows << " instances x " << csv->num_columns << " attributes loaded!\n";
}

__global__ void _create_pheromone_trails(float *dvc_pheromone_trails, size_t size, int initial_pheromone) {
    int x = blockIdx.x*blockDim.x+threadIdx.x;
    int y = blockIdx.y*blockDim.y+threadIdx.y;

    if (x < size && y < size) {
        if (x != y)
            dvc_pheromone_trails[y*size+x] = initial_pheromone;
        else
            dvc_pheromone_trails[y*size+x] = 0;
    }
}

float calc_acertos(int* matches, int total) {
    int acertos = 0;

    for (int i = 0; i < total; i++) {
        if( matches[i] == 1)
            acertos++;
    }

    return (float) acertos/total;
}

__device__ float distance(float* instance1, float* instance2, int attributes) {
    float sum_squares = 0.0f;
    for (int k = 0; k < attributes; k++)
        sum_squares += powf(instance1[k] - instance2[k], 2);

    return sqrtf(sum_squares);
}

__device__ void ant_action(int ant, int* the_colony, float* tst_matrix, float* matrix, int* matches, int *matches_completo,
                           int num_inst, int num_inst_test, int num_attr, int num_attr_tst, int random_numbers_seed)
{
    uint tid = blockDim.x * blockIdx.x + threadIdx.x;
	rocrand_state_xorwow state;
	rocrand_init(random_numbers_seed, tid, 0, &state);

    int ajk;
    for (int j = 0; j < num_inst; j++) {
        ajk = rocrand(&state) % 100;

        if (the_colony[ant * num_inst + j] == -1) {
            if (ajk >= 40)
                the_colony[ant * num_inst + j] = 1;
            else
                the_colony[ant * num_inst + j] = 0;
        }
    }

    // KNN process
    for (int inst_tst = 0; inst_tst < num_inst_test; inst_tst++) {
        float menor_distance = FLT_MAX;
        float current_class = -1.0f;

        for (int inst_select = 0; inst_select < num_inst; inst_select++) {
            if (the_colony[ant * num_inst + inst_select] == 1) {
                float dist = distance(&tst_matrix[inst_tst * num_attr_tst], &matrix[inst_select * num_attr], num_attr_tst - 1);
                if (dist < menor_distance) {
                    menor_distance = dist;
                    current_class = matrix[inst_select * num_attr + num_attr - 1];
                }
            }
        }
        if (current_class == tst_matrix[inst_tst * num_attr_tst + num_attr_tst - 1]) {
            matches[num_inst_test*ant + inst_tst] = 1;
        } else {
            matches[num_inst_test*ant + inst_tst] = 0;
        }

        float menor_distance_conjunto_completo = FLT_MAX;
        float current_class_conjunto_completo = -1.0f;

        for (int inst_select = 0; inst_select < num_inst; inst_select++) { //
            float dist_completo = distance(&tst_matrix[inst_tst * num_attr_tst], &matrix[inst_select * num_attr], num_attr_tst - 1);

            if (dist_completo < menor_distance_conjunto_completo) {
                menor_distance_conjunto_completo = dist_completo;
                current_class = matrix[inst_select * num_attr + num_attr - 1];
            }
        }
        if (current_class == tst_matrix[inst_tst * num_attr_tst + num_attr_tst - 1]) {
            matches_completo[num_inst_test*ant + inst_tst] = 1;
        } else {
            matches_completo[num_inst_test*ant + inst_tst] = 0;
        }
    }
}

// Ant colony main kernel function
__global__ void ant_kernel(int* the_colony, float* tst_matrix, float* matrix, int* matches, int* matches_completo, int num_inst,
                           int num_inst_tst, int num_attr, int num_attr_tst, int random_numbers_seed) {
    int ant = threadIdx.x;
    ant_action(ant, the_colony, tst_matrix, matrix, matches, matches_completo, num_inst, num_inst_tst, num_attr, num_attr_tst, random_numbers_seed);
}

static int best_solution_size(int *the_colony, size_t the_colony_size, size_t row_size, int best_ant) {
    int instances_selected = 0;

    std::cout << "Best ant: " << best_ant << ":\n";

    for (int i = 0; i < the_colony_size; i++) {
        if (the_colony[row_size*best_ant + i] == 1) {
            instances_selected++;
            std::cout << the_colony[row_size*best_ant + i] << "  ";
        }
    }

    std::cout << "\n";

    return instances_selected;
}

static int print_accuracy_results(int *matches, size_t size) {
    int bestAnt = -1;
    float bestAccuracy = 0.0f;

    float accuracyI;
    for (int i = 0; i < size; i++) {
        accuracyI = calc_acertos(&matches[i*size], size);

        if (accuracyI > bestAccuracy) {
            bestAccuracy = accuracyI;
            bestAnt = i;
        }
    }

    printf("%f", bestAccuracy);

    return bestAnt;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Usage: %s train_db.csv test_db.csv\n", argv[0]);
        exit(-1);
    }


    clock_t start_time, end_time;
    double total_time;

    srand(time(nullptr));
    start_time = clock();

    Matrix *train = new Matrix();
    Matrix *test  = new Matrix();

    // Reading train and test CSV
    read_csv(argv[1], train);
    read_csv(argv[2], test);

    // Saving values to the variables of the original code author
    NUM_INSTANCES     = train->num_rows;
    NUM_ATTR          = train->num_columns;
    NUM_INSTANCES_TST = test->num_rows;
    NUM_ATTR_TST      = test->num_columns;

    // Creating auxiliar matrix
    hipMalloc(&d_last_choices, MAX_INSTANCES*sizeof(int));
    hipMalloc(&d_ultimo, MAX_INSTANCES*sizeof(int));

    create_colony();
    printf("Colony created in VRAM.\n");

    // Creating the matches matrix all 0 for the GPU results
    int *d_matches;
    hipMalloc(&d_matches, NUM_INSTANCES_TST*NUM_INSTANCES_TST*sizeof(int));
    hipMemset(d_matches, 0, NUM_INSTANCES_TST*NUM_INSTANCES_TST*sizeof(int));

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        printf("Kernel Launch Error: %s\n", hipGetErrorString(err));
    }

    // Setting up kernel launch parameters
    dim3 dimBlock(NUM_INSTANCES);
    dim3 dimGrid(1);

    // Creating a vector containing an ant
    int *d_matches_completo;
    hipMalloc(&d_matches_completo, MAX_INSTANCES*sizeof(int));

    // Creating a random seed
    int seed = rand();

    // Launching the kernel
    hipLaunchKernelGGL(ant_kernel, dimGrid, dimBlock, 0, 0, d_the_colony, test->matrix, train->matrix, d_matches, d_matches_completo,
                       NUM_INSTANCES, NUM_INSTANCES_TST, NUM_ATTR, NUM_ATTR_TST, seed);

    // Synchronizing device
    hipDeviceSynchronize();

    err = hipGetLastError();
    if (err != hipSuccess) {
        printf("Kernel Launch Error: %s\n", hipGetErrorString(err));
    }

    // Retrieving results back to host
    int matches[NUM_INSTANCES_TST*NUM_INSTANCES_TST];
    hipMemcpy(matches, d_matches, NUM_INSTANCES_TST*NUM_INSTANCES_TST*sizeof(int), hipMemcpyDeviceToHost);

    int matches_completo[NUM_INSTANCES_TST*sizeof(int)];
    hipMemcpy(matches_completo, d_matches_completo, NUM_INSTANCES_TST*sizeof(int), hipMemcpyDeviceToHost);


    printf("Accuracy of ");
    int ant = print_accuracy_results(matches, NUM_INSTANCES_TST);
    printf(" got from ant %d\n", ant);

    std::cout << "-------------\n";

    printf("Accuracy of ");
    int default_ant = print_accuracy_results(matches_completo, NUM_INSTANCES_TST);
    printf(" got from the default ant\n");

    std::cout << "-------------\n";

    int h_the_colony[NUM_INSTANCES_TST*NUM_INSTANCES_TST];
    hipMemcpy(h_the_colony, d_the_colony, NUM_INSTANCES_TST*NUM_INSTANCES_TST*sizeof(int), hipMemcpyDeviceToHost);

    int best_size = best_solution_size(h_the_colony, NUM_INSTANCES, train->num_rows, ant);

    std::cout << "Size of the database: " << NUM_INSTANCES << ", size of the best ant solution: " << best_size << "\n";

    // Freeing GPU memory
    hipFree(d_the_colony);
    hipFree(d_ultimo);
    hipFree(d_ant_choices);
    hipFree(d_matches);
    hipFree(d_matches_completo);
    delete train;
    delete test;

    end_time = clock();
    total_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("Total Execution Time: %f seconds\n", total_time);

    return 0;
}

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

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <float.h>
#include <time.h>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#define CSV_MAX_BYTE_SIZE 8192
#define MAX_INSTANCES 15000
#define MAX_ATTR 100
#define NUM_THREADS 12
int NUM_INSTANCES = 0;
int NUM_ATTR = 0;
int NUM_INSTANCES_TST = 0;
int NUM_ATTR_TST = 0;
float INITIAL_PHEROMONE = 1;
float evaporation_rate = 0.1;
float Q = 1.0;

int *d_the_colony;
//int the_colony[MAX_INSTANCES][MAX_INSTANCES];
int *d_ant_choices;
int *d_last_choices;
int *d_ultimo;
int best_solution = 0;
float *d_pheromone_trails;
float *d_matrix; // Contém um ponteiro para a a matrix dentro da GPU
float *d_tst_matrix; //contem as instancias para teste dentro da GPU
int row, col;


__global__ void _set_colony_diagonal(int* dvc_the_colony, size_t size) {
    int x = blockIdx.x*blockDim.x+threadIdx.x;
    int y = blockIdx.y*blockDim.y+threadIdx.y;

    if (x < size && y < size) {
        if (x == y)
            dvc_the_colony[y*size+x] = 1;
        else
            dvc_the_colony[y*size+x] = -1;
    }
}

//Cria a colonia n x n
void create_colony() {
    size_t size = NUM_INSTANCES*NUM_INSTANCES;

    hipMalloc(&d_the_colony, size*sizeof(int));
    //Preenche toda a matriz colonia com -1 e
    //posiciona cada formiga em uma instância diferente
    dim3 dimBlock(32, 32);  // e.g., BLOCK_WIDTH and BLOCK_HEIGHT might be 16, 32, etc.
    dim3 dimGrid((NUM_INSTANCES + dimBlock.x - 1) / dimBlock.x,
                 (NUM_INSTANCES + dimBlock.y - 1) / dimBlock.y);

    hipLaunchKernelGGL(_set_colony_diagonal, dimGrid, dimBlock, 0, 0, d_the_colony, NUM_INSTANCES);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
        std::cerr << "Error: " << hipGetErrorString(err) << std::endl;

    // Sync all GPU threads
    hipDeviceSynchronize();
}

float* check_matrix_size(char* file_name) {
    FILE *fp;
    char line[CSV_MAX_BYTE_SIZE];
    char *token;
    row = 0;
    col = 0;
    fp = fopen(file_name, "r");
    if (fp == NULL) {
        printf("Erro ao abrir o arquivo.\n");
        exit(1);
    }

    while (fgets(line, CSV_MAX_BYTE_SIZE, fp)) {
        col = 0;
        token = strtok(line, ",");
        while (token != NULL) {
            col++;
            token = strtok(NULL, ",");
        }
        row++;
    }
    fclose(fp);

    hipMalloc(&d_matrix, row*col*sizeof(float));

    NUM_INSTANCES = row;
    NUM_ATTR = col;
    return d_matrix;
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

void create_pheromone_trails() {
    // printf("Tentando criar matriz de feromonios\n");
    size_t size = NUM_INSTANCES*NUM_INSTANCES;

    hipMalloc(&d_pheromone_trails, size*sizeof(float));

    dim3 dimBlock(32, 32);  // e.g., BLOCK_WIDTH and BLOCK_HEIGHT might be 16, 32, etc.
    dim3 dimGrid((NUM_INSTANCES + dimBlock.x - 1) / dimBlock.x,
                 (NUM_INSTANCES + dimBlock.y - 1) / dimBlock.y);

    hipLaunchKernelGGL(_create_pheromone_trails, dimGrid, dimBlock, 0, 0,
                       d_pheromone_trails, NUM_INSTANCES, INITIAL_PHEROMONE);

    // Sync all GPU threads
    hipDeviceSynchronize();

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
        std::cerr << "Error: " << hipGetErrorString(err) << std::endl;
}

__global__ void _init(int* d_last_choices, int* d_ultimo, int* d_ant_choices, int* d_the_colony,
                          float* d_pheromone_trails, int numInstances, float initialPheromone) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < numInstances && j < numInstances) {
        if (j == 0) {
            d_ant_choices[i*numInstances+j] = (j == 0) ? i : -1;
        }

        if (i == j) {
            d_the_colony[i*numInstances+j] = 1;
            d_pheromone_trails[i*numInstances+j] = 0;
        } else {
            d_the_colony[i*numInstances+j] = -1;
            d_pheromone_trails[i*numInstances+j] = initialPheromone;
        }

        if(j == 0) {
            d_last_choices[i] = i;
            d_ultimo[i] = 1;
        }
    }
}


void init() {
    dim3 dimBlock(32, 32); // Adjust these values based on device properties and performance profiling
    dim3 dimGrid((NUM_INSTANCES + dimBlock.x - 1) / dimBlock.x,
                 (NUM_INSTANCES + dimBlock.y - 1) / dimBlock.y);

    hipLaunchKernelGGL(_init, dimGrid, dimBlock, 0, 0, d_last_choices, d_ultimo, d_ant_choices,
                       d_the_colony, d_pheromone_trails, NUM_INSTANCES, INITIAL_PHEROMONE);


    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error: " << hipGetErrorString(err) << std::endl;
    }

    hipDeviceSynchronize();
    printf("Program initiated\n");
}

void read_csv(char* file_name, float *d_matr) {
    FILE *fp;
    char line[CSV_MAX_BYTE_SIZE], header[CSV_MAX_BYTE_SIZE];
    char *token;
    int row = 0;
    int col = 0;
    fp = fopen(file_name, "r");
    if (fp == NULL) {
        printf("Erro ao abrir o arquivo.\n");
        exit(1);
    }

    float matrix[NUM_INSTANCES_TST][NUM_ATTR_TST];

    fgets(header, CSV_MAX_BYTE_SIZE, fp); //desprezar o cabecalho
    while (fgets(line, CSV_MAX_BYTE_SIZE, fp)) {
        col = 0;
        token = strtok(line, ",");
        while (token != NULL) {
            matrix[row][col] = atof(token);
            col++;
            token = strtok(NULL, ",");
        }
        row++;
    }
    fclose(fp);

    // Copying the entire matrix to the GPU
    hipMemcpy(&d_matr, &matrix, NUM_INSTANCES_TST*NUM_ATTR_TST*sizeof(float), hipMemcpyHostToDevice);

    NUM_INSTANCES = row;
    NUM_ATTR = col;
    printf("%d instancies x %d atributes loaded!\n", row, col);
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
    for (int k = 0; k < attributes; k++) {
        sum_squares += powf(instance1[k] - instance2[k], 2);
    }
    return sqrtf(sum_squares);
}

__device__ void ant_action(int ant, int* the_colony, float* tst_matrix, float* matrix, int* matches, int num_inst, int num_inst_test, int num_attr, int num_attr_tst) {
    int ajk = threadIdx.x % 100;

    for (int j = 0; j < num_inst; j++) {
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
    }
}

// Ant colony main kernel function
__global__ void ant_kernel(int* the_colony, float* tst_matrix, float* matrix, int* matches, int num_inst, int num_inst_tst, int num_attr, int num_attr_tst) {
    int ant = blockIdx.x * blockDim.x + threadIdx.x;
    ant_action(ant, the_colony, tst_matrix, matrix, matches, num_inst, num_inst_tst, num_attr, num_attr_tst);
}

void print_accuracy_results(int *matches) {
    int bestAnt = -1;
    float bestAccuracy = 0.0f;

    float accuracyI;
    for (int i = 0; i < NUM_INSTANCES_TST; i++) {
        accuracyI = calc_acertos(&matches[i*NUM_INSTANCES_TST], NUM_INSTANCES_TST);

        if (accuracyI > bestAccuracy) {
            bestAccuracy = accuracyI;
            bestAnt = i;
        }
    }

    printf("Best ant: %d with an accuracy of %f\n", bestAnt, bestAccuracy);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Usage: %s train_db.csv test_db.csv\n", argv[0]);
        exit(-1);
    }

    clock_t start_time, end_time;
    double total_time;
    float Q = 1.0;

    srand(time(NULL));
    start_time = clock();

    hipMalloc(&d_ant_choices, MAX_INSTANCES*MAX_INSTANCES*sizeof(int));
    hipMalloc(&d_last_choices, MAX_INSTANCES*sizeof(int));
    hipMalloc(&d_ultimo, MAX_INSTANCES*sizeof(int));

    d_tst_matrix = check_matrix_size(argv[2]); // Read the file, allocate memory for test matrix
    NUM_INSTANCES_TST = NUM_INSTANCES;
    NUM_ATTR_TST = NUM_ATTR;
    printf("Test Matrix: ");
    read_csv(argv[2], d_tst_matrix); // Read the file and load data into the matrix

    d_matrix = check_matrix_size(argv[1]); // Read the file, allocate memory for training matrix
    printf("Training Matrix: ");
    read_csv(argv[1], d_matrix); // Read the file and load data into the matrix

    create_pheromone_trails();
    printf("Pheromone trails created!\n");
    create_colony();
    printf("Colony created in VRAM.\n");
    init();
    printf("All init functions executed\n");

    // Creating the matches matrix all 0 for the GPU results
    int *d_matches;
    hipMalloc(&d_matches, NUM_INSTANCES_TST*NUM_INSTANCES_TST*sizeof(int));
    hipMemset(d_matches, 0, NUM_INSTANCES_TST*NUM_INSTANCES_TST*sizeof(int));

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        printf("Kernel Launch Error: %s\n", hipGetErrorString(err));
    }

    // Setting up kernel launch parameters
    dim3 dimBlock(32, 32);  // e.g., BLOCK_WIDTH and BLOCK_HEIGHT might be 16, 32, etc.
    dim3 dimGrid((NUM_INSTANCES + dimBlock.x - 1) / dimBlock.x,
                 (NUM_INSTANCES + dimBlock.y - 1) / dimBlock.y);

    // Launching the kernel
    hipLaunchKernelGGL(ant_kernel, dimGrid, dimBlock, 0, 0, d_the_colony, d_tst_matrix, d_matrix, d_matches, NUM_INSTANCES, NUM_INSTANCES_TST, NUM_ATTR, NUM_ATTR_TST);

    // Synchronizing device
    hipDeviceSynchronize();

    err = hipGetLastError();
    if (err != hipSuccess) {
        printf("Kernel Launch Error: %s\n", hipGetErrorString(err));
    }

    // Retrieving results back to host
    int matches[NUM_INSTANCES_TST*NUM_INSTANCES_TST];
    hipMemcpy(matches, d_matches, NUM_INSTANCES_TST*NUM_INSTANCES_TST*sizeof(int), hipMemcpyDeviceToHost);

    print_accuracy_results(matches);

    // Freeing GPU memory
    hipFree(d_last_choices);
    hipFree(d_pheromone_trails);
    hipFree(d_the_colony);
    hipFree(d_ultimo);
    hipFree(d_ant_choices);
    hipFree(d_matrix);
    hipFree(d_tst_matrix);
    hipFree(d_matches);

    end_time = clock();
    total_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("Total Execution Time: %f seconds\n", total_time);

    return 0;
}

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
float **matrix; //contem a base de dados de treino
float **tst_matrix; //contem as instancias para teste
int best_solution = 0;
int row, col;
float* d_pheromone_trails;

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

float** check_matrix_size(char* file_name) {
    FILE *fp;
    char line[1024];
    char *token;
    int row = 0;
    int col = 0;
    fp = fopen(file_name, "r");
    if (fp == NULL) {
        printf("Erro ao abrir o arquivo.\n");
        exit(1);
    }

    while (fgets(line, 1024, fp)) {
        col = 0;
        token = strtok(line, ",");
        while (token != NULL) {
            col++;
            token = strtok(NULL, ",");
        }
        row++;
    }
    fclose(fp);
    matrix = (float**) malloc (row * sizeof (float *));
    for (int i = 0; i < row; ++i)
        matrix[i] = (float*) malloc (col * sizeof (float));

    NUM_INSTANCES = row;
    NUM_ATTR = col;
    return matrix;
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

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
        std::cerr << "Error: " << hipGetErrorString(err) << std::endl;

    // Sync all GPU threads
    hipDeviceSynchronize();
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
}

void read_csv(char* file_name, float** matrix) {
    FILE *fp;
    char line[1024], header[2048];
    char *token;
    int row = 0;
    int col = 0;
    fp = fopen(file_name, "r");
    if (fp == NULL) {
        printf("Erro ao abrir o arquivo.\n");
        exit(1);
    }
    fgets(header, 2048, fp); //desprezar o cabecalho
    while (fgets(line, 1024, fp)) {
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
    NUM_INSTANCES = row;
    NUM_ATTR = col;
    printf("%d instancias x %d atributos carregada com sucesso!\n", row, col);
}

/*/Calcula a distancia euclidiana entre cada par de instancias
void get_pairwise_distance2(float** matrix) {
    float sum_squares = 0;

    //float **euclid_distances;
    distances = malloc (NUM_INSTANCES * sizeof (float *));
    for (int i = 0; i < NUM_INSTANCES; ++i) {
        distances[i] = malloc (NUM_INSTANCES * sizeof (float));
    }
    //Para cada instancia i e j, calcule a distancia euclidiana entre elas
    for (int i = 0; i < NUM_INSTANCES; i++) {
        for (int j = 0; j < NUM_INSTANCES; j++) {
            sum_squares = 0;
            for (int k = 0; k < (NUM_ATTR - 1); k++) {//O atributo de classe não entra
                sum_squares += pow(matrix[i][k] - matrix[j][k], 2);
            }
            distances[i][j] = sqrt(sum_squares);
        }
    }
    printf("Distancias calculadas com sucesso!\n");
    //return euclid_distances;
}*/

/*Calcula a visibilidade entre cada par de instancias
void get_visibility_rates_by_distances2() {

    visibilities = malloc (NUM_INSTANCES * sizeof (float *));
    for (int i = 0; i < NUM_INSTANCES; ++i)
        visibilities[i] = malloc (NUM_INSTANCES * sizeof (float));

    //Para cada instancia i e j, calcula a visibilidade = 1/distancia
    for (int i = 0; i < NUM_INSTANCES; i++)
    {
        for (int j = 0; j < NUM_INSTANCES; j++)
        {
            if ( i != j) {
                if (distances[i][j] != 0)
                    visibilities[i][j] = 1.0/distances[i][j];
                else
                    visibilities[i][j] = 0;
            }
        }
    }
    printf("Visibilidades calculadas com sucesso!\n");
    //return visibilities;
}*/

/*/Calcula a distancia euclidiana entre cada par de instancias
void get_pairwise_distance() {
    float sum_squares = 0;
    //Para cada instancia i e j, calcule a distancia euclidiana entre elas
    //#pragma omp target map(from:distances) map(to: matrix)
    //#pragma omp teams distribute parallel for simd private(sum_squares)
    //#pragma omp parallel for private(sum_squares) shared(distances)
    for (int i = 0; i < NUM_INSTANCES; i++) {
        for (int j = 0; j < NUM_INSTANCES; j++) {
            sum_squares = 0;
            for (int k = 0; k < (NUM_ATTR - 1); k++) {//O atributo de classe não entra
                sum_squares += pow(matrix[i][k] - matrix[j][k], 2);
            }
            distances[i][j] = sqrt(sum_squares);
        }
    }
    //printf("Distancias calculadas com sucesso!\n");
}*/

/*Calcula a visibilidade entre cada par de instancias
void get_visibility_rates_by_distances() {
    int i, j;
    //Para cada instancia i e j, calcula a visibilidade = 1/distancia
    //#pragma omp target map(tofrom:visibilities) map(to: distances)
    //#pragma omp teams distribute parallel for simd private(j)
    //#pragma omp parallel for private(i, j) shared(distances, visibilities)
    for (i = 0; i < NUM_INSTANCES; i++) {
        for (j = 0; j < NUM_INSTANCES; j++) {
            if ( i != j) {
                if (distances[i][j] != 0)
                    visibilities[i][j] = 1.0/distances[i][j];
                else
                    visibilities[i][j] = 0;
            }
        }
    }
    //printf("Visibilidades calculadas com sucesso!\n");
}

void swap(int ant, int i, int j) {
    int tmp_instance;
    int tmp_probability;
    tmp_instance = probab_instances[ant][i];
    tmp_probability = probabilities[ant][i];
    probab_instances[ant][i] = probab_instances[ant][j];
    probabilities[ant][i] = probabilities[ant][j];
    probab_instances[ant][j] = tmp_instance;
    probabilities[ant][j] = tmp_probability;
}

// Ordena crescentemente o array de probabilidades com selectionsort
void selectionsort(int ant, int n) {
    int limit = n < 10 ? n : 10;
    //for(int i = 0; i < (NUM_INSTANCES/20); i++) //ordena 5% da qtde instancias base
    for (int i = 0; i < (limit-1); i++) { //coloca as 10 maiores probabilidades na frente
        int posmaior = i;
        for (int j = (i+1); j < n; j++) {
            if(probabilities[ant][posmaior] < probabilities[ant][j]) {
                posmaior = j;
            }
        }
        swap(ant, posmaior, i);
    }
}*/

float calc_acertos(int* matches, int total) {
    int acertos = 0;
    //omp_set_nested(1);
    //// #pragma omp parallel for simd reduction(+:acertos)
    for (int i = 0; i < total; i++) {
        if( matches[i] == 1)
            acertos++;
    }
    return acertos/(1.0*total);
}

int* clear(int* matches, int n) {
    //omp_set_nested(1);
    //// #pragma omp parallel for simd
    for (int i = 0; i < n; i++) {
        matches[i] = 0;
    }
    return matches;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Usage: %s train_db.csv test_db.csv\n", argv[0]);
        exit(-1);
    }

    clock_t start_time, end_time;
    double total_time;
    float Q = 1.0;
    char* file_name = argv[1];
    char* tst_file_name = argv[2];
    int last_choice, ant_pos, next_instance, ajk;
    float probability, final_probability;

    srand(time(NULL));
    start_time = clock();

    tst_matrix = check_matrix_size(tst_file_name); //le arquivo, aloca memoria p matriz teste
    NUM_INSTANCES_TST = NUM_INSTANCES;
    NUM_ATTR_TST = NUM_ATTR;
    printf("Matriz teste: ");
    read_csv(tst_file_name, tst_matrix); //le arquivo e carrega dados na matriz

    matrix = check_matrix_size(file_name); //le arquivo, aloca memoria p matriz treino
    printf("Matriz treino: ");
    read_csv(file_name, matrix); //le arquivo e carrega dados na matriz
    create_pheromone_trails();
    create_colony();
    init();

    int ant = 0, index;
    float bestaccuracy = 0.0;
    float sum_squares = 0, menor_distance, path_smell;
    float distance;

    // Allocating all necessary memory in the GPU
    hipMalloc(&d_last_choices, MAX_INSTANCES*sizeof(int));
    hipMalloc(&d_ultimo, MAX_INSTANCES*sizeof(int));
    hipMalloc(&d_ant_choices, MAX_INSTANCES*MAX_INSTANCES*sizeof(int));

    //Descomente as duas proximas linhas para rodar em GPU
    //// #pragma omp target map(tofrom:bestaccuracy, the_colony, best_solution) map(to:tst_matrix, matrix)
    //// #pragma omp teams distribute parallel for simd private(menor_distance, distance, class, ajk) shared(the_colony, bestaccuracy)
    //Descomente a linha abaixo para setar a qtde de threads
    //omp_set_num_threads(NUM_THREADS);
    //Descomente a linha abaixo para permitir que threads criem outras threads
    //omp_set_nested(1); //esse comando nao deu diferenca no resultado
    //Descomente a linha abaixo para rodar codigo paralelo vetorizado em CPU
    //#pragma omp parallel for simd private(menor_distance, distance, class, ajk) shared(the_colony, bestaccuracy, best_solution)
    for (ant = 0; ant < NUM_INSTANCES; ant++) {
        int* matches = (int*) malloc(NUM_INSTANCES_TST * sizeof(int));

        for (int j = 0; j < NUM_INSTANCES; j++) {
            ajk = rand() % 100;
            if(the_colony[ant][j] == -1) {
                if(ajk >= 40)
                    the_colony[ant][j] = 1;
                else
                    the_colony[ant][j] = 0;
            }
        }
        // Roda o KNN com as instancias de teste para verificar a acurácia da solução da formiga
        for(int inst_tst = 0; inst_tst < NUM_INSTANCES_TST; inst_tst++) { // Para cada instancia de teste
            menor_distance = FLT_MAX;
            for (int inst_select = 0; inst_select < NUM_INSTANCES; inst_select++) // pega o subset de instancias selecionadas
            {
                if(the_colony[ant][inst_select] == 1 ) {//|| the_colony[ant][inst_select] == 0) { //Se a instancia foi selecionada
                    //printf("Inst_selected %d\n", inst_select);
                    sum_squares = 0;
                    for (int k = 0; k < (NUM_ATTR_TST - 1); k++) // Percorre os atributos 1 a 1
                    {
                        sum_squares += pow(tst_matrix[inst_tst][k] - matrix[inst_select][k], 2);
                    }
                    distance = sqrt(sum_squares);
                    if(distance < menor_distance) {
                        //printf("Menor Distance tst %d - knn%d = %f\n", inst_tst, inst_select, distance);
                        menor_distance = distance;
                        class = matrix[inst_select][NUM_ATTR - 1];
                    }
                }
            }
            if(class == tst_matrix[inst_tst][NUM_ATTR_TST-1]) {
                matches[inst_tst] = 1;
            } else matches[inst_tst] = 0;
        }

        float acuracia = calc_acertos(matches, NUM_INSTANCES_TST);
        if(acuracia > bestaccuracy) {
            bestaccuracy = acuracia;
            best_solution = ant;
            printf("Formiga%d acuracia:%f\n", ant, acuracia);
        }

        matches = clear(matches, NUM_INSTANCES_TST);
        free(matches);
    }

    printf("Formiga %d - Acuracia global:%f\n", best_solution, bestaccuracy);
    end_time = clock();
    total_time = ((double) (end_time - start_time)) / CLOCKS_PER_SEC;
    printf("Tempo total de execucao: %f segundos\n", total_time);

    int qtde_selected = 0;
    //#pragma omp target map(tofrom:qtde_selected)
    //#pragma omp teams distribute parallel for reduction(+:qtde_selected)
    //omp_set_num_threads(NUM_THREADS);
    //// #pragma omp parallel for reduction(+:qtde_selected)
    for (int i = 0; i < NUM_INSTANCES; i++) {
        if(the_colony[best_solution][i] == 1)
            qtde_selected++;
    }
    printf("%d",qtde_selected);

    hipFree(d_last_choices);
    hipFree(d_pheromone_trails);
    hipFree(d_the_colony);
    hipFree(d_ultimo);
    hipFree(d_ant_choices);

    return 0;
}

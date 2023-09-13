/*
    Esse algoritmo de Colonia de Formiga implementa a busca
    estocástica sem a vantagem heurística do melhor caminho a ser escolhido.
    Ele foi reestruturado para beneficiar a paralelização. Assim, cada formiga
    escolhe seu conjunto e já roda o knn sobre sua solução. ESTOU TENTANDO ALOCAR
    UM POUCO DE MEMÓRIA DINAMICAMENTE E UM POUCO ESTÁTICA PARA VER SE CONSIGO RODAR
    BASES MAIORES QUE 14000 INSTÂNCIAS.
    A base maior que rodei foi a dos presos que forneceu a saida em 23hs mais ou menos.
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <float.h>
#include <time.h>

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

int **the_colony;
int ant_choices[MAX_INSTANCES][MAX_INSTANCES];
int last_choices[MAX_INSTANCES];
int ultimo[MAX_INSTANCES];
float **matrix; //contem a base de dados de treino
float **tst_matrix; //contem as instancias para teste
int best_solution = 0;
int row, col;

//Cria a colonia n x n
void create_colony() {
    int i, j;

    the_colony = malloc (NUM_INSTANCES * sizeof (int *));
    if(the_colony) {
        for (i = 0; i < NUM_INSTANCES; i++) {
            the_colony[i] = malloc (NUM_INSTANCES * sizeof (int));
            if(the_colony[i] == NULL)
                printf("NÃO CONSEGUIU alocar a memoria necessaria:%d", i);
        }
    }
    //Preenche toda a matriz colonia com -1 e
    //posiciona cada formiga em uma instância diferente
    for (i = 0; i < NUM_INSTANCES; i++)
    {
        for (j = 0; j < NUM_INSTANCES; j++)
        {
            if ( i == j )
                the_colony[i][j] = 1;
            else
                the_colony[i][j] = -1;
        }
    }

    printf("Colonia %dx%d criada com sucesso!\n", NUM_INSTANCES, NUM_INSTANCES);
}

float** check_matrix_size(const char* file_name) {
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
    matrix = malloc (row * sizeof (float *));
    for (int i = 0; i < row; ++i)
        matrix[i] = malloc (col * sizeof (float));

    NUM_INSTANCES = row;
    NUM_ATTR = col;
    return matrix;
}

void init() {
    for (int i = 0; i < NUM_INSTANCES; i++) {
        last_choices[i] = i; //No inicio, a ultima escolha eh o ponto de partida
        ultimo[i] = 1; //marca a posicao onde sera inserida a instancia para onde a formiga vai
    }
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

float calc_acertos(int* matches, int total) {
    int acertos = 0;
    for (int i = 0; i < total; i++) {
        if( matches[i] == 1)
            acertos++;
    }
    return acertos/(1.0*total);
}

int* clear(int* matches, int n) {
    for (int i = 0; i < n; i++) {
        matches[i] = 0;
    }
    return matches;
}

int main() {
    clock_t start_time, end_time;
    double total_time;
    float Q = 1.0;
    char* file_name = "";
    char* tst_file_name = "";
    int last_choice, ant_pos, next_instance, ajk;
    float probability, final_probability;

    start_time = clock();

    tst_matrix = check_matrix_size(tst_file_name); //le arquivo, aloca memoria p matriz teste
    NUM_INSTANCES_TST = NUM_INSTANCES;
    NUM_ATTR_TST = NUM_ATTR;
    printf("Matriz teste: ");
    read_csv(tst_file_name, tst_matrix); //le arquivo e carrega dados na matriz

    matrix = check_matrix_size(file_name); //le arquivo, aloca memoria p matriz treino
    printf("Matriz treino: ");
    read_csv(file_name, matrix); //le arquivo e carrega dados na matriz
    create_colony();
    init();

    int ant = 0, index;
    float bestaccuracy = 0.0;
    float sum_squares = 0, menor_distance, path_smell;
    float distance, class;

    for (ant = 0; ant < NUM_INSTANCES; ant++) {
        srand(time(NULL));
        int* matches = malloc(NUM_INSTANCES_TST * sizeof(int));
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
                if(the_colony[ant][inst_select] == 1 ) {
                    sum_squares = 0;
                    for (int k = 0; k < (NUM_ATTR_TST - 1); k++) // Percorre os atributos 1 a 1
                    {
                        sum_squares += pow(tst_matrix[inst_tst][k] - matrix[inst_select][k], 2);
                    }
                    distance = sqrt(sum_squares);
                    if(distance < menor_distance) {
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
    for (int i = 0; i < NUM_INSTANCES; i++) {
        if(the_colony[best_solution][i] == 1)
            qtde_selected++;
    }
    printf("%d",qtde_selected);

    return 0;
}

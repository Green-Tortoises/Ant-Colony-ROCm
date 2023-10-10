import sys
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from typing import *
import random
import math
import time


def get_pairwise_distance(matrix: np.ndarray) -> np.ndarray:
    return euclidean_distances(matrix)


def get_visibility_rates_by_distances(distances: np.ndarray) -> np.ndarray:
    visibilities = np.zeros(distances.shape)
    for i in range(distances.shape[0]):
        for j in range(distances.shape[1]):
            if i != j:
                if distances[i,j] == 0:
                    visibilities[i, j] = 0
                else:
                    visibilities[i, j] = 1 / distances[i, j]

    return visibilities


def create_colony(num_ants):
    return np.full((num_ants, num_ants), -1)


def create_pheromone_trails(search_space: np.ndarray, initial_pheromone: float) -> np.ndarray:
    trails = np.full(search_space.shape, initial_pheromone, dtype=np.float64)
    np.fill_diagonal(trails, 0)
    return trails


def get_pheromone_deposit(ant_choices: List[Tuple[int, int]], distances: np.ndarray, deposit_factor: float) -> float:
    tour_lenght = 0
    for path in ant_choices:
        tour_lenght += distances[path[0], path[1]]

    if tour_lenght == 0:
        return 0

    if math.isinf(tour_lenght):
        print('deu muito ruim!')

    return deposit_factor / tour_lenght

def get_probabilities_paths_ordered(ant: np.array, visibility_rates: np.array, phe_trails) \
        -> Tuple[Tuple[int, Any]]:
    available_instances = np.nonzero(ant < 0)[0]
    # The pheromones over the available paths
    smell = np.sum(
        phe_trails[available_instances]
        * visibility_rates[available_instances])

    # Calculate the probalilty by avaible instance using
    # the sum of pheromones in rest of tour
    # instanciasPossiveisX2 [0] -> Ins [1] -> prob
    probabilities = np.zeros((len(available_instances), 2))
    for i, available_instance in enumerate(available_instances):
        probabilities[i, 0] = available_instance
        path_smell = phe_trails[available_instance] * \
                     visibility_rates[available_instance]

        if path_smell == 0:
            probabilities[i, 1] = 0
        else:
            probabilities[i, 1] = path_smell / smell

    sorted_probabilities = probabilities[probabilities[:, 1].argsort()][::-1]
    return tuple([(int(i[0]), i[1]) for i in sorted_probabilities])


def get_best_solution(ant_solutions: np.ndarray, X, Y) -> np.array:
    accuracies = np.zeros(ant_solutions.shape[0], dtype=np.float64)
    best_solution = 0
    for i, solution in enumerate(ant_solutions):
        instances_selected = np.nonzero(solution)[0]
        X_train = X[instances_selected, :]
        Y_train = Y[instances_selected]
        classifier_1nn = KNeighborsClassifier(n_neighbors=1).fit(X_train, Y_train)
        Y_pred = classifier_1nn.predict(X)
        accuracy = accuracy_score(Y, Y_pred)
        accuracies[i] = accuracy
        if accuracy > accuracies[best_solution]:
            best_solution = i

    # print(f"The winner is ant {best_solution} with accucarcy {accuracies[best_solution]}")
    return ant_solutions[best_solution]


def run_colony(X, Y, initial_pheromone, evaporarion_rate, Q):
    distances = get_pairwise_distance(X)
    visibility_rates = get_visibility_rates_by_distances(distances)
    the_colony = create_colony(X.shape[0])
    for i in range(X.shape[0]):
        the_colony[i, i] = 1


    ant_choices = [[(i, i)] for i in range(the_colony.shape[0])]
    pheromone_trails = create_pheromone_trails(distances, initial_pheromone)
    while -1 in the_colony:

        # Each ant will choose thier next istance
        for i, ant in enumerate(the_colony):

            if -1 in ant:
                last_choice = ant_choices[i][-1]
                ant_pos = last_choice[1]
                choices = get_probabilities_paths_ordered(
                    ant,
                    visibility_rates[ant_pos, :],
                    pheromone_trails[ant_pos, :])

                for choice in choices:
                    next_instance = choice[0]
                    probability = choice[1]

                    ajk = random.randint(0, 1)
                    final_probability = probability * ajk
                    if final_probability != 0:
                        ant_choices[i].append((ant_pos, next_instance))
                        the_colony[i, next_instance] = 1
                        break
                    else:
                        the_colony[i, next_instance] = 0

        # Ant deposits the pheromones
        for i in range(the_colony.shape[0]):
            ant_deposit = get_pheromone_deposit(ant_choices[i], distances, Q)
            for path in ant_choices[i][1:]:  # Never deposit in pheromone on i == j!
                pheromone_trails[path[0], path[1]] += ant_deposit

        # Pheromones evaporation
        for i in range(pheromone_trails.shape[0]):
            for j in range(pheromone_trails.shape[1]):
                pheromone_trails[i, j] = (1 - evaporarion_rate) * pheromone_trails[i, j]

    instances_selected = np.nonzero(get_best_solution(the_colony, X, Y))[0]
    return instances_selected


def main():
    database = sys.argv[1]
    reduced_database = sys.argv[2]
    columns = sys.argv[3]

    # dataframe = pd.read_csv("databases/ecoli.csv", header=None)
    #last_row = len(dataframe.columns) - 1
    #classes = dataframe[last_row]
    #dataframe = dataframe.drop(columns=[0, last_row])
    #num_instances = len(dataframe.index)
    start_time = time.time()
    original_df = pd.read_csv(database, sep=',')

    dataframe = pd.read_csv(database, sep=',')

    classes = dataframe[columns]
    dataframe = dataframe.drop(columns=[columns])
    initial_pheromone = 1
    Q = 1
    evaporation_rate = 0.1
    print('Starting search')
    indices_selected =  run_colony(dataframe.to_numpy(), classes.to_numpy(),
                                  initial_pheromone, evaporation_rate, Q)
    print('End Search')
    print(len(indices_selected))
    #print(indices_selected)
    reduced_dataframe = original_df.iloc[indices_selected]
    reduced_dataframe.to_csv(reduced_database, index=False)
    print("Execution finished")
    print("--- %s Hours ---" % ((time.time() - start_time)//3600))
    print("--- %s Minutes ---" % ((time.time() - start_time)//60))
    print("--- %s Seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    main()

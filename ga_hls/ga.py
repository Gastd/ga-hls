import numpy as np
import matplotlib.pyplot as plt
from pgraph import UGraph, DGraph
from individual import Individual
# from carpath import CarPath
import statistics
import random
import math

from tqdm import tqdm

CROSSOVER_RATE = 0.95 ## Rate defined by Núnez-Letamendia
MUTATION_RATE = 0.1  ## Rate defined by Núnez-Letamendia
POPULATION_SIZE = 30  ## Must be an EVEN number
GENE_LENGTH = 32
MAX_ALLOWABLE_GENERATIONS = 616 ##Calculated using ALANDER , J. 1992. On optimal population size of genetic algorithms.
NUMBER_OF_PARAMETERS = 17 ## Number of parameters to be evolved
CHROMOSOME_LENGTH = GENE_LENGTH * NUMBER_OF_PARAMETERS
CHROMOSOME_TO_PRESERVE = 4            ## Must be an EVEN number
PARENTS_TO_BE_CHOSEN = 10

class GA(object):
    """docstring for GA"""
    def __init__(self, graph: UGraph, size = 0, ncars: int = 0, positions: list = []):
        super(GA, self).__init__()

        random.seed()

        self.graph = graph
        self.size = size
        self.ncars = ncars
        self.pos = positions
        self.population = []
        self.generation_counter = 0
        self.checks = random.randrange(4) + 1 ## 1 to 5 checkpoints
        print(f'{self.checks} checkpoints selected.')

        self.init_population()

    def init_population(self):
        for i in range(0, self.size):
            solution = Individual(self.graph, self.ncars, self.pos, self.checks)
            self.population.append(solution)
        print("Population initialized. Size = {}".format(self.size))

    def show(self):
        self.fitness()
        self.population.sort(key=lambda x: x.fitness)
        print(f"Best solution so far have fitness = {self.population[0].fitness}")
        print(f"Best solution has {self.count_distinct_edges(self.population[0])} against all {len(self.graph.edges())} edges")
        print("Best solution = ", self.population[0])
        print("Paths")
        for i in range(self.ncars):
            car = self.population[0].paths[i]
            print('->'.join([str(x.name) for x in car.path]))

    def get_best(self):
        return self.population[0]

    def pool(self):
        return self.population[ random.randint(0, 32767) % PARENTS_TO_BE_CHOSEN]

    def check_evolution(self):
        evolved = self.count_distinct_edges(self.population[0]) >= len(self.graph.edges())
        # max_allowed = self.generation_counter < MAX_ALLOWABLE_GENERATIONS
        # print(f"{self.count_distinct_edges(self.population[0])} < {len(self.graph.edges())} = {evolved}")
        return (evolved)

    def evolve(self):
        # loop
        self.generation_counter = 0
        for i in tqdm(range(MAX_ALLOWABLE_GENERATIONS)):
            ## score population
            self.fitness()
            ## retain elite
            self.population.sort(key=lambda x: x.fitness)
            new_population = self.population[:4]

            # select parents
            parents = new_population
            # perform crossover and mutation
            offsprings_size = int((self.size - len(new_population))/2)
            for i in range(offsprings_size):
                p1 = self.pool()
                p2 = self.pool()
                offsprings = self.crossover(p1, p2)
                self.mutate(offsprings[0], MUTATION_RATE)
                self.mutate(offsprings[1], MUTATION_RATE)
                new_population.append(offsprings[0])
                new_population.append(offsprings[1])

            for i in range(self.size - len(new_population)):
                p1 = self.pool()
                offspring = self.mutate(p1)
                new_population.append(offspring)
            self.population = new_population
            self.generation_counter += 1
            if(self.check_evolution()):
                break
        print('Found solution!')
        # print(f"Best solution so far have fitness = {self.population[0].fitness}")

    def count_distinct_edges(self, solution):
        w, h = self.graph.n, self.graph.n
        adj_mtx = [[0 for x in range(w)] for y in range(h)]
        for k in range(0, self.ncars):
            car = solution.paths[k]
            for i in range(0, len(car.path)-1):
                if int(car.path[i].name) < int(car.path[i+1].name):
                    adj_mtx[int(car.path[i].name)-1][int(car.path[i+1].name)-1] = 1
                else:
                    adj_mtx[int(car.path[i+1].name)-1][int(car.path[i].name)-1] = 1

        return np.sum(adj_mtx)

    def fitness(self):
        for solution in self.population:
            ## Different edges count
            w, h = self.graph.n, self.graph.n
            adj_mtx = [[0 for x in range(w)] for y in range(h)]
            for k in range(0, self.ncars):
                car = solution.paths[k]
                for i in range(0, len(car.path)-1):
                    if int(car.path[i].name) < int(car.path[i+1].name):
                        adj_mtx[int(car.path[i].name)-1][int(car.path[i+1].name)-1] = 1
                    else:
                        adj_mtx[int(car.path[i+1].name)-1][int(car.path[i].name)-1] = 1

            distinct_factor = self.count_distinct_edges(solution) / len(self.graph.edges())

            ## Standard deviation between paths
            paths_lengths = [len(car.path) for car in solution.paths]
            stddev_path = 0.0
            if (len(paths_lengths) != 1):
                stddev_path = statistics.stdev(paths_lengths)

            ## total path length
            total_len = 0.0
            for k in range(0, self.ncars):
                car = solution.paths[k]
                for i in range(len(car.path)-1):
                    dx = car.path[i+1].coord[0] - car.path[i].coord[0]
                    dy = car.path[i+1].coord[1] - car.path[i].coord[1]
                    dist = math.sqrt(dx**2 + dy**2)
                    total_len += dist

            solution.fitness = 20*(1 - distinct_factor) + stddev_path + 0.01*total_len

    def mutate(self, parent: Individual, rate: float):
        g = parent.graph
        ncars = parent.ncars
        pos = parent.positions
        offspring = Individual(g, ncars, pos)
        offspring.paths.clear()
        for i in range(0, parent.ncars):
            car = parent.paths[i]
            if ((random.random() > rate) or (len(car.path) <= 1)):
                offs_path = CarPath(car.init_pos)
                offs_path.set_path(car.path)
                offspring.paths.append(offs_path)
                continue

            try:
                point = random.randrange(len(car.path)-2) + 1
            except:
                point = 1
            fst_half = car.path[:point]
            new_point = str(random.randrange(parent.graph.n)+1)
            snd_half = car.path[point + 1:]

            path = fst_half + g.path_Astar(fst_half[-1].name, new_point)[1:-1]
            if(len(snd_half) != 0):
                path += g.path_Astar(new_point, snd_half[0])[0:-1] + snd_half

            offs_path = CarPath(car.init_pos)
            offs_path.set_path(path)
            offspring.paths.append(offs_path)

        return offspring

    def crossover(self, parent1: Individual, parent2: Individual):
        g = parent1.graph
        ncars = parent1.ncars
        pos = parent1.positions
        offsprings = [
                        Individual(g, ncars, pos),
                        Individual(g, ncars, pos)
                     ]
        for o in offsprings:
            o.paths.clear()

        for i in range(0, parent1.ncars):

            car1 = parent1.paths[i]

            try:
                point = random.randrange(len(car1.path)-2) + 1
            except:
                point = 1

            car1fst_half = car1.path[:point]
            car1snd_half = car1.path[point + 1:]

            car2 = parent2.paths[i]
            try:
                point = random.randrange(len(car2.path)-2) + 1
            except:
                point = 1

            car2fst_half = car2.path[:point]
            car2snd_half = car2.path[point + 1:]

            new_p = self.merge_paths(g, car1fst_half, car2snd_half)

            offs_path = CarPath(car1.init_pos)
            offs_path.set_path(new_p)
            offsprings[0].paths.append(offs_path)

            new_p = self.merge_paths(g, car2fst_half, car1snd_half)

            offs_path = CarPath(car2.init_pos)
            offs_path.set_path(new_p)
            offsprings[1].paths.append(offs_path)

        return offsprings

    def merge_paths(self, graph, path1: list, path2: list):
        if((len(path1) == 0) or (len(path2) == 0)):
            return path1 + path2
        elif(path1[-1].name == path2[0].name):
            return path1 + path2[1:]
        else:
            p = graph.path_Astar(path1[-1].name, path2[0].name)
            return path1 + p[1:-1] + path2

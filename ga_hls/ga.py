import os
import math
import time
import copy
import shlex
import random
import datetime
# import statistics
import subprocess

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import treenode
import individual

CROSSOVER_RATE = 0.95 ## Rate defined by Núnez-Letamendia
MUTATION_RATE = 0.1  ## Rate defined by Núnez-Letamendia
POPULATION_SIZE = 30  ## Must be an EVEN number
GENE_LENGTH = 32
MAX_ALLOWABLE_GENERATIONS = 30 #616 ##Calculated using ALANDER , J. 1992. On optimal population size of genetic algorithms.
# MAX_ALLOWABLE_GENERATIONS = 1 #616 ##Calculated using ALANDER , J. 1992. On optimal population size of genetic algorithms.
NUMBER_OF_PARAMETERS = 17 ## Number of parameters to be evolved
CHROMOSOME_LENGTH = GENE_LENGTH * NUMBER_OF_PARAMETERS
CHROMOSOME_TO_PRESERVE = 4            ## Must be an EVEN number
PARENTS_TO_BE_CHOSEN = 10

class GA(object):
    """docstring for GA"""
    def __init__(self, init_form):
        super(GA, self).__init__()

        random.seed()
        self.size = POPULATION_SIZE
        self.population = []
        self.now = datetime.datetime.now()
        curr_path = os.getcwd()

        self.init_population(init_form)
        self.init_log(curr_path)
        self.s, self.e = self.get_line('ga_hls')
        print(f's = {self.s}, e = {self.e}')

    def init_population(self, init_form):
        root = treenode.parse(init_form)
        terminators = list(set(treenode.get_terminators(root)))
        print(f'terminators = {terminators}')
        print(f'Initial formula: {root}')
        for i in range(0, self.size):
            solution = copy.deepcopy(individual.Individual(root, terminators))
            n = random.randrange(len(root))
            solution.mutate(1, n)
            self.population.append(copy.deepcopy(solution))
        print("Population initialized. Size = {}".format(self.size))

    def init_log(self, parent_dir):
        directory = str(self.now)
        self.path = os.path.join(parent_dir, directory)
        try:
            os.mkdir(self.path)
            pass
        except OSError as error:
            print(error)

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

    def get_line(self, file):
        print(f'Running on {os.getcwd()} folder')
        file_path = 'ga_hls/prop_ctrl.py'
        # f = open("ga_hls/prop_ctrl.py")
        newf_str = ''
        with open(file_path) as f:
            for l in f:
                # print(l)
                if l.find('z3solver.check') > 0:
                    break
                else:
                    newf_str += l
            print(f'py file seek at = {len(newf_str)}')
            d1 = newf_str.rfind('\t')
            d2 = newf_str[1:d1-1].rfind('\t')
            print(f'py file seek at = {d2}')
            self.first = newf_str
            return d2, newf_str.rfind('\n')

    def save_file(self, s, e, nline):
        src = 'ga_hls/prop_ctrl.py'
        dst = 'ga_hls/temp.py'
        with open(src) as firstfile, open(dst,'w') as secondfile:
            firstfile.seek(e)
            secondfile.write(self.first[:s])
            secondfile.write('\n\n')
            secondfile.write(f'\tz3solver.add({nline})\n')
            for l in firstfile:
                secondfile.write(l)

    def get_best(self):
        return self.population[0]

    def write_population(self, generation):
        with open('{}/{:0>2}.txt'.format(self.path, generation), 'w') as f:
            f.write('Formula\tFitness\tSatisfied\n')
            for i in self.population:
                f.write(str(i))
                f.write(f'\t{i.fitness}')
                f.write(f'\t{i.madeit}')
                f.write('\n')

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
            new_population = self.population[CHROMOSOME_TO_PRESERVE:]

            population_counter = CHROMOSOME_TO_PRESERVE
            while(population_counter < POPULATION_SIZE):
                # print(f'gen {self.generation_counter} / ind {population_counter}')
                offspring1 = self.pool()
                offspring2 = self.pool()

                self.crossover(offspring1, offspring2)
                offspring1.mutate(MUTATION_RATE)
                offspring2.mutate(MUTATION_RATE)
                new_population.append(offspring1)
                new_population.append(offspring2)
                population_counter += 2
            self.generation_counter += 1
            self.write_population(self.generation_counter)

    def fitness(self):
        for i in self.population:
            # print(i.format())
            self.save_file(self.s, self.e, i.format())

            # folder_name = os.getcwd()[os.getcwd().rfind('/')+1:]
            folder_name = 'ga_hls'
            # print(f'Running on {os.getcwd()} folder')
            run_str = f'python3 {folder_name}/demo.py'
            run_tk = shlex.split(run_str)
            run_process = subprocess.run(run_tk,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             universal_newlines=True)
            # print(run_process.stdout.find('SATISFIED'))
            if run_process.stdout.find('SATISFIED') > 0:
                i.fitness = (1+random.randrange(10))/10
                i.madeit = True
            else:
                i.fitness = (1+random.randrange(10))*10
                i.madeit = False
            # print(run_process.stderr)
            # time.sleep(1)

    def crossover(self, parent1: individual.Individual, parent2: individual.Individual):
        offsprings = [parent1, parent2]

        off0_tree, off0_sub, node0 = offsprings[0].root.cut_tree_random()
        # print(f'off0_tree = {off0_tree}')
        # print(f'off0_sub = {off0_sub}')
        # print(f'node0 = {node0}')
        off1_tree, off1_sub, node1 = offsprings[1].root.cut_tree_random()
        # print(f'off1_tree = {off1_tree}')
        # print(f'off1_sub = {off1_sub}')
        # print(f'node1 = {node1}')

        node0 += off1_sub
        node1 += off0_sub
        # print(f'off0_tree = {off0_tree}')
        # print(f'off1_tree = {off1_tree}')

        return offsprings

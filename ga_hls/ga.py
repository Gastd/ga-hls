import os
import math
import time
import json
import shlex
import random
import datetime
# import statistics
import subprocess

import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy
from tqdm import tqdm

from components import treenode
from components import individual
from components.helper import *

from analysis.smith_waterman import Smith_Waterman, SW_Result
from analysis.dendogram import create_dendrogram

from components.individual import QUANTIFIERS, RELATIONALS, EQUAL, ARITHMETICS, MULDIV, EXP, LOGICALS, NEG, IMP

# from anytree import Node
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.cluster.hierarchy import dendrogram, linkage
# from timeit import default_timer as timer

import matplotlib
# import numpy
# import sys

# import analyse
# from analyse import Smith_Waterman

from config.ga_params import *

def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

class GA(object):
    """docstring for GA"""
    def __init__(self, property_str, trace_file):
        super(GA, self).__init__()

        random.seed()
        self.size = POPULATION_SIZE
        self.population = []
        self.now = datetime.datetime.now()
        curr_path = os.getcwd()

        self.trace_file = "ga_hls/config/" + trace_file

        self.init_population(property_str)
        self.init_log(curr_path)
        self.first, self.s, self.e = get_line(file= self.trace_file)
        self.execution_report = {'TOTAL': 0}

        self.SW = Smith_Waterman()
        self.get_max_score()
        self.check_highest_sat(None)
        self.hypots = []
        self.sats = []
        self.unsats = []
        # self.diag = diagnosis.Diagnosis()
        # self.seed = treenode.parse(json.loads('["ForAll",[["s"],["Implies",[["And",[[">",["s",0]],["<",["s",10]]]],["And",[["<",["signal_4(s)",1000]],[">=",["signal_2(s)",-15.27]]]]]]]]'))

    def check_highest_sat(self, chromosome):
        if chromosome is None:
            self.highest_sat = None
        else:
            if self.highest_sat is None:
                self.highest_sat = chromosome
                self.max_score += self.SW.compare(list(self.seed), list(self.highest_sat), 0,0).traceback_score
            else:
                result_old = self.SW.compare(list(self.seed), list(self.highest_sat), 0,0)
                result_new = self.SW.compare(list(self.seed), list(chromosome), 0,0)
                if result_old < result_new:
                    self.max_score -= result_old.traceback_score
                    self.highest_sat = chromosome
                    self.max_score += result_new.traceback_score

    def get_max_score(self):
        result = self.SW.compare(self.replace_token(list(self.seed)), self.replace_token(list(self.seed)), 0,0)
        self.max_score = result.traceback_score

    def init_population(self, property_str):
        root = treenode.parse(property_str)
        self.seed = root
        terminators = list(set(treenode.get_terminators(root)))
        self.seed_ch = deepcopy(individual.Individual(root, terminators,self.trace_file))
        print(f'terminators = {terminators}')
        print(f'Initial formula: {root}')
        for i in tqdm(range(0, self.size)):
            chromosome = deepcopy(individual.Individual(root, terminators,self.trace_file))
            n = random.randrange(len(root))
            chromosome.mutate(1, n)
            print(f"{i}: chromosome {chromosome} is {'viable' if chromosome.is_viable() else 'not viable'}")
            while not chromosome.is_viable():
                chromosome = deepcopy(individual.Individual(root, terminators,self.trace_file))
                chromosome.mutate(1, random.randrange(len(chromosome)))
                print(f"{i}: chromosome {chromosome} is {'viable' if chromosome.is_viable() else 'not viable'}")
            self.population.append(deepcopy(chromosome))
        # raise Exception('')
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
        self.evaluate()
        self.population.sort(key=lambda x: x.fitness)
        print(f"Best solution so far have fitness = {self.population[0].fitness}")
        print(f"Best solution has {self.count_distinct_edges(self.population[0])} against all {len(self.graph.edges())} edges")
        print("Best solution = ", self.population[0])
        print("Paths")
        for i in range(self.ncars):
            car = self.population[0].paths[i]
            print('->'.join([str(x.name) for x in car.path]))

    def set_sat_check(self, chromosome):
        pass

    def test_chromosome(self, chromosome):
        # print(f'writing test for: {str(chromosome)}')
        # def find_traces_in_file(file):
        #     print(f'Running on {os.getcwd()} folder')
        #     print(f'Running on {file} folder')
        #     newf_str1 = ''
        #     newf_str2 = ''
        #     with open(file) as f:
        #         for l in f:
        #             if l.find('z3solver.check') > 0:
        #                 break
        #             else:
        #                 newf_str2 += l
        #         f.seek(0, 0)
        #         for l in f:
        #             if l.find('z3solver.add') > 0:
        #                 break
        #             else:
        #                 newf_str1 += l
        #         print(f'py file seek at = {len(newf_str1)}')
        #         d1 = newf_str2.rfind('\n')
        #         d2 = newf_str2[1:d1-1].rfind('\t')
        #         print(f'py file seek at = {d2}')
        #         self.first = newf_str1
        #         return newf_str1.rfind('\n'), d2
        # s, e = find_traces_in_file()
        # save_z3check(s, e, f'Not({chromosome.format()})')

        start, end, lines = get_file_w_traces(self.trace_file)
        save_check_wo_traces(start, end, lines, f'Not({chromosome.format()})')
        f = open(self.trace_file, 'r')
        f.seek(0, 0)
        f.close()

        folder_name = 'ga_hls'
        run_str = f'python3 {folder_name}/z3check.py'
        run_tk = shlex.split(run_str)
        run_process = subprocess.run(run_tk,
                                     stderr=subprocess.PIPE,
                                     stdout=subprocess.PIPE,
                                     universal_newlines=True,
                                     timeout=10)
        # print(run_process.stdout)
        if run_process.stdout.find('SATISFIED') > 0:
            # print('Chromosome not viable')
            return False
        elif run_process.stdout.find('VIOLATED') > 0:
            # print('Chromosome viable')
            return True
        else:
            # print('Chromosome not viable')
            return False
        return

    def get_best(self):
        return self.population[0]

    def clusterize(self, generation):
        # this method applies the SW algorithm to all formulas
        results = []
        score_matrix = [[0 for x in range(self.size)] for y in range(self.size)]
        labels = []
        SW = Smith_Waterman()
        for idx, el in enumerate(self.population):
            labels.append('{:0>2}{}'.format(idx,'T' if el.madeit else 'F'))
        for i, c1 in enumerate(self.population):
            for j, c2 in zip(range(i+1, self.size), self.population[i+1:]):
                new_result = SW.compare(list(c1), list(c2), i, j)
                score_matrix[j][i] = score_matrix[i][j] = new_result.traceback_score
                results.append(new_result)
        # return list(sorted(results, reverse = True)), score_matrix

        hypot = [float('inf'), '']
        ## write dendrogram
        Z = create_dendrogram('{}/sw_ga_{:0>2}'.format(self.path, generation), score_matrix, labels, inverse_score=True)
        if Z is not None:
            for idx, el in enumerate(self.population):
                if el.madeit is True: #and Z[idx-1][2] < 0.0333:
                    print(f'{idx}: {Z[idx-1][2]}: {el}')
                    self.hypots.append([Z[idx-1][2], el])
                    # if hypot[0] > Z[idx-1][2]:
                    #     hypot = [Z[idx-1][2], el]
            # self.hypots.append(hypot)

    def generate_dataset_qty(self):
        res = []
        [res.append(x) for x in self.population if x not in res]
        [self.unsats.append(x) for x in self.population if (x not in self.unsats) and (x.madeit == False)]
        [self.sats.append(x) for x in self.population if (x not in self.sats) and (x.madeit == True)]

    def generate_dataset_threshold(self):
        res = []
        [res.append(x) for x in self.population if x not in res]
        [self.unsats.append(x) for x in self.population if (x not in self.unsats) and (x.madeit == False)]
        [self.sats.append(x) for x in self.population if (x not in self.sats) and (x.sw_score > SW_THRESHOLD) and (x.madeit == True)]
        # print(f'\nWe have so far {len(self.sats)} satisfied')
        # print(f'and {len(self.unsats)} unsatisfied')

    def build_attributes(self, formulae: list):
        count_op = {
            'QUANTIFIERS': 0,
            'RELATIONALS': 0,
            'EQUAL': 0,
            'ARITHMETICS': 0,
            'MULDIV': 0,
            'EXP': 0,
            'LOGICALS': 0,
            'NEG': 0,
            'IMP': 0,
            'NUM': 0,
            'SINGALS': 0,
            'TERM': 0
        }
        terminators = list(set(treenode.get_terminators(self.seed)))
        terminators = [value for value in terminators if not isinstance(value, int) and not isinstance(value, float)]
        ret = []
        for term in formulae:
            if term in QUANTIFIERS:
                qstring = str(QUANTIFIERS)
                qstring = qstring.replace('\'', '')
                qstring = qstring.replace(']', '}')
                qstring = qstring.replace('[', '{')
                ret.append(f'QUANTIFIERS{count_op["QUANTIFIERS"]} {qstring}')
                count_op['QUANTIFIERS'] = count_op['QUANTIFIERS'] + 1
            if term in RELATIONALS:
                qstring = str(RELATIONALS)
                qstring = qstring.replace('\'', '')
                qstring = qstring.replace(']', '}')
                qstring = qstring.replace('[', '{')
                ret.append(f'RELATIONALS{count_op["RELATIONALS"]} {qstring}')
                count_op['RELATIONALS'] = count_op['RELATIONALS'] + 1
            if term in EQUAL:
                qstring = str(EQUAL)
                qstring = qstring.replace('\'', '')
                qstring = qstring.replace(']', '}')
                qstring = qstring.replace('[', '{')
                ret.append(f'EQUAL{count_op["EQUAL"]} {qstring}')
                count_op['EQUAL'] = count_op['EQUAL'] + 1
            if term in ARITHMETICS:
                qstring = str(ARITHMETICS)
                qstring = qstring.replace('\'', '')
                qstring = qstring.replace(']', '}')
                qstring = qstring.replace('[', '{')
                ret.append(f'ARITHMETICS{count_op["ARITHMETICS"]} {qstring}')
                count_op['ARITHMETICS'] = count_op['ARITHMETICS'] + 1
            if term in MULDIV:
                qstring = str(MULDIV)
                qstring = qstring.replace('\'', '')
                qstring = qstring.replace(']', '}')
                qstring = qstring.replace('[', '{')
                ret.append(f'MULDIV{count_op["MULDIV"]} {qstring}')
                count_op['MULDIV'] = count_op['MULDIV'] + 1
            if term in EXP:
                qstring = str(EXP)
                qstring = qstring.replace('\'', '')
                qstring = qstring.replace(']', '}')
                qstring = qstring.replace('[', '{')
                ret.append(f'EXP{count_op["EXP"]} {qstring}')
                count_op['EXP'] = count_op['EXP'] + 1
            if term in LOGICALS:
                qstring = str(LOGICALS)
                qstring = qstring.replace('\'', '')
                qstring = qstring.replace(']', '}')
                qstring = qstring.replace('[', '{')
                ret.append(f'LOGICALS{count_op["LOGICALS"]} {qstring}')
                count_op['LOGICALS'] = count_op['LOGICALS'] + 1
            if term in NEG:
                qstring = str(NEG)
                qstring = qstring.replace('\'', '')
                qstring = qstring.replace(']', '}')
                qstring = qstring.replace('[', '{')
                ret.append(f'NEG{count_op["NEG"]} {qstring}')
                count_op['NEG'] = count_op['NEG'] + 1
            if term in IMP:
                qstring = str(IMP)
                qstring = qstring.replace('\'', '')
                qstring = qstring.replace(']', '}')
                qstring = qstring.replace('[', '{')
                ret.append(f'IMP{count_op["IMP"]} {qstring}')
                count_op['IMP'] = count_op['IMP'] + 1
            if term in terminators:
                qstring = str(terminators)
                qstring = qstring.replace('\'', '')
                qstring = '{'+qstring[1:-1]+'}'
                ret.append(f'TERM{count_op["TERM"]} {qstring}')
                count_op['TERM'] = count_op['TERM'] + 1
            if term.isnumeric() or isfloat(term):
                ret.append(f'NUM{count_op["NUM"]} NUMERIC')
                count_op['NUM'] = count_op['NUM'] + 1
        return ret

    def pool(self):
        return deepcopy(self.population[random.randint(0, 32767) % PARENTS_TO_BE_CHOSEN])

    def roulette_wheel_selection(self):
        population_fitness = sum([chromosome.fitness for chromosome in self.population])
        chromosome_probabilities = [chromosome.fitness/population_fitness for chromosome in self.population]
        
        return deepcopy(np.random.choice(self.population, p=chromosome_probabilities))

    def check_evolution(self):
        evolved = self.count_distinct_edges(self.population[0]) >= len(self.graph.edges())
        # max_allowed = self.generation_counter < MAX_ALLOWABLE_GENERATIONS
        # print(f"{self.count_distinct_edges(self.population[0])} < {len(self.graph.edges())} = {evolved}")
        return (evolved)

    def evolve(self):
        # loop
        self.generation_counter = 0
        # for i in range(MAX_ALLOWABLE_GENERATIONS):
        for i in tqdm(range(MAX_ALLOWABLE_GENERATIONS)):
            ## score population
            self.evaluate()

            ## retain elite
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            write_population(generation=self.generation_counter, path=self.path, highest_sat=self.highest_sat, population=self.population, execution_report=self.execution_report)
            dist2VF(generation=self.generation_counter,population=self.population,path=self.path,seed_ch=self.seed_ch,execution_report=self.execution_report)
            self.generate_dataset_qty()
            # self.clusterize(self.generation_counter)
            new_population = self.population[:CHROMOSOME_TO_PRESERVE]

            terminators = list(set(treenode.get_terminators(self.seed)))
            
            population_counter = CHROMOSOME_TO_PRESERVE
            while(population_counter < POPULATION_SIZE):
                offspring1 = self.roulette_wheel_selection()
                offspring2 = self.roulette_wheel_selection()

                self.crossover(offspring1, offspring2)
                
                offspring1.mutate(MUTATION_RATE, random.randrange(len(offspring1)))
                # print(f"offspring1 is {'viable' if offspring1.is_viable() else 'not viable'}")

                while not offspring1.is_viable():
                    offspring1 = deepcopy(individual.Individual(self.seed, terminators,self.trace_file))
                    offspring1.mutate(MUTATION_RATE, random.randrange(len(offspring1)))
                    # print(f"offspring1 is {'viable' if offspring1.is_viable() else 'not viable'}")
                offspring2.mutate(MUTATION_RATE, random.randrange(len(offspring2)))
                # print(f"offspring1 is {'viable' if offspring2.is_viable() else 'not viable'}")
                while not offspring2.is_viable():
                    offspring2 = deepcopy(individual.Individual(self.seed, terminators,self.trace_file))
                    offspring2.mutate(MUTATION_RATE, random.randrange(len(offspring2)))
                    # print(f"offspring2 is {'viable' if offspring2.is_viable() else 'not viable'}")
                
                new_population.append(offspring1)
                new_population.append(offspring2)
                # raise Exception('')

                # Reset fitness
                offspring1.reset()
                offspring2.reset()

                population_counter += 2
            self.generation_counter += 1
            self.population = new_population
            self.diagnosis()

        store_dataset_qty(per_cut=1., sats=self.sats, unsats=self.unsats, build_attributes=self.build_attributes, path=self.path, now=self.now)
        store_dataset_qty(per_cut=.2, sats=self.sats, unsats=self.unsats, build_attributes=self.build_attributes, path=self.path, now=self.now)

        write_hypothesis(self.path, self.hypots)

    def replace_token(self, tk_list):
        l = list()
        for tk in tk_list:
            if tk in [">=", "<="]:
                l.append(tk.value[0])
                l.append(tk.value[1])
            else:
                l.append(tk)
        return l


    def diagnosis(self):
        return
        hypots = []
        for hypot in self.hypots:
            hypots.append(list(hypot[1]))

        # if len(hypots) == 0:
        length = len(hypots[0])
        aux = np.array(hypots)
        aux = aux.transpose()
        d = {}
        for i in range(len(aux)):
            d[i] = {}
            # print(list(aux[i]))
            for j in range(len(list(aux[i]))):
                # print(aux[i][j].value)
                val = deepcopy(aux[i][j].value)
                d[i][list(aux[i]).count(val)] = val
        # print(json.dumps(d))
        diag = []
        for pos in d:
            keys = list(d[pos].keys())
            keys.sort(reverse=True)
            diag.append(d[pos][keys[0]])
        print(diag)

    def evaluate(self):
        # for chromosome in self.population:
        for chromosome in tqdm(self.population):
            if chromosome.fitness != -1:
                continue

            # self.test_chromosome(chromosome)
            print(f'Chromosome {chromosome.format()}')
            # if self.test_chromosome(chromosome) == False:
            #     continue
            # print(f'self.s={self.s}, self.e={self.e}')
            save_file(self.first, self.s, self.e, f'Not({chromosome.format()})', self.trace_file)

            folder_name = 'ga_hls'
            run_str = f'python3 {folder_name}/temp.py'
            run_tk = shlex.split(run_str)
            try:
                run_process = subprocess.run(run_tk,
                                             stderr=subprocess.PIPE,
                                             stdout=subprocess.PIPE,
                                             universal_newlines=True,
                                             timeout=10)
                # print(f'Not({chromosome.format()})')
                # print(f'Chromosome {chromosome.format()}')
                # print(run_process.stdout)
                # print(run_process.stderr)
                err = ''
                if len(run_process.stderr) > 0:
                    errorn = run_process.stderr.find('Error:')
                    if errorn == -1:
                        errorn = run_process.stderr.find('Exception:')
                    newln = run_process.stderr[:errorn].rfind('\n')
                    err = run_process.stderr[newln+1:-1]
                else:
                    errorn = run_process.stdout.find('REQUIREMENT')
                    err = run_process.stdout[errorn:-1]
                if err in self.execution_report.keys():
                    self.execution_report[err] += 1
                else:
                    self.execution_report[err] = 1
            except:
                chromosome.madeit = False
            # print(self.execution_report)
            self.execution_report['TOTAL'] += 1
            
            ## running sw
            new_result = 0
            if self.highest_sat is not None:
                new_result = self.SW.compare(self.replace_token(list(self.highest_sat)), self.replace_token(list(chromosome)), 0,0).traceback_score
            new_result_seed = self.SW.compare(self.replace_token(list(self.seed)), self.replace_token(list(chromosome)), 0,0).traceback_score
            # print(f'diff = {treenode.compare_tree(self.seed, chromosome.root)}\n')
            chromosome.sw_score = new_result_seed #+ (treenode.compare_tree(self.seed, chromosome.root) * SCALE)

            # print(f'\n')
            # print(f'{list(self.seed)}')
            # print(f'{list(chromosome)}')
            # print(f'SW score = {self.SW.compare(list(self.seed), list(chromosome), 0,0).traceback_score}')
            # print(f'\n\n')
            # print(f'{list(self.replace_token(self.seed))}')
            # print(f'{list(self.replace_token(chromosome))}')
            # print(f'SW score = {self.SW.compare(self.replace_token(list(self.seed)), self.replace_token(list(chromosome)), 0,0).traceback_score}')
            # print(f'\n')

            chromosome.fitness = int(100 * (new_result + new_result_seed) / self.max_score)# + (treenode.compare_tree(self.seed, chromosome.root) * SCALE)
            if run_process.stdout.find('SATISFIED') > 0:
                chromosome.madeit = True
                self.check_highest_sat(chromosome)
            elif run_process.stdout.find('VIOLATED') > 0:
                chromosome.madeit = False
            else:
                chromosome.madeit = False
                # chromosome.madeit = '\n'+run_process.stderr
                # print(run_process.stdout)

    def crossover(self, parent1: individual.Individual, parent2: individual.Individual):
        offsprings = [parent1, parent2]
        # print(f'{offsprings[0]}:\tParent 1 lenght: {len(offsprings[0])} \n{offsprings[1]}:\tParent 2 lenght: {len(offsprings[1])}')
        # if False: # len(offsprings[0]) != len(offsprings[1]):
        #     raise ValueError("Error in crossover: offsprings[0] and offsprings[1] lenght does not match: {} != {}.\n{}\n{}" \
        #         .format(len(offsprings[0].root), len(offsprings[1].root), offsprings[0], offsprings[1]))
        # else:
        cut_idx = random.randint(1, len(offsprings[0])-2) if len(offsprings[0]) < len(offsprings[1]) else random.randint(1, len(offsprings[1])-2)

        try:
            off0_tree, off0_sub, node0 = offsprings[0].root.cut_tree(cut_idx)
            off1_tree, off1_sub, node1 = offsprings[1].root.cut_tree(cut_idx)
        except Exception as e:
            # print(e)
            return offsprings

        node0 += off1_sub
        node1 += off0_sub

        return offsprings
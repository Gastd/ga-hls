import os
import math
import time
import json
import shlex
import random
import datetime
import subprocess
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy
from typing import Optional

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import matplotlib

from . import treenode, defs
from .individual import (
    Individual,
    QUANTIFIERS,
    RELATIONALS,
    EQUALS,
    ARITHMETICS,
    MULDIV,
    EXP,
    LOGICALS,
    NEG,
    IMP,
    FUNC,
)
from .diagnosis import Diagnosis
from .diagnostics.j48 import run_j48
from .diagnostics.arff import write_dataset_all, write_dataset_qty, write_dataset_threshold

from .lang.python_printer import formula_to_python_expr
from .lang.internal_parser import parse_internal_obj
from .lang.ast import Formula
from .harness_script import build_z3check_script
from .harness import run_property_script, Verdict
from .fitness_smithwaterman import Fitness, SmithWatermanFitness
from .fitness_smithwaterman import Smith_Waterman as SW

CROSSOVER_RATE = 0.95 ## Rate defined by Núnez-Letamendia
MUTATION_RATE = 0.9  ## Rate defined by Núnez-Letamendia
POPULATION_SIZE = 50 #30  ## Must be an EVEN number
# GENE_LENGTH = 32
# MAX_ALLOWABLE_GENERATIONS = 1000 #10 #616 ##Calculated using ALANDER , J. 1992. On optimal population size of genetic algorithms.
# MAX_ALLOWABLE_GENERATIONS = 3 #10 #616 ##Calculated using ALANDER , J. 1992. On optimal population size of genetic algorithms.
# NUMBER_OF_PARAMETERS = 17 ## Number of parameters to be evolved
# CHROMOSOME_LENGTH = GENE_LENGTH * NUMBER_OF_PARAMETERS
CHROMOSOME_TO_PRESERVE = 0 #4            ## Must be an EVEN number
PARENTS_TO_BE_CHOSEN = 10

SW_THRESHOLD = 35
FOLDS = 10

SCALE = 0.5

class GA(object):
    """docstring for GA"""
    def __init__(self,init_form,mutations=None,target_sats: int = 2, population_size: int | None = None, max_generations: int | None = None, seed: int | None = None, fitness: Fitness | None = None, output_root: str
    ):
        super(GA, self).__init__()

        # Log the timespaneach one of the tree steps in the approach
        self.checkin_start = {
            'mutation_timestamp': 0.0,
            'tracheck_timestamp': 0.0,
            'diagnosi_timestamp': 0.0
        }
        self.timespan_log = {
            'mutation_timestamp': 0.0,
            'tracheck_timestamp': 0.0,
            'diagnosi_timestamp': 0.0
        }

        self.mutations = None
        self.force_mutation = False
        print(mutations)
        if mutations is not None:
            self.set_mutation_ranges(mutations)
            self.set_force_mutations(True)

        # Seed the RNG (configurable)
        if seed is not None:
            random.seed(seed)
        else:
            random.seed()

        # Fitness configuration (default: Smith–Waterman)
        if fitness is None:
            self.fitness: Fitness = SmithWatermanFitness()
        else:
            self.fitness = fitness

        # Population size from config, falling back to legacy constant
        if population_size is not None:
            self.size = int(population_size)
        else:
            self.size = POPULATION_SIZE

        # Max generations from config, falling back to legacy constant
        # (assuming MAX_ALLOWABLE_GENERATIONS is defined in defs)
        if max_generations is not None:
            self.max_generations = int(max_generations)
        else:
            self.max_generations = MAX_ALLOWABLE_GENERATIONS

        self.highest_sat = None
        self.population = []
        self.now = datetime.datetime.now()
        self.output_root = Path(output_root)
        self.init_log(self.output_root)

        self.init_form = init_form

        # AST view of the seed formula
        self.seed_ast: Optional[Formula]
        try:
            self.seed_ast = parse_internal_obj(self.init_form)
        except Exception as exc:
            # Do not break GA if the AST view fails; just log and continue
            print(f"[ga-hls] Warning: failed to build seed AST: {exc}")
            self.seed_ast = None
        
        # DEBUG: print the seed AST for inspection
        self.print_seed_ast()
        print("seed_ast type:", type(self.seed_ast))
        if self.seed_ast is not None:
            try:
                from ga_hls.lang.analysis import formula_size, formula_depth, collect_vars
            except ImportError:
                from .lang.analysis import formula_size, formula_depth, collect_vars

            print("seed_ast size:", formula_size(self.seed_ast))
            print("seed_ast depth:", formula_depth(self.seed_ast))
            print("seed_ast vars:", sorted(collect_vars(self.seed_ast)))

        self.target_sats = int(target_sats)
        self.target_mutation = False
        self.check_if_target_is_reachable()

        self.init_population()
        self.execution_report = {'TOTAL': 0}

        self.hypots = []
        self.sats = []
        self.unsats = []
        self.unknown = []
        self.entire_dataset = []  # collects sats/unsats/unknown for diagnostics

        name = self.copy_temp_file(self.path)
        defs.FILEPATH = name
        defs.FILEPATH2= name
        print(f'Runnnig script {defs.FILEPATH} and {defs.FILEPATH2}')
        # print(f'Runnnig script {name}')
        with open('{}/hypot.txt'.format(self.path), 'a') as f:
            f.write(f'\t{defs.FILEPATH}\n')


    def _progress(self, iterable, desc: str = ""):
        """
        Wrap an iterable in a tqdm progress bar if available; otherwise return
        the iterable unchanged. Used to avoid spamming 'evaluating  0' logs.
        """
        if tqdm is not None:
            return tqdm(iterable, desc=desc, leave=False)
        else:
            if desc:
                print(desc)
            return iterable

    def print_seed_ast(self):
        """Print the AST representation of the seed formula, if available."""
        if self.seed_ast is None:
            print("No seed AST available.")
        else:
            print(self.seed_ast)

    def check_if_target_is_reachable(self):
        total_combination = 1
        combination_set = {}
        f = math.factorial
        if self.mutations is None:
            return
        else:
            for i in self.mutations:
                if self.mutations[i][0] == 'float':
                    return
                elif self.mutations[i][0] == 'int':
                    combination_set[i] = self.mutations[i][1][1] - self.mutations[i][1][0] + 1
                else:
                    combination_set[i] = len(self.mutations[i][1])
        # Check if the combinations are greater or equal than the requested satisfied requirements
        for mutation_comb in combination_set:
            print(mutation_comb)
            total_combination = total_combination * combination_set[mutation_comb]
        if self.target_sats > total_combination:
            print('Requested Satisfied requirements unreachable. Changing to all possible combinations')
            print(f'From {self.target_sats} to {total_combination}')
            self.target_sats = total_combination
            self.target_mutation = True
        return

    def get_max_score(self):
        tokens = self.replace_token(list(self.seed))
        self.max_score = self.fitness.compute_max_score(tokens=tokens)

    def set_force_mutations(self, forced: bool):
        self.force_mutation = forced

    def set_mutation_ranges(self, mutations: dict):
        self.mutations = mutations

    def init_population(self):
        self.population = []
        root = treenode.parse(self.init_form)
        self.seed = root
        terminators = list(set(treenode.get_terminators(root)))

        # Seed individual has both tree and AST
        self.seed_ch = deepcopy(Individual(root, terminators, self.seed_ast))
        self.seed_ch.show_idx()
        print(f'terminators = {terminators}')

        self.max_score = 0.0
        self.get_max_score()


        self.checkin('mutation_timestamp')
        for i in self._progress(range(self.size), desc="Initializing population"):
            # Each chromosome starts from the same tree + AST seed
            chromosome = deepcopy(Individual(root, terminators, self.seed_ast))
            chromosome.mutations = deepcopy(self.mutations)

            n = random.randrange(len(root))
            if self.force_mutation:
                chromosome.force_mutate(list(self.mutations.keys()), n)
            else:
                chromosome.mutate(1, n)

            self.population.append(deepcopy(chromosome))
        self.checkout('mutation_timestamp')
        print("Population initialized. Size = {}".format(self.size))

    def init_log(self, parent_dir):
        directory = str(self.now)
        self.path = os.path.join(parent_dir, directory)
        self.path = self.path.replace(' ', '_')
        self.path = self.path.replace(':', '_')
        print(self.path)
        try:
            os.mkdir(self.path)
            pass
        except OSError as error:
            print(error)

    def copy_temp_file(self, folder_path):
        lines = []
        with open(defs.FILEPATH2,'r') as file:
            for l in file:
                lines.append(l)

        filename = defs.FILEPATH2.split('/')[-1]

        with open(f'{folder_path}/{filename}','w') as file:
            for l in lines:
                file.write(l)
        return f'{folder_path}/{filename}'

    def write_population(self, generation):
        with open('{}/{:0>2}.txt'.format(self.path, generation), 'w') as f:
            f.write('Formula\tFitness\tSatisfied\n')
            if self.highest_sat:
                f.write('HC: ')
                f.write(str(self.highest_sat))
                f.write(f'\t{self.highest_sat.fitness}')
                f.write(f'\t{self.highest_sat.madeit}')
                f.write('\n')
            for i, chromosome in enumerate(self.population):
                f.write('{:0>2}'.format(i)+': ')
                # print(chromosome.format())
                f.write(str(chromosome))
                f.write(f'\t{chromosome.fitness}')
                f.write(f'\t{chromosome.madeit}')
                f.write('\n')
        json_object = json.dumps(self.execution_report, indent=4)
        with open(f"{self.path}/report.json", "w") as outfile:
            outfile.write(json_object)

    def generate_dataset_qty(self):
        res = []
        [res.append(x) for x in self.population if x not in res]
        [self.unknown.append(x) for x in self.population if (x not in self.unknown) and (x.madeit == 'Unknown')]
        [self.unsats.append(x) for x in self.population if (x not in self.unsats) and (x.madeit == 'False')]
        [self.sats.append(x) for x in self.population if (x not in self.sats) and (x.madeit == 'True')]
        print(f'\nWe have so far {len(self.sats)} satisfied')
        print(f'and {len(self.unsats)} unsatisfied')
        print(f'and {len(self.unknown)} unknown')

    def store_dataset_all(self):
        return write_dataset_all(
            path=self.path,
            now=self.now,
            seed=self.seed,
            population=self.population,
            seed_ch=self.seed_ch,
            unknown=self.unknown,
            unsats=self.unsats,
            sats=self.sats,
            entire_dataset=self.entire_dataset,
        )


    def store_dataset_threshold(self):
        return write_dataset_threshold(
            path=self.path,
            unknown=self.unknown, 
            unsats=self.unsats, 
            sats=self.sats, 
            now=self.now)

    def roulette_wheel_selection(self):
        population_fitness = sum([chromosome.fitness for chromosome in self.population])
        if population_fitness == 0:
            chromosome_probabilities = [1/len(self.population) for chromosome in self.population]
        else:
            chromosome_probabilities = [chromosome.fitness/population_fitness for chromosome in self.population]
        
        return deepcopy(np.random.choice(self.population, p=chromosome_probabilities))

    def check_evolution(self):
        if self.target_mutation == True:
            evolved = (len(self.sats)+len(self.unsats)) >= self.target_sats
        else:
            evolved = (len(self.sats) >= self.target_sats) and (len(self.unsats) >= self.target_sats)
        return (evolved)

    def checkin(self, logtype: str):
        self.checkin_start[logtype] = time.time()
        print(f'Check in: {logtype} {self.checkin_start[logtype]} seconds')

    def checkout(self, logtype: str):
        self.checkin_start[logtype] = time.time() - self.checkin_start[logtype]
        print(f'Check out: {logtype} {self.checkin_start[logtype]} seconds')
        self.timespan_log[logtype] = self.timespan_log[logtype] + self.checkin_start[logtype]
        print(f'Timespan: {logtype} {self.timespan_log[logtype]} seconds')

    def write_timespan_log(self):
        json_object = json.dumps(self.timespan_log, indent=4)
        with open(f"{self.path}/timespan.json", "w") as outfile:
            outfile.write(json_object)

    def evolve(self):
        with open('{}/hypot.txt'.format(self.path), 'a') as f:
            for hypot in self.hypots:
                f.write(f'\t{hypot[1]}\n')
        # loop
        self.generation_counter = 0
       
        self.generate_dataset_qty()
        s100 = self.store_dataset_all()
        
        self.checkin('tracheck_timestamp')
        self.evaluate()
        self.checkout('tracheck_timestamp')
        
        while (not self.check_evolution()) and (self.generation_counter < self.max_generations):
            self.checkin('mutation_timestamp')
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            self.write_population(self.generation_counter)
            self.generate_dataset_qty()
            s100 = self.store_dataset_all()

            new_population = self.population[:CHROMOSOME_TO_PRESERVE]

            terminators = list(set(treenode.get_terminators(self.seed)))
            
            population_counter = CHROMOSOME_TO_PRESERVE
            while(population_counter < self.size):
                offspring1 = self.roulette_wheel_selection()
                offspring2 = self.roulette_wheel_selection()

                self.crossover(offspring1, offspring2)

                if self.force_mutation:
                    offspring1.force_mutate(list(self.mutations.keys()), random.randrange(len(offspring1)))
                else:
                    offspring1.mutate(MUTATION_RATE)

                if self.force_mutation:
                    offspring2.force_mutate(list(self.mutations.keys()), random.randrange(len(offspring2)))
                else:
                    offspring2.mutate(MUTATION_RATE)

                new_population.append(offspring1)
                new_population.append(offspring2)

                # Reset fitness 
                offspring1.reset()
                offspring2.reset()

                population_counter += 2
            self.generation_counter += 1
            self.population = new_population
            self.checkout('mutation_timestamp')

            # self.diagnosis()

            # write population before trace checker
            self.generate_dataset_qty()
            s100 = self.store_dataset_all()

            ## score population
            self.checkin('tracheck_timestamp')
            self.evaluate()

            s100 = write_dataset_qty(self.path, self.now, self.seed, self.seed_ch, self.sats, self.unsats, self.unknown, per_cut=1.0)
            s025 = write_dataset_qty(self.path, self.now, self.seed, self.seed_ch, self.sats, self.unsats, self.unknown, per_cut=.25)
            s020 = write_dataset_qty(self.path, self.now, self.seed, self.seed_ch, self.sats, self.unsats, self.unknown, per_cut=.20)
            s015 = write_dataset_qty(self.path, self.now, self.seed, self.seed_ch, self.sats, self.unsats, self.unknown, per_cut=.15)
            s010 = write_dataset_qty(self.path, self.now, self.seed, self.seed_ch, self.sats, self.unsats, self.unknown, per_cut=.10)
            self.store_dataset_all()
            self.checkout('tracheck_timestamp')

            self.write_timespan_log()

        self.checkin('diagnosi_timestamp')
        s100 = write_dataset_qty(self.path, self.now, self.seed, self.seed_ch, self.sats, self.unsats, self.unknown, per_cut=1.0)
        s025 = write_dataset_qty(self.path, self.now, self.seed, self.seed_ch, self.sats, self.unsats, self.unknown, per_cut=.25)
        s020 = write_dataset_qty(self.path, self.now, self.seed, self.seed_ch, self.sats, self.unsats, self.unknown, per_cut=.20)
        s015 = write_dataset_qty(self.path, self.now, self.seed, self.seed_ch, self.sats, self.unsats, self.unknown, per_cut=.15)
        s010 = write_dataset_qty(self.path, self.now, self.seed, self.seed_ch, self.sats, self.unsats, self.unknown, per_cut=.10)
        run_j48(s100, 1.0, self.path)
        run_j48(s025, .25, self.path)
        run_j48(s020, .20, self.path)
        run_j48(s015, .15, self.path)
        run_j48(s010, .10, self.path)
        self.checkout('diagnosi_timestamp')
        self.write_timespan_log()


    def replace_token(self, tk_list):
        l = list()
        for tk in tk_list:
            if tk in [">=", "<="]:
                l.append(tk.value[0])
                l.append(tk.value[1])
            else:
                l.append(tk)
        return l

    def save_file(self, nline: str):
        """
        Create self.path/temp.py by copying the original property script
        (defs.FILEPATH) and replacing the property z3solver.add(Not(ForAll(...)))
        line with the given expression.

        This preserves the order of interval_t / conditions_t definitions.
        """
        src = defs.FILEPATH
        dst = f"{self.path}/temp.py"

        with open(src, "r") as f:
            lines = f.readlines()

        # First, try to replace the specific property line:
        #   z3solver.add(Not(ForAll([t], Implies(interval_t, conditions_t))))
        new_lines = []
        replaced = False
        marker = "z3solver.add(Not(ForAll"

        for line in lines:
            if (marker in line) and not replaced:
                # Preserve existing indentation
                indent = line[: len(line) - len(line.lstrip())]
                new_lines.append(f"{indent}z3solver.add({nline})\n")
                replaced = True
            else:
                new_lines.append(line)

        # If that exact marker wasn't found (different property shape),
        # fall back to replacing the first z3solver.add(...) in the file.
        if not replaced:
            new_lines = []
            replaced = False
            for line in lines:
                if ("z3solver.add" in line) and not replaced:
                    indent = line[: len(line) - len(line.lstrip())]
                    new_lines.append(f"{indent}z3solver.add({nline})\n")
                    replaced = True
                else:
                    new_lines.append(line)

        if not replaced:
            raise RuntimeError(f"Could not find a z3solver.add(...) line in {src}")

        with open(dst, "w") as f:
            f.writelines(new_lines)

    def evaluate(self):
        for idx, chromosome in enumerate(self._progress(self.population, desc=f"Evaluating gen {getattr(self, 'generation_counter', 0)}")
):
            if chromosome.fitness != -1:
                continue

            self.save_file(f'Not({chromosome.format()})')

            script_path = Path(self.path) / "temp.py"
            err = ""

            result = run_property_script(script_path, timeout=60 * 60)

            # Decide madeit based on harness verdict / output
            if result.verdict == Verdict.SAT:
                chromosome.madeit = "True"
            elif result.verdict == Verdict.UNSAT:
                chromosome.madeit = "False"
            else:
                # ERROR or unknown: try to classify as UNDECIDED if possible
                out = (result.stdout or "") + (result.stderr or "")
                if "UNDECIDED" in out:
                    chromosome.madeit = "Unknown"
                    err = "REQUIREMENT UNDECIDED"
                else:
                    print(result.stdout)
                    print(result.stderr)
                    chromosome.madeit = "Problem"

            if chromosome.madeit == 'Problem':
                # When we cannot evaluate the chromosome set fitness to zero and evaluate next chromosome
                chromosome.sw_score = 0
                chromosome.fitness = 0
                continue

            if err in self.execution_report.keys():
                self.execution_report[err] += 1
            else:
                self.execution_report[err] = 1
            self.execution_report['TOTAL'] += 1
            
            ## running sw
            seed_tokens = self.replace_token(list(self.seed))
            chrom_tokens = self.replace_token(list(chromosome))

            result_seed = self.fitness.score(seed_tokens, chrom_tokens)

            # Keep sw_score for backwards compatibility / ARFF
            chromosome.sw_score = result_seed

            # Normalized fitness in [0, 100] based on max_score
            if self.max_score > 0:
                chromosome.fitness = int(100 * (result_seed) / self.max_score)
            else:
                chromosome.fitness = 0
            
            # Track the best satisfied chromosome seen so far for logging
            if chromosome.madeit == 'True':
                if self.highest_sat is None or chromosome.fitness > self.highest_sat.fitness:
                    self.highest_sat = chromosome


    def crossover(self, parent1: Individual, parent2: Individual):
        offsprings = [parent1, parent2]

        # Guard: if either offspring is too short, skip crossover.
        min_len = min(len(offsprings[0]), len(offsprings[1]))
        if min_len <= 2:
            # Nothing sensible to cross; return parents as-is.
            return offsprings

        # Choose a cut index that is valid for both trees.
        cut_idx = random.randint(1, min_len - 2)

        try:
            off0_tree, off0_sub, node0 = offsprings[0].root.cut_tree(cut_idx)
            off1_tree, off1_sub, node1 = offsprings[1].root.cut_tree(cut_idx)
        except Exception:
            # If cut_tree fails for any reason, fall back to unmodified parents.
            return offsprings

        node0 += off1_sub
        node1 += off0_sub

        return offsprings

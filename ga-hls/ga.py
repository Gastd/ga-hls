import os
import math
import time
import json
import random
import datetime
import subprocess
from tqdm import tqdm

import numpy as np

from copy import deepcopy
from typing import Optional

from pathlib import Path

from . import treenode
from .defs import (
    CROSSOVER_RATE,
    MUTATION_RATE,
    POPULATION_SIZE,
    CHROMOSOME_TO_PRESERVE
    )
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
    FUNC,
)
from .diagnostics.j48 import run_j48
from .diagnostics.arff import write_dataset_all, write_dataset_qty

from .lang.python_printer import formula_to_python_expr
from .lang.internal_parser import parse_internal_obj
from .lang.internal_encoder import FormulaLayout
from .lang.ast import Formula, Not

from .harness import run_property_script, Verdict

from .fitness import Fitness, SmithWatermanFitness

from .mutation import MutationConfig

class GA(object):
    """GA over requirement formulas with AST-based mutation and harness-backed evaluation."""
    def __init__(
        self,
        init_form,
        target_sats: int = 2,
        population_size: int | None = None,
        max_generations: int | None = None,
        crossover_rate: float | None = None,
        mutation_rate: float | None = None,
        seed: int | None = None,
        fitness: Fitness | None = None,
        output_root: str | None = None,
        mutation_config: MutationConfig | None = None,
        property_path: str | None = None,
        formula_layout: FormulaLayout | None = None
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

        # AST-level mutation configuration (from pipeline / config.json)
        self.mutation_config: MutationConfig | None = mutation_config

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
        if max_generations is not None:
            self.max_generations = int(max_generations)
        else:
            self.max_generations = MAX_ALLOWABLE_GENERATIONS

        # Crossover rate from config, falling back to legacy constant
        if crossover_rate is not None:
            self.crossover_rate = crossover_rate
        else:
            self.crossover_rate = CROSSOVER_RATE
        
        if mutation_rate is not None:
            self.mutation_rate = mutation_rate
        else:
            self.mutation_rate = MUTATION_RATE
        
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

        self.target_sats = int(target_sats)
        self.target_mutation = False

        self.init_population()
        self.execution_report = {'TOTAL': 0}

        self.hypots = []
        self.sats = []
        self.unsats = []
        self.unknown = []
        self.entire_dataset = []  # collects sats/unsats/unknown for diagnostics

        self.property_path = property_path 
        self.base_property_script: str | None = None

        # Copy the original property script into this run's output dir
        # for traceability, and remember where it ended up.
        self.base_property_script = self.copy_temp_file()

        if self.base_property_script:
            print(f'Running script {self.base_property_script}')
            with open(f'{self.path}/hypot.txt', 'a') as f:
                f.write(f'\t{self.base_property_script}\n')
        else:
            print('[ga-hls] WARNING: no property script copied (property_path not set?)')

        self.formula_layout = formula_layout

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

    def get_max_score(self):
        tokens = self.replace_token(list(self.seed))
        self.max_score = self.fitness.compute_max_score(tokens=tokens)

    def init_population(self):
        self.population = []
        root = treenode.parse(self.init_form)
        self.seed = root
        terminators = list(set(treenode.get_terminators(root)))

        # Seed individual has both tree and AST
        self.seed_ch = deepcopy(Individual(root, terminators, self.seed_ast))
        # self.seed_ch.show_idx()
        #print(f"terminators = {terminators}")

        self.checkin("mutation_timestamp")
        for i in range(0, self.size):
            # Each chromosome starts from the same tree + AST seed
            chromosome = deepcopy(Individual(root, terminators, self.seed_ast))

            n = random.randrange(len(root))

            if self.mutation_config is not None:
                chromosome.mutate(1, mutation_config=self.mutation_config)

            self.population.append(deepcopy(chromosome))
        self.checkout("mutation_timestamp")
        print(f"Population initialized. Size = {self.size}")

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

    def copy_temp_file(self) -> str:
        """
        Copy the original property script (self.property_path) into this run's
        output directory for traceability.

        Returns the full path to the copied file, or "" if nothing was copied.
        """
        if not self.property_path:
            return ""

        src = Path(self.property_path).resolve()
        filename = src.name
        dst = Path(self.path) / filename

        try:
            with src.open('r', encoding='utf-8') as infile, dst.open('w', encoding='utf-8') as outfile:
                for line in infile:
                    outfile.write(line)
        except FileNotFoundError as exc:
            print(f"[ga-hls] WARNING: could not copy property script {src}: {exc}")
            return ""

        return str(dst)

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
            layout=self.formula_layout
        )

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

                # Offspring 1 mutation
                if self.mutation_config is not None:
                    offspring1.mutate(
                        self.mutation_rate,
                        mutation_config=self.mutation_config,
                    )

                # Offspring 2 mutation
                if self.mutation_config is not None:
                    offspring2.mutate(
                        self.mutation_rate,
                        mutation_config=self.mutation_config,
                    )

                new_population.append(offspring1)
                new_population.append(offspring2)

                # Reset fitness 
                offspring1.reset()
                offspring2.reset()

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

            s100 = write_dataset_qty(self.path, self.now, self.seed, self.seed_ch, self.sats, self.unsats, self.unknown, self.formula_layout, per_cut=1.0)
            s025 = write_dataset_qty(self.path, self.now, self.seed, self.seed_ch, self.sats, self.unsats, self.unknown, self.formula_layout, per_cut=.25)
            s020 = write_dataset_qty(self.path, self.now, self.seed, self.seed_ch, self.sats, self.unsats, self.unknown, self.formula_layout, per_cut=.20)
            s015 = write_dataset_qty(self.path, self.now, self.seed, self.seed_ch, self.sats, self.unsats, self.unknown, self.formula_layout, per_cut=.15)
            s010 = write_dataset_qty(self.path, self.now, self.seed, self.seed_ch, self.sats, self.unsats, self.unknown, self.formula_layout, per_cut=.10)
            self.store_dataset_all()
            self.checkout('tracheck_timestamp')

            self.write_timespan_log()

        self.checkin('diagnosi_timestamp')
        s100 = write_dataset_qty(self.path, self.now, self.seed, self.seed_ch, self.sats, self.unsats, self.unknown, self.formula_layout, per_cut=1.0)
        s025 = write_dataset_qty(self.path, self.now, self.seed, self.seed_ch, self.sats, self.unsats, self.unknown, self.formula_layout, per_cut=.25)
        s020 = write_dataset_qty(self.path, self.now, self.seed, self.seed_ch, self.sats, self.unsats, self.unknown, self.formula_layout, per_cut=.20)
        s015 = write_dataset_qty(self.path, self.now, self.seed, self.seed_ch, self.sats, self.unsats, self.unknown, self.formula_layout, per_cut=.15)
        s010 = write_dataset_qty(self.path, self.now, self.seed, self.seed_ch, self.sats, self.unsats, self.unknown, self.formula_layout, per_cut=.10)
        self.checkout('diagnosi_timestamp')
        self.write_timespan_log()

        # Return the ARFF paths for the diagnostics layer (pipeline) to consume.
        return {
            1.0: s100,
            0.25: s025,
            0.20: s020,
            0.15: s015,
            0.10: s010,
        }



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
        (self.property_path) and replacing the property z3solver.add(Not(ForAll(...)))
        line with the given expression.
        """
        src = self.property_path or self.base_property_script
        if not src:
            raise RuntimeError("[ga-hls] save_file: no property_path/base_property_script set")

        dst = f"{self.path}/temp.py"

        with open(src, "r", encoding="utf-8") as f:
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

            # Wrap the ForAll(...) formula back into a Not(...) for trace checking
            wrapped: Formula = Not(chromosome.ast)
            nline = formula_to_python_expr(wrapped)

            self.save_file(nline)

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
            self.get_max_score()
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

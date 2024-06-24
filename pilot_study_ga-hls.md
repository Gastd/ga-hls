## Pilot Experiment.

- Begins at 23-02-22 10:05
- Finishes at 23-02-22 11:38

[gastd/ga-hls](https://github.com/gastd/ga-hls) Release v1.0

### Requirements

1. [ThEodorE 1.0](https://github.com/SNTSVV/ThEodorE/releases/tag/v1.0)
2. Property04 from ThEodorE examples

### Step-by-Step

1. Configure ThEodorE (2min)
    1. Write requirements.hls manually.
    2. Execute ThEodorE to generate properties_X.py (just save the file in Eclipse)
    3. The file will be in src-gen/*.py
2. Configure ga-hls
    1. (1min) Copy-paste the generated property_X.py into ga-hls.
    2. (2min) Set filepath (ga-hls/defs.py) for the properties_X.py containing the formula.
    3. (~30min) Tranform formula from property_X.py into list of list. Example:

        Input: 
        ```
        Not(ForAll([s], Implies(And(s>0, s<18), And(Implies(signal_5[s]==0, signal_6[s]==1), Implies(signal_5[s]==1, signal_6[s]==0)))))
        ```
        Output: 
        ```
        '["ForAll", [["s"], ["Implies", [["And", [[">", ["s",0]], ["<", ["s",18] ] ]], ["And", [["Implies", [ ["==", ["signal_5[s]",0]], ["==", ["signal_6[s]",1]] ]], ["Implies", [ ["==", ["signal_5[s]",1]], ["==", ["signal_6[s]",0]] ]]]]]]]]'
        ```
        * Removes 'Not' because the tree does not deal well with unary operations.
        * Changes parentheses with squared brackets
        * Changes the notation of the operations: 'op, [args*]' instead of 'arg1 op arg2'
        * All non-numeric terms should be in "" and the whole formula should be in ''
    4. (2min) Test ga-hls/__main__.py
        ```ex = json.loads([insert formula here])```
        run and wait for the generation of mutations. they should make sense.
 3. (1min) Configure evolution (ga-hls/ga.py and ga-hls/individual.py)
      1. ga-hls/ga.py -- (CROSSOVER_RATE, MUTATION_RATE, POPULATION_SIZE, GENE_LENGTH, MAX_ALLOWABLE_GENERATIONS, CHROMOSOME_TO_PRESERVE, PARENTS_TO_BE_CHOSEN)
      2. ga-hls/individual.py -- MUTATION_NODES (i.e., quantity of nodes to be mutated every mutation)
      3. Default values
 4. (~15min ++depends on the formulas' size and traces' size) execute mutation
      ```cd ga-hls && python3 ga_hls```
      generates .arff
 5. (4min) Fix the .arff features (@attribute)
 6. (9min) Prune the dataset to pareto
      1. save five datasets with different sized samples (10%, 15%, 20%, 25%, all)
 7. (3min) Run WEKA
      1. copy-paste datasets to weka 
      2. load the dataset 
      3. run J48
    



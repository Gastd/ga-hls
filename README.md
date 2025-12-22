# ga-hls

`ga-hls` is a research tool for **diagnosing falsified requirements** in HLS/ThEodorE-based workflows using **search-based mutation** and **decision-tree learning**.

Given:

- a **falsified requirement** (a ThEodorE-generated Python property file),
- a set of **execution traces** (encoded as Z3 constraints),
- and a configuration describing how formulas may **mutate**,

`ga-hls`:

1. Parses the requirement into a **typed AST**.
2. Uses a **genetic algorithm (GA)** over structured mutations of that AST.
3. Evaluates each mutated requirement against the traces via **Z3**.
4. Logs the resulting verdicts and features into an **ARFF dataset**.
5. Runs **Weka J48** to learn a **decision tree** that explains why the original requirement failed.

The result is a **diagnostic decision tree** that relates parts of the requirement to satisfiability/violation patterns over execution traces.

---

## Package layout (`ga-hls`)

```text
ga-hls/
├── cli.py                   # CLI: run, explain-positions, inspect-internal, ...
├── config.py                # JSON config -> dataclasses
├── pipeline.py              # Unified pipeline: property → AST → GA → ARFF → J48
│
├── lang/
│   ├── ast.py               # Typed AST definitions
│   ├── theodore_parser.py   # Parse ThEodorE Python property → AST
│   ├── internal_encoder.py  # AST → GA internal format + ARFF layout
│   ├── python_printer.py    # AST → executable Python script for Z3
│   └── analysis.py          # AST utilities (size, vars, depth, etc.)
│
├── mutation/
│   └── api.py               # MutationConfig + mutate_formula (typed, safe)
│
├── diagnostics/
│   ├── arff.py              # ARFF writer
│   └── j48.py               # Weka J48 runner
│
├── ga.py                    # GA loop
├── individual.py            # GA individual (AST + internal treenode)
├── harness.py               # Safe Z3 evaluation
└── examples/                # Example ThEodorE properties
```

---  

## What the tool does (high-level)

1. Read a ThEodorE Python property (z3solver.add(Not(ForAll(...)))).
2. Parse to a typed AST (lang.ast).
3. Encode AST → internal GA format + ARFF layout.
4. Mutate formulas through typed AST operators (numeric jitter, flips, bounds).
5. Evaluate mutated properties via the safe Z3 harness (SAT / UNSAT / ERROR).
6. Generate ARFF datasets from features extracted during encoding.
7. Learn decision trees via Weka J48.

All of this orchestrated through:
```
ga-hls run --config <file.json>
```

---

## Prerequisites

You can use the docker installation or local cli installation:

For Docker (recommended):
- Docker 

Build docker. From /docker run:
```
docker compose build
```

Check CLI availability:
```
docker compose run --rm ga-hls --help
docker compose run --rm ga-hls --version
```

If you do not want to use the docker installation, you'll need:
- Python ≥ 3.8
- Z3 (pip install z3-solver)
- Java (for using Weka J48)

Install the python envrionment:
```
python -m pip install -e .
```

Simply use the tool via:
```
ga-hls --help
```

---

## Usage (docker-oriented)

Basic command help:
```
docker compose run --rm ga-hls --help
```

Full pipeline (AT1 example):
```
docker compose run --rm  ga-hls run --config ga-hls/configs/AT1_config_exp1.json
```

This performs:
```
ThEodorE Python property
        ↓
   AST (lang.ast)
        ↓ encode
Internal GA format + ARFF layout
        ↓
      GA loop
  (AST-based mutation)
        ↓ evaluate
     Z3 harness
        ↓
  ARFF dataset(s)
        ↓ 
    Weka J48
        ↓
 Diagnostic tree
```

All output is written in the output directory.

---

## Configuration Overview

A config JSON includes:
- input.requirement_file – Python property file from ThEodorE
- input.traces_file – trace Z3 encoding
- ga.* – population, generations, seed, rates
- mutation.* – allowed positions, numeric bounds, operator toggles
- diagnostics.* – ARFF output, Weka path, J48 options

Example:
```json
{
  "input": {
    "requirement_file": "ga-hls/examples/AT1_AT001.py",   // ThEodorE-generated Python requirement
    "traces_file": "ga-hls/tracesAT.csv",                // Execution traces encoded for Z3
    "output_dir": "outputs"                              // Directory for ARFF + decision trees + logs
  },

  "ga": {
    "population_size": 50,       // Number of candidate formulas per generation
    "generations": 10,           // Number of evolution iterations
    "crossover_rate": 0.95,      // Probability of combining two parents
    "mutation_rate": 0.10,       // Probability of mutating an individual
    "seed": 42                   // Random seed for reproducibility
  },

  "mutation": {
    "max_mutations": 1,          // Max number of AST-level edits per mutation step

    "enable_numeric_perturbation": true,  // Allow jittering numeric constants
    "enable_relop_flip": true,            // Allow flipping relational operators (< → >, ≤ → ≥, etc.)
    "enable_logical_flip": true,          // Allow switching AND ↔ OR ↔ Implies
    "enable_quantifier_flip": false,      // Allow switching ∀ ↔ ∃

    // Specific AST positions that are allowed to mutate (see explain-positions)
    "allowed_positions": [2,6,11],

    // Fine-grained control over *what* each position may change into
    "allowed_changes": {

      // Position 2: logical connective restricted to And, Or only, excludes Implies
      "2": {
        "logical": ["And","Or"]
      },

      // Position 6: relational operator restricted to < or >
      "6": {
        "relational": ["<", ">"]
      },

      // Position 11: numeric literal bounded to [100.0, 140.0]
      "11": {
        "numeric": [100.0, 140.0]
      }
  }
}

```

---

## Helper Commands for Introspection (AST ↔ positions ↔ ARFF)

To see where every mutation position sits in the requirement:
```
docker compose run --rm  ga-hls explain-positions --config configs/AT1_config_exp1.json
```

Example:
```
Pos  NodeType            Node                                      Features
--------------------------------------------------------------------------------
  0  ForAll              ∀ t. (((0 <= t) ∧ (t <= 20000000)) → ...  QUANTIFIERS0
  1  Implies             (((0 <= t) ∧ (t <= 20000000)) → (v_sp...  LOGICALS0
  2  And                 ((0 <= t) ∧ (t <= 20000000))              LOGICALS1
  3  RelOp               (0 <= t)                                  RELATIONALS0
  4  IntConst            0                                         NUM0
  5  Var                 t                                         TERM0
  6  RelOp               (t <= 20000000)                           RELATIONALS1
  7  Var                 t                                         TERM1
  8  IntConst            20000000                                  NUM1
  9  RelOp               (v_speed[ToInt((RealVal(0) + ((t - 0...   RELATIONALS2
 10  Subscript           v_speed[ToInt((RealVal(0) + ((t - 0 ...   TERM2
 11  RealConst           120.0                                     NUM2

```


To inspect a single mutation slot:
```
docker compose run --rm  ga-hls explain-position --config configs/AT1_config_exp1.json --position 12
```

---

Happy diagnosing!

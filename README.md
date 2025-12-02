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

## Package layout (`tool/ga_hls`)

```text
ga_hls/
├── cli.py                   # `ga-hls` console entry (run, explain-positions, ...)
├── config.py                # Config dataclasses + JSON loader
├── defs.py                  # Legacy path settings for benchmarks
├── diagnosis.py             # Diagnosis scaffold
│
├── diagnostics/
│   ├── arff.py              # ARFF writer utilities
│   └── j48.py               # Weka J48 subprocess wrapper
│
├── examples/
│   └── AT1_AT001.py         # Minimal example property
│
├── fitness_smithwaterman.py # Smith–Waterman similarity (fitness support)
├── ga.py                    # Genetic Algorithm loop (selection, crossover, mutation)
├── harness.py               # Robust Z3 harness: SAT/UNSAT/ERROR execution
├── harness_script.py        # Utilities for generating temporary Z3 scripts
├── individual.py            # GA individual (AST + internal treenode view)
│
├── lang/
│   ├── analysis.py          # AST analysis helpers (vars, depth, size,…)
│   ├── ast.py               # Typed AST nodes
│   ├── hls_parser.py        # (Optional) HLS textual parser
│   ├── internal_encoder.py  # AST → internal GA format + ARFF feature layout
│   ├── internal_parser.py   # Parser for old JSON list-of-lists format
│   ├── python_printer.py    # Pretty-printer: AST → executable Python/Z3 script
│   └── theodore_parser.py   # Extract AST from ThEodorE Python properties
│
├── mutation/
│   └── api.py               # MutationConfig + mutate_formula (typed AST mutation)
│
├── pipeline.py              # Unified pipeline: property → AST → GA → ARFF → J48
└── treenode.py              # Binary tree representation
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

- Python ≥ 3.8
- Z3 (pip install z3-solver)
- Java (for using Weka J48)

Install the package in editable mode:
```
python -m pip install -e .
```

Check CLI availability:
```
ga-hls --help
ga-hls --version
```

---

## Usage

Full pipeline:
```
ga-hls run --config ga-hls/configs/AT1_config_exp1.json
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
    "requirement_file": "tool/ga_hls/examples/AT1_AT001.py",
    "traces_file": "tool/ga_hls/tracesAT.csv",
    "output_dir": "outputs"
  },
  "ga": {
    "population_size": 50,
    "generations": 10,
    "crossover_rate": 0.95,
    "mutation_rate": 0.10,
    "seed": 42
  },
  "mutation": {
    "max_mutations": 1,

    "enable_numeric_perturbation": true,
    "enable_relop_flip": false,
    "enable_logical_flip": false,
    "enable_quantifier_flip": false,

    "allowed_positions": [11],
    "numeric_bounds": {
      "11": [100.0, 140.0]
    }
  },
  "diagnostics": {
    "arff_filename": "outputs/example.arff",
    "weka_jar": null,
    "j48_options": ["-C", "0.25", "-M", "2"]
  }
}
```

---

## Helper Commands for Introspection (AST ↔ positions ↔ ARFF)

To see where every mutation position sits in the requirement:
```
ga-hls explain-positions --config configs/AT1_config_exp1.json
```

Example:
```
Pos  NodeType            Node                                      Features
--------------------------------------------------------------------------------
  0  ForAll              ∀ t. (((0 <= t) ∧ (t <= 20000000)) → ...  QUANTIFIERS0
  1  Implies             (((0 <= t) ∧ (t <= 20000000)) → (v_sp...  -
  2  And                 ((0 <= t) ∧ (t <= 20000000))              LOGICALS0
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
ga-hls explain-position --config configs/AT1_config_exp1.json --position 12
```

---

Happy diagnosing!

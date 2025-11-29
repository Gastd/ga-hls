from __future__ import annotations

from pathlib import Path

from .config import Config
from . import defs
from .ga import GA
from .lang.theodore_parser import load_formula_from_property
from .lang.internal_encoder import formula_to_internal_obj
from .diagnostics.j48 import run_j48


def _ensure_output_dir(path: str) -> str:
    """
    Ensure the configured output directory exists and return its absolute path.
    """
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return str(out.resolve())


def build_ga_from_config(cfg: Config) -> GA:
    """
    Construct a GA instance from the configuration

        ThEodorE property Python file
            -> Python AST
            -> Formula AST
            -> internal list-of-lists format
            -> GA(init_form)

    This uses:
      - ga_hls.lang.theodore_parser.load_formula_from_property
      - ga_hls.lang.internal_encoder.formula_to_internal_obj
    """
    # 1) Load the formula as a typed AST from the ThEodorE property file.
    formula_ast = load_formula_from_property(cfg.input.requirement_file)
 
    # 1.1) Load output directory
    output_root = Path(cfg.input.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    # 2) Encode the AST into the legacy internal format GA expects (list-of-lists).
    init_form = formula_to_internal_obj(formula_ast)


    # 3) Build the GA instance. We keep `target_sats` at its default (2) for now
    #    to preserve existing semantics; if you later add a `target_sats` field
    #    to GAConfig, you can pass it through here.
    ga = GA(
        init_form=init_form,
        mutations=cfg.input.mutations if hasattr(cfg.input, "mutations") else None,
        population_size=cfg.ga.population_size,
        max_generations=cfg.ga.generations,
        crossover_rate=cfg.ga.crossover_rate,
        mutation_rate=cfg.ga.mutation_rate,
        seed=cfg.ga.seed,
        output_root=cfg.input.output_dir
    )

    return ga


def run_diagnostics(cfg: Config) -> None:
    """
    High-level orchestration for ga-hls:

        1. Ensure the configured output root directory exists.
        2. Point defs.FILEPATH / FILEPATH2 to the configured property file.
        3. Build GA from the config via the Stage-4 loader, wiring GAConfig,
           MutationConfig, fitness, and output_root.
        4. Run GA.evolve() to perform the GA search and record all bookkeeping
           (population logs, datasets, reports) under the GA run directory.
        5. Delegate ARFF/J48 work to the diagnostics layer (outside GA).
    """
    # 1. Ensure the configured output root exists and get it as a Path
    output_root = _ensure_output_dir(cfg.input.output_dir)

    # 2. Wire property paths into defs.py so GA's harness sees the right script.
    #    Currently FILEPATH and FILEPATH2 both point to the same requirement;
    #    if you later distinguish "err" vs "reference" properties, extend Config
    #    and update these assignments accordingly.
    defs.FILEPATH = cfg.input.requirement_file
    defs.FILEPATH2 = cfg.input.requirement_file

    # 3. Build GA from config (property Python -> AST -> internal format -> GA),
    #    passing through GA, mutation, fitness, and output_root configuration.
    ga = build_ga_from_config(cfg)

    # 4. Run evolution. GA.evolve() now focuses on the GA loop and bookkeeping
    #    (population snapshots, reports, etc.) under ga.path.
    datasets = ga.evolve()

    # 5. ARFF generation and J48 invocation are handled by the diagnostics
    #    modules (e.g., diagnostics.arff / diagnostics.j48) using cfg.diagnostics
    #    and ga.path as the run directory; they are called from here, not GA.
    for qty, arff_path in datasets.items():
        run_j48(arff_path, qty, ga.path)

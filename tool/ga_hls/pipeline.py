from __future__ import annotations

from pathlib import Path

from .config import Config
from . import defs
from .ga import GA
from .lang.theodore_parser import load_formula_from_property
from .lang.internal_encoder import formula_to_internal_obj
from .diagnostics.j48 import run_j48
from .mutation import MutationConfig


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
    """
    # 1) Load the formula as a typed AST from the ThEodorE property file.
    formula_ast = load_formula_from_property(cfg.input.requirement_file)

    # 2) Encode the AST into the legacy internal format GA expects (list-of-lists).
    init_form = formula_to_internal_obj(formula_ast)

    # 3) Build the AST-level mutation configuration from cfg.mutation.
    mutation_cfg = MutationConfig(
        max_mutations=cfg.mutation.max_mutations,
        enable_numeric_perturbation=cfg.mutation.enable_numeric_perturbation,
        enable_relop_flip=cfg.mutation.enable_relop_flip,
        enable_logical_flip=cfg.mutation.enable_logical_flip,
        enable_quantifier_flip=cfg.mutation.enable_quantifier_flip,
        allowed_positions=cfg.mutation.allowed_positions,
        numeric_bounds=cfg.mutation.numeric_bounds,
    )

    # 4) Build the GA instance. We keep `target_sats` at its default (2) for now.
    ga = GA(
        init_form=init_form,
        population_size=cfg.ga.population_size,
        max_generations=cfg.ga.generations,
        crossover_rate=cfg.ga.crossover_rate,
        mutation_rate=cfg.ga.mutation_rate,
        seed=cfg.ga.seed,
        output_root=cfg.input .output_dir,
        mutation_config=mutation_cfg
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
    defs.FILEPATH = cfg.input.requirement_file
    defs.FILEPATH2 = cfg.input.requirement_file

    # 3. Build GA from config (property Python -> AST -> internal format -> GA)
    ga = build_ga_from_config(cfg)

    # 4. Run evolution. GA.evolve() returns {fraction_used -> arff_path}.
    datasets = ga.evolve()

    # 5. Run J48 for each dataset via diagnostics layer.
    for qty, arff_path in datasets.items():
        run_j48(arff_path, qty, ga.path)

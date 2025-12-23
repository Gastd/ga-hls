from __future__ import annotations

import time
from pathlib import Path

from .config import Config
from .ga import GA
from .config import Config
from .lang.theodore_parser import load_formula_from_property
from .lang.internal_encoder import formula_to_internal_obj, encode_with_layout, FormulaLayout
from .diagnostics.j48 import run_j48
from .diagnostics.summary import write_summary, parse_j48_out, sha256_file
from .mutation import MutationConfig


def _ensure_output_dir(path: str) -> str:
    """
    Ensure the configured output directory exists and return its absolute path.
    """
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return str(out.resolve())

def build_layout_from_config(cfg: Config) -> FormulaLayout:
    """
    Build a FormulaLayout for the requirement in `cfg`, without running GA.
    """
    _ensure_output_dir(cfg.input.output_dir)

    # Load the ThEodorE Python property and parse it into an AST
    formula_ast = load_formula_from_property(cfg.input.requirement_file)

    # Encode to internal representation and collect layout metadata
    _, layout = encode_with_layout(formula_ast)

    return layout

def build_ga_from_config(cfg: Config) -> GA:
    """
    Construct a GA instance from the configuration

        ThEodorE property Python file
            -> Python AST
            -> Formula AST
            -> internal list-of-lists format
            -> GA(init_form)
    """

    # 1) Build the AST-level mutation configuration from cfg.mutation.
    mutation_cfg = MutationConfig(
        max_mutations=cfg.mutation.max_mutations,
        enable_numeric_perturbation=cfg.mutation.enable_numeric_perturbation,
        enable_relop_flip=cfg.mutation.enable_relop_flip,
        enable_logical_flip=cfg.mutation.enable_logical_flip,
        enable_quantifier_flip=cfg.mutation.enable_quantifier_flip,
        allowed_positions=cfg.mutation.allowed_positions,
        allowed_changes=cfg.mutation.allowed_changes,
    )

    formula_ast = load_formula_from_property(cfg.input.requirement_file)
    internal, layout = encode_with_layout(formula_ast)

    # 2) Build the GA instance.
    ga = GA(
        init_form=internal,
        target_sats=cfg.ga.target_sats,
        population_size=cfg.ga.population_size,
        max_generations=cfg.ga.generations,
        crossover_rate=cfg.ga.crossover_rate,
        mutation_rate=cfg.ga.mutation_rate,
        seed=cfg.ga.seed,
        output_root=cfg.input .output_dir,
        mutation_config=mutation_cfg,
        property_path=str(cfg.input.requirement_file),
        formula_layout=layout,
    )

    return ga


def run_diagnostics(cfg: Config) -> None:
    """
    High-level orchestration for ga-hls:

        1. Ensure the configured output root directory exists.
        2. Build GA from the config via the Stage-4 loader, wiring GAConfig,
           MutationConfig, fitness, and output_root (and the property path).
        3. Run GA.evolve() to perform the GA search and record all bookkeeping
           (population logs, datasets, reports) under the GA run directory.
        4. Delegate ARFF/J48 work to the diagnostics layer (outside GA).
    """
    t0 = time.time()

    output_root = _ensure_output_dir(cfg.input.output_dir)

    summary = {
        "requirement_file": str(cfg.input.requirement_file),
        "traces_file": str(cfg.input.traces_file),
        "output_dir": str(output_root),
        "seed": cfg.ga.seed,
        "success": False,
        "error": None,
        "runtime_sec": None,
        "ga_sec": None,
        "j48_runs": [],        # list of {qty, arff_path, out_path, tree_*}
        "requirement_sha256": sha256_file(str(cfg.input.requirement_file)),
        "traces_sha256": sha256_file(str(cfg.input.traces_file)),
    }

    ga = None
    try:
        ga = build_ga_from_config(cfg)

        t_ga = time.time()
        datasets = ga.evolve()
        if hasattr(ga, "stats") and isinstance(ga.stats, dict):
            summary.update({f"ga_{k}": v for k, v in ga.stats.items()})
        summary["ga_sec"] = time.time() - t_ga

        # J48 per dataset
        for qty, arff_path in datasets.items():
            out_path = run_j48(arff_path, qty, ga.path)

            out_text = ""
            try:
                out_text = Path(out_path).read_text(encoding="utf-8", errors="replace")
            except Exception:
                out_text = ""

            tree_stats = parse_j48_out(out_text)
            summary["j48_runs"].append({
                "qty": qty,
                "arff_path": str(arff_path),
                "j48_out_path": str(out_path),
                **tree_stats
            })

        # Define success: at least one J48 run produced a non-empty tree root
        summary["success"] = any(r.get("root_split") for r in summary["j48_runs"])

    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"

    finally:
        summary["runtime_sec"] = time.time() - t0

        # If GA exposes anything useful already, capture it opportunistically
        if ga is not None:
            # These are safe and won't crash if attrs don't exist.
            summary["ga_path"] = getattr(ga, "path", None)
            summary["population_size"] = getattr(ga, "population_size", None) or getattr(ga, "size", None)
            summary["max_generations"] = getattr(ga, "max_generations", None)
            summary["ga_target_sats"] = getattr(ga, "target_sats", None)

        # Write summary in the GA run directory if available; else output_root
        out_dir = getattr(ga, "path", None) or output_root
        write_summary(out_dir, summary)

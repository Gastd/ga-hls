from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# --- Core config dataclasses -------------------------------------------------


@dataclass
class InputConfig:
    """
    Configuration for the falsified requirement and traces.
    """

    requirement_file: str
    traces_file: str
    output_dir: str = "outputs"


@dataclass
class GAConfig:
    """
    Genetic algorithm configuration parameters.
    """

    population_size: int = 50
    generations: int = 50
    crossover_rate: float = 0.9
    mutation_rate: float = 0.1
    elitism: int = 1
    seed: Optional[int] = None

@dataclass
class MutationConfig:
    """
    Configuration for mutations.
    """
    max_mutations: int = 1

    enable_numeric_perturbation: bool = True
    enable_relop_flip: bool = True
    enable_logical_flip: bool = True
    enable_quantifier_flip: bool = True

    # Positions in the flattened AST (preorder indices)
    allowed_positions: Optional[List[int]] = None

    # Bounds keyed by AST index; values are (low, high)
    numeric_bounds: Dict[str, NumericBounds] = field(default_factory=dict)

@dataclass
class Config:
    """
    Top-level configuration object for a ga-hls run.
    """
    input: InputConfig = field(default_factory=InputConfig)
    ga: GAConfig = field(default_factory=GAConfig)
    mutation: MutationConfig = field(default_factory=MutationConfig)

# --- Loader utilities --------------------------------------------------------


class ConfigError(RuntimeError):
    """Raised when there is a problem loading or validating a configuration."""


def _as_dict(obj: Any) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        raise ConfigError(f"Expected dict at top level, got {type(obj)!r}")
    return obj


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Failed to read config file {path!s}: {exc}") from exc

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON in config file {path!s}: {exc}") from exc

    return _as_dict(data)


def load_config(path: str | Path) -> Config:
    """
    Load a configuration from a JSON file.

    Expected high-level structure:

    {
      "input": {
        "requirement_file": "tool/requirements.hls",
        "traces_file": "tool/example_traces.txt",
        "output_dir": "outputs"
      },
      "ga": {
        "population_size": 80,
        "generations": 60,
        "crossover_rate": 0.9,
        "mutation_rate": 0.1,
        "elitism": 2,
        "seed": 42
      },
      "diagnostics": {
        "arff_filename": "outputs/example.arff",
        "weka_jar": "path/to/weka.jar",
        "j48_options": ["-C", "0.25", "-M", "2"]
      },
      "mutation": {
        "max_mutations": 1,
        "enable_numeric_perturbation": true,
        "enable_relop_flip": false,
        "enable_logical_flip": false,
        "enable_quantifier_flip": false,
        "allowed_positions": [14],
        "numeric_bounds": {
            "14": [100.0, 140.0]
      }
    }
    """
    path = Path(path)
    data = _load_json(path)

    input_data = _as_dict(data.get("input", {}))
    ga_data = _as_dict(data.get("ga", {}))
    diag_data = _as_dict(data.get("diagnostics", {}))
    mut_data = _as_dict(data.get("mutation", {}))

    try:
        input_cfg = InputConfig(
            requirement_file=input_data["requirement_file"],
            traces_file=input_data["traces_file"],
            output_dir=input_data.get("output_dir", "outputs"),
        )
    except KeyError as exc:
        missing = exc.args[0]
        raise ConfigError(
            f"Missing required field 'input.{missing}' in config file {path!s}"
        ) from exc

    ga_cfg = GAConfig(
        population_size=int(ga_data.get("population_size", GAConfig.population_size)),
        generations=int(ga_data.get("generations", GAConfig.generations)),
        crossover_rate=float(ga_data.get("crossover_rate", GAConfig.crossover_rate)),
        mutation_rate=float(ga_data.get("mutation_rate", GAConfig.mutation_rate)),
        elitism=int(ga_data.get("elitism", GAConfig.elitism)),
        seed=ga_data.get("seed"),
    )

    # --- mutation section (NEW) ---
    # numeric_bounds keys come from JSON as strings -> convert to int
    raw_bounds = mut_data.get("numeric_bounds", {})
    numeric_bounds: Dict[int, Tuple[float, float]] = {}
    for k, v in raw_bounds.items():
        idx = int(k)  # "14" -> 14
        lo, hi = v
        numeric_bounds[idx] = (float(lo), float(hi))

    mut_cfg = MutationConfig(
        max_mutations=int(mut_data.get("max_mutations", 1)),
        enable_numeric_perturbation=bool(
            mut_data.get("enable_numeric_perturbation", True)
        ),
        enable_relop_flip=bool(mut_data.get("enable_relop_flip", True)),
        enable_logical_flip=bool(mut_data.get("enable_logical_flip", True)),
        enable_quantifier_flip=bool(mut_data.get("enable_quantifier_flip", True)),
        allowed_positions=mut_data.get("allowed_positions"),
        numeric_bounds=numeric_bounds,
    )

    return Config(input=input_cfg, ga=ga_cfg, mutation=mut_cfg)

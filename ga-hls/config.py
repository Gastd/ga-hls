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
    target_sats: int = 2


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

    # Unified per-position constraints:
    # idx -> {"numeric": [lo, hi], "relational": [...], "equals": [...], "logical": [...], "arith": [...], "quantifier": [...]}
    allowed_changes: Dict[int, Dict[str, object]] = field(default_factory=dict)


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

_ALLOWED_CHANGE_KEYS = {"numeric", "logical", "relational", "equals", "quantifier", "arith"}

def _parse_allowed_changes(mut_data: Dict[str, Any], path: Path) -> Dict[int, Dict[str, object]]:
    raw = mut_data.get("allowed_changes", {})
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(
            f"Expected 'mutation.allowed_changes' to be an object/dict in {path!s}, got {type(raw)!r}"
        )

    allowed_changes: Dict[int, Dict[str, object]] = {}

    for k, spec in raw.items():
        try:
            idx = int(k)  # JSON keys are strings -> int
        except Exception as exc:
            raise ConfigError(
                f"Invalid key in mutation.allowed_changes: {k!r} (expected an integer-like string)"
            ) from exc

        if not isinstance(spec, dict):
            raise ConfigError(
                f"mutation.allowed_changes[{k!r}] must be an object/dict, got {type(spec)!r}"
            )

        parsed_spec: Dict[str, object] = {}
        for family, payload in spec.items():
            if family not in _ALLOWED_CHANGE_KEYS:
                raise ConfigError(
                    f"Unknown allowed_changes family {family!r} at position {idx}. "
                    f"Expected one of: {sorted(_ALLOWED_CHANGE_KEYS)}"
                )

            if family == "numeric":
                if not (isinstance(payload, (list, tuple)) and len(payload) == 2):
                    raise ConfigError(
                        f"mutation.allowed_changes[{idx}].numeric must be [lo, hi], got: {payload!r}"
                    )
                lo, hi = payload
                parsed_spec["numeric"] = [float(lo), float(hi)]
                continue

            # operator families: must be non-empty list[str]
            if not (isinstance(payload, list) and payload and all(isinstance(x, str) for x in payload)):
                raise ConfigError(
                    f"mutation.allowed_changes[{idx}].{family} must be a non-empty list of strings, got: {payload!r}"
                )
            parsed_spec[family] = payload

        allowed_changes[idx] = parsed_spec

    return allowed_changes


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
      "mutation": {
        "max_mutations": 1,
        "enable_numeric_perturbation": true,
        "enable_relop_flip": false,
        "enable_logical_flip": false,
        "enable_quantifier_flip": false,
        "allowed_positions": [14],
        "allowed_changes": {
          "14": {"numeric": [100.0, 140.0]}
        }
      }
    }
    """
    path = Path(path)
    data = _load_json(path)

    input_data = _as_dict(data.get("input", {}))
    ga_data = _as_dict(data.get("ga", {}))
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
        target_sats=ga_data.get("target_sats")
    )

    # --- mutation section ---
    allowed_changes = _parse_allowed_changes(mut_data, path)

    mut_cfg = MutationConfig(
        max_mutations=int(mut_data.get("max_mutations", 1)),
        enable_numeric_perturbation=bool(mut_data.get("enable_numeric_perturbation", True)),
        enable_relop_flip=bool(mut_data.get("enable_relop_flip", True)),
        enable_logical_flip=bool(mut_data.get("enable_logical_flip", True)),
        enable_quantifier_flip=bool(mut_data.get("enable_quantifier_flip", True)),
        allowed_positions=mut_data.get("allowed_positions"),
        allowed_changes=allowed_changes,
    )

    return Config(input=input_cfg, ga=ga_cfg, mutation=mut_cfg)

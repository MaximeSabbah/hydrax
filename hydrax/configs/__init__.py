"""Tuning configuration files and their loaders.

Each yaml here is THE single tuning surface for one deployment: values
only (cost weights, solver knobs, plan timing) — cost functions and
everything structural stay code. The loaders bind a yaml to its typed
dataclass schema and fail loudly on unknown sections or keys, so a typo
can never be silently ignored. Nothing in the ROS parameter layer can
override these values; the Tier A example and the ROS planner adapter
load the same file.
"""

from typing import Tuple

from hydrax import ROOT
from hydrax.tasks.panda_pregrasp import (
    PandaPregraspOptions,
    PregraspControllerConfig,
)

PREGRASP_CONFIG_YAML = ROOT + "/configs/pregrasp.yaml"

# yaml section/key -> (dataclass, field). Exhaustive: unknown sections or
# keys in the yaml are an error.
_PREGRASP_YAML_SCHEMA = {
    "costs": {
        "configuration_weight": ("options", "configuration_cost_weight"),
        "velocity_weight": ("options", "velocity_cost_weight"),
        "control_weight": ("options", "control_cost_weight"),
    },
    "solver": {
        "num_samples": ("config", "num_samples"),
        "noise_scale": ("config", "noise_scale"),
        "temperature": ("config", "temperature"),
        "num_knots": ("config", "num_knots"),
        "plan_horizon": ("config", "plan_horizon"),
        "iterations": ("config", "iterations"),
        "num_gain_samples": ("config", "num_gain_samples"),
    },
    "plan": {
        "duration_sec": ("options", "duration_sec"),
        "max_velocity_fraction": ("options", "max_velocity_fraction"),
    },
}


def load_pregrasp_config(
    path: str | None = None,
) -> Tuple[PandaPregraspOptions, PregraspControllerConfig]:
    """Load the pregrasp tuning yaml into its typed dataclasses.

    The single construction path for the OCP tuning: the Tier A example and
    the ROS planner adapter both read the same file, so the gates always
    certify exactly the configuration that deploys. Keys the yaml omits
    keep the dataclass defaults.
    """
    import yaml

    with open(PREGRASP_CONFIG_YAML if path is None else path) as f:
        raw = yaml.safe_load(f) or {}

    values: dict = {"options": {}, "config": {}}
    unknown = set(raw) - set(_PREGRASP_YAML_SCHEMA)
    if unknown:
        raise ValueError(f"unknown pregrasp yaml sections: {sorted(unknown)}")
    for section, keys in raw.items():
        schema = _PREGRASP_YAML_SCHEMA[section]
        unknown = set(keys) - set(schema)
        if unknown:
            raise ValueError(
                f"unknown pregrasp yaml keys in '{section}': {sorted(unknown)}"
            )
        for key, value in keys.items():
            target, field_name = schema[key]
            values[target][field_name] = value

    return (
        PandaPregraspOptions(**values["options"]),
        PregraspControllerConfig(**values["config"]),
    )

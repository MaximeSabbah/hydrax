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
from hydrax.tasks.panda_pick_place import (
    PandaPickPlaceOptions,
    PickPlaceControllerConfig,
)
from hydrax.tasks.panda_pregrasp import (
    PandaPregraspOptions,
    PregraspControllerConfig,
)

PREGRASP_CONFIG_YAML = ROOT + "/configs/pregrasp.yaml"
PICK_PLACE_CONFIG_YAML = ROOT + "/configs/pick_place.yaml"

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
        "mean_adaptation_rate": ("config", "mean_adaptation_rate"),
        "num_knots": ("config", "num_knots"),
        "spline_type": ("config", "spline_type"),
        "plan_horizon": ("config", "plan_horizon"),
        "iterations": ("config", "iterations"),
        "num_gain_samples": ("config", "num_gain_samples"),
    },
    "plan": {
        "duration_sec": ("options", "duration_sec"),
        "max_velocity_fraction": ("options", "max_velocity_fraction"),
    },
    "feedforward": {
        "kp": ("options", "kp_fixed"),
        "kd": ("options", "kd_fixed"),
    },
}

_PICK_PLACE_YAML_SCHEMA = {
    "costs": {
        "configuration_weight": ("options", "configuration_cost_weight"),
        "velocity_weight": ("options", "velocity_cost_weight"),
        "control_weight": ("options", "control_cost_weight"),
    },
    "solver": {
        "num_samples": ("config", "num_samples"),
        "noise_scale": ("config", "noise_scale"),
        "temperature": ("config", "temperature"),
        "mean_adaptation_rate": ("config", "mean_adaptation_rate"),
        "num_knots": ("config", "num_knots"),
        "spline_type": ("config", "spline_type"),
        "plan_horizon": ("config", "plan_horizon"),
        "iterations": ("config", "iterations"),
        "num_gain_samples": ("config", "num_gain_samples"),
    },
    "plan": {
        "max_velocity_fraction": ("options", "max_velocity_fraction"),
        "min_segment_sec": ("options", "min_segment_sec"),
        "dwell_sec": ("options", "dwell_sec"),
        "dwell_settle_sec": ("options", "dwell_settle_sec"),
    },
    "geometry": {
        "cube_pos": ("options", "cube_pos"),
        "target_pos": ("options", "target_pos"),
    },
    "feedforward": {
        "kp": ("options", "kp_fixed"),
        "kd": ("options", "kd_fixed"),
    },
}


def _load_yaml_values(path: str, schema: dict, label: str) -> dict:
    """Bind a tuning yaml to a schema; unknown sections/keys are an error."""
    import yaml

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    values: dict = {"options": {}, "config": {}}
    unknown = set(raw) - set(schema)
    if unknown:
        raise ValueError(f"unknown {label} yaml sections: {sorted(unknown)}")
    for section, keys in raw.items():
        section_schema = schema[section]
        unknown = set(keys) - set(section_schema)
        if unknown:
            raise ValueError(
                f"unknown {label} yaml keys in '{section}': {sorted(unknown)}"
            )
        for key, value in keys.items():
            target, field_name = section_schema[key]
            values[target][field_name] = value
    return values


def load_pregrasp_config(
    path: str | None = None,
) -> Tuple[PandaPregraspOptions, PregraspControllerConfig]:
    """Load the pregrasp tuning yaml into its typed dataclasses.

    The single construction path for the OCP tuning: the Tier A example and
    the ROS planner adapter both read the same file, so the gates always
    certify exactly the configuration that deploys. Keys the yaml omits
    keep the dataclass defaults.
    """
    values = _load_yaml_values(
        PREGRASP_CONFIG_YAML if path is None else path,
        _PREGRASP_YAML_SCHEMA,
        "pregrasp",
    )
    return (
        PandaPregraspOptions(**values["options"]),
        PregraspControllerConfig(**values["config"]),
    )


def load_pick_place_config(
    path: str | None = None,
) -> Tuple[PandaPickPlaceOptions, PickPlaceControllerConfig]:
    """Load the pick-and-place tuning yaml into its typed dataclasses.

    Same contract as ``load_pregrasp_config``: the Tier A example and the
    ROS planner adapter read the same file, keys the yaml omits keep the
    dataclass defaults.
    """
    values = _load_yaml_values(
        PICK_PLACE_CONFIG_YAML if path is None else path,
        _PICK_PLACE_YAML_SCHEMA,
        "pick_place",
    )
    return (
        PandaPickPlaceOptions(**values["options"]),
        PickPlaceControllerConfig(**values["config"]),
    )

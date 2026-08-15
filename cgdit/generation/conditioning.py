import re
import random

import numpy as np
import torch


def parse_condition_assignments(assignments):
    """Parse repeated ``NAME=VALUE`` command-line condition assignments."""
    condition_values = {}
    for assignment in assignments or []:
        if "=" not in assignment:
            raise ValueError(
                f"Invalid condition '{assignment}'. Expected NAME=VALUE, "
                "for example band_gap=2.0."
            )

        name, raw_value = assignment.split("=", 1)
        name = name.strip()
        raw_value = raw_value.strip()
        if not name or not raw_value:
            raise ValueError(f"Invalid condition '{assignment}'. Expected NAME=VALUE.")

        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ValueError(
                f"Condition '{name}' must have a numeric value, got '{raw_value}'."
            ) from exc

        if name in condition_values and condition_values[name] != value:
            raise ValueError(f"Condition '{name}' was provided more than once with different values.")
        condition_values[name] = value

    return condition_values


def condition_values_from_args(args):
    """Merge the repeatable interface with the legacy single-condition arguments."""
    condition_values = parse_condition_assignments(getattr(args, "condition", None))
    property_name = getattr(args, "property_name", None)
    target_value = getattr(args, "target_value", None)

    if property_name is None and target_value is not None:
        raise ValueError("--target_value requires --property_name.")
    if property_name is not None and target_value is None:
        raise ValueError("--property_name requires --target_value.")

    if property_name is not None:
        value = float(target_value)
        if property_name in condition_values and condition_values[property_name] != value:
            raise ValueError(
                f"Condition '{property_name}' was provided with conflicting target values."
            )
        condition_values[property_name] = value

    return condition_values


def validate_condition_values(condition_values, condition_configs):
    condition_configs = condition_configs or {}
    unknown = sorted(set(condition_values) - set(condition_configs))
    if unknown:
        raise ValueError(
            f"Model was not trained with condition(s) {unknown}. "
            f"Available conditions: {sorted(condition_configs)}."
        )

    for name, value in condition_values.items():
        condition_type = condition_configs[name].get("type", "scalar")
        if condition_type == "categorical" and not float(value).is_integer():
            raise ValueError(f"Categorical condition '{name}' requires an integer value.")


def apply_condition_values(batch, condition_values, condition_configs):
    """Set one shared set of target properties on every graph in a batch."""
    validate_condition_values(condition_values, condition_configs)
    device = batch.batch.device if hasattr(batch, "batch") else batch.device
    for name, value in condition_values.items():
        condition_type = condition_configs[name].get("type", "scalar")
        if condition_type == "categorical":
            tensor = torch.full(
                (batch.num_graphs,), int(value), dtype=torch.long, device=device
            )
        else:
            tensor = torch.full(
                (batch.num_graphs, 1), float(value), dtype=torch.float, device=device
            )
        setattr(batch, name, tensor)
    return batch


def condition_label(condition_values):
    if not condition_values:
        return "uncond"

    parts = []
    for name, value in condition_values.items():
        value_text = f"{value:g}".replace("-", "m").replace(".", "p")
        safe_name = re.sub(r"[^A-Za-z0-9_-]+", "-", name).strip("-")
        parts.append(f"{safe_name}-{value_text}")
    return "joint_" + "_".join(parts)


def add_condition_arguments(parser):
    parser.add_argument(
        "--condition",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help=(
            "Target condition. Repeat for joint conditioning, for example "
            "--condition formation_energy_per_atom=-1.5 --condition band_gap=2.0."
        ),
    )
    return parser


def seed_generation(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

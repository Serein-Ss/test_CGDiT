"""Sample one real base-CGDiT trajectory and export PPT-ready PNG assets."""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from ase import Atoms
from ase.data.colors import jmol_colors
from ase.utils import rotate
from ase.visualize.plot import plot_atoms
from torch_geometric.loader import DataLoader

from cgdit.common.evaluation_utils import lattices_to_params_shape, load_model
from cgdit.generation.conditioning import seed_generation
from cgdit.generation.general import SampleDataset
from cgdit.rl.symmetry_quotient import representative_indices


ROTATION = "10x,20y,0z"
RADIUS_SCALE = 0.36
MASK_COLOR = np.array([0.62, 0.65, 0.70])
FIGURE_SIZE = (2.7, 2.7)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=project_root / "output/singlerun/2026-06-27/00-32-50-mp20_base",
    )
    parser.add_argument(
        "--existing-final",
        type=Path,
        default=(
            project_root
            / "output/singlerun/2026-06-27/00-32-50-mp20_base"
            / "generated_structures/pilot/template/unconditional"
            / "eval_gen_pilot_template_uncond_seed42.pt"
        ),
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=project_root / "assets/tmp/cgdit_base_trajectory_seed42.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "assets/tmp/素材",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selection-seed", type=int, default=20260823)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--resample", action="store_true")
    return parser.parse_args()


def load_existing_final(path: Path) -> dict[str, object]:
    torch.serialization.add_safe_globals([argparse.Namespace])
    return torch.load(path, map_location="cpu", weights_only=True)


def sample_real_trajectory(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA allocation is required to sample the full trajectory")

    model, loaders, _ = load_model(args.model_dir.resolve(), load_data=True, testing=False)
    if loaders is None:
        raise RuntimeError("The base-model training dataset was not loaded")
    train_loader, _ = loaders

    seed_generation(args.seed)
    dataset = SampleDataset(train_loader.dataset, total_num=1, seed=args.seed)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(loader)).to("cuda")
    model = model.to("cuda").eval()

    _, trajectory = model.sample(
        batch,
        step_lr=1e-5,
        guidance_scale=0.0,
        retain_trajectory_stack=True,
    )
    if trajectory is None:
        raise RuntimeError("CGDiT did not return its trajectory stack")

    payload: dict[str, object] = {
        "model_dir": str(args.model_dir.resolve()),
        "existing_final": str(args.existing_final.resolve()),
        "seed": args.seed,
        "step_lr": 1e-5,
        "guidance_scale": 0.0,
        "rotation": ROTATION,
        "num_atom_types": int(model.num_atom_types),
        "num_atoms": trajectory["num_atoms"].detach().cpu(),
        "all_atom_types": trajectory["all_atom_types"].detach().cpu(),
        "all_frac_coords": trajectory["all_frac_coords"].detach().cpu(),
        "all_lattices": trajectory["all_lattices"].detach().cpu(),
        "spacegroup": batch.spacegroup.detach().cpu(),
        "anchor_index": batch.anchor_index.detach().cpu(),
    }
    compare_with_existing_final(payload, args.existing_final.resolve())
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.cache)
    return payload


def compare_with_existing_final(payload: dict[str, object], path: Path) -> None:
    existing = load_existing_final(path)
    coords = payload["all_frac_coords"][-1]
    atom_types = payload["all_atom_types"][-1]
    lattice = payload["all_lattices"][-1]
    lengths, angles = lattices_to_params_shape(lattice)

    coord_delta = torch.abs(coords - existing["frac_coords"])
    coord_delta = torch.minimum(coord_delta, 1.0 - coord_delta)
    payload["existing_final_max_coord_delta"] = float(coord_delta.max())
    payload["existing_final_atom_types_equal"] = bool(
        torch.equal(atom_types, existing["atom_types"])
    )
    payload["existing_final_max_length_delta"] = float(
        torch.abs(lengths - existing["lengths"]).max()
    )
    payload["existing_final_max_angle_delta"] = float(
        torch.abs(angles - existing["angles"]).max()
    )
    payload["matches_existing_final"] = bool(
        payload["existing_final_atom_types_equal"]
        and payload["existing_final_max_coord_delta"] < 1e-5
        and payload["existing_final_max_length_delta"] < 1e-5
        and payload["existing_final_max_angle_delta"] < 1e-4
    )


def choose_middle_pair(payload: dict[str, object], selection_seed: int) -> int:
    atom_types = payload["all_atom_types"]
    mask_class = int(payload["num_atom_types"])
    time_start = atom_types.shape[0] - 1
    low = max(2, int(round(time_start * 0.35)))
    high = min(time_start - 1, int(round(time_start * 0.65)))

    candidates = []
    for timestep in range(low, high + 1):
        index = time_start - timestep
        next_index = index + 1
        current = atom_types[index]
        adjacent = atom_types[next_index]
        current_mask = float((current == mask_class).float().mean())
        adjacent_mask = float((adjacent == mask_class).float().mean())
        if 0.05 <= current_mask <= 0.95 and 0.05 <= adjacent_mask <= 0.95:
            candidates.append(timestep)

    if not candidates:
        candidates = list(range(low, high + 1))
    rng = np.random.default_rng(selection_seed)
    return int(rng.choice(candidates))


def trajectory_state(payload: dict[str, object], timestep: int) -> dict[str, np.ndarray]:
    time_start = payload["all_atom_types"].shape[0] - 1
    if not 0 <= timestep <= time_start:
        raise ValueError(f"Timestep {timestep} is outside [0, {time_start}]")
    index = time_start - timestep
    return {
        "atom_types": payload["all_atom_types"][index].numpy(),
        "frac_coords": payload["all_frac_coords"][index].numpy(),
        "lattice": payload["all_lattices"][index, 0].numpy(),
    }

def representative_state(
    state: dict[str, np.ndarray], representatives: np.ndarray
) -> dict[str, np.ndarray]:
    return {
        "atom_types": state["atom_types"][representatives],
        "frac_coords": state["frac_coords"][representatives],
        "lattice": state["lattice"],
    }



def make_atoms(
    state: dict[str, np.ndarray],
    mask_class: int,
    mode: str,
) -> tuple[Atoms, np.ndarray, float, int]:
    atom_types = state["atom_types"].astype(int)
    if mode == "lattice":
        atoms = Atoms(
            numbers=[1],
            scaled_positions=[[0.0, 0.0, 0.0]],
            cell=state["lattice"],
            pbc=True,
        )
        return atoms, np.array([[1.0, 1.0, 1.0, 0.0]]), 0.001, 2

    if mode == "mask":
        numbers = np.ones(atom_types.shape[0], dtype=int)
        colors = np.repeat(MASK_COLOR[None, :], atom_types.shape[0], axis=0)
    else:
        is_mask = atom_types == mask_class
        numbers = np.where(is_mask, 1, atom_types + 1)
        colors = jmol_colors[numbers].copy()
        colors[is_mask] = MASK_COLOR

    atoms = Atoms(
        numbers=numbers,
        scaled_positions=state["frac_coords"],
        cell=state["lattice"],
        pbc=True,
    )
    show_cell = 2 if mode in {"full", "mask"} else 0
    return atoms, colors, RADIUS_SCALE, show_cell


def common_bbox(states: list[dict[str, np.ndarray]]) -> tuple[float, float, float, float]:
    rotation = rotate(ROTATION)
    projected = []
    corners = np.array(list(product((0.0, 1.0), repeat=3)))
    for state in states:
        cell = state["lattice"]
        projected.append((state["frac_coords"] @ cell) @ rotation)
        projected.append((corners @ cell) @ rotation)
    points = np.concatenate(projected, axis=0)
    low = points[:, :2].min(axis=0)
    high = points[:, :2].max(axis=0)
    center = (low + high) / 2.0
    span = max(float((high - low).max()), 1.0) * 1.12
    half = span / 2.0
    return center[0] - half, center[1] - half, center[0] + half, center[1] + half


def render_asset(
    state: dict[str, np.ndarray],
    mask_class: int,
    mode: str,
    bbox: tuple[float, float, float, float],
    output_path: Path,
    dpi: int,
) -> None:
    atoms, colors, radii, show_cell = make_atoms(state, mask_class, mode)
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plot_atoms(
        atoms,
        ax,
        rotation=ROTATION,
        radii=radii,
        colors=colors,
        show_unit_cell=show_cell,
        bbox=bbox,
    )
    ax.set_axis_off()
    ax.set_facecolor("white")
    fig.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(fig)


def export_assets(payload: dict[str, object], args: argparse.Namespace) -> list[Path]:
    time_start = payload["all_atom_types"].shape[0] - 1
    middle_t = choose_middle_pair(payload, args.selection_seed)
    payload["middle_timestep"] = middle_t
    payload["adjacent_timestep"] = middle_t - 1
    torch.save(payload, args.cache)

    initial = trajectory_state(payload, time_start)
    middle = trajectory_state(payload, middle_t)
    adjacent = trajectory_state(payload, middle_t - 1)
    final = trajectory_state(payload, 0)
    initial_bbox = common_bbox([initial])
    representatives = representative_indices(payload["anchor_index"]).numpy()
    middle_representatives = representative_state(middle, representatives)
    adjacent_representatives = representative_state(adjacent, representatives)
    payload["orbit_representative_indices"] = torch.from_numpy(representatives)
    payload["num_orbits"] = int(representatives.size)
    torch.save(payload, args.cache)
    middle_full_bbox = common_bbox([middle])
    adjacent_full_bbox = common_bbox([adjacent])
    final_bbox = common_bbox([final])
    middle_bbox = common_bbox([middle, adjacent])
    mask_class = int(payload["num_atom_types"])

    specs = [
        ("01_trajectory_initial.png", initial, "full", initial_bbox),
        ("02_trajectory_middle_t.png", middle, "full", middle_full_bbox),
        ("03_trajectory_middle_t_minus_1.png", adjacent, "full", adjacent_full_bbox),
        ("04_trajectory_final.png", final, "full", final_bbox),
        ("05_middle_t_lattice.png", middle, "lattice", middle_bbox),
        ("06_middle_t_elements.png", middle_representatives, "elements", middle_bbox),
        ("07_middle_t_mask_coordinates.png", middle_representatives, "mask", middle_bbox),
        ("08_middle_t_minus_1_lattice.png", adjacent, "lattice", middle_bbox),
        ("09_middle_t_minus_1_elements.png", adjacent_representatives, "elements", middle_bbox),
        ("10_middle_t_minus_1_mask_coordinates.png", adjacent_representatives, "mask", middle_bbox),
        ("11_middle_t_orbit_representative_skeleton.png", middle_representatives, "full", middle_bbox),
        ("12_middle_t_minus_1_orbit_representative_skeleton.png", adjacent_representatives, "full", middle_bbox),
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for filename, state, mode, bbox in specs:
        output_path = args.output_dir / filename
        render_asset(state, mask_class, mode, bbox, output_path, args.dpi)
        outputs.append(output_path)
    return outputs


def main() -> None:
    args = parse_args()
    if args.dpi < 300:
        raise ValueError("PNG assets require --dpi >= 300")
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    if args.cache.exists() and not args.resample:
        payload = torch.load(args.cache, map_location="cpu", weights_only=True)
    else:
        payload = sample_real_trajectory(args)
    outputs = export_assets(payload, args)
    print(f"matches_existing_final={payload.get('matches_existing_final')}")
    print(f"middle_pair=M{payload['middle_timestep']} -> M{payload['adjacent_timestep']}")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()

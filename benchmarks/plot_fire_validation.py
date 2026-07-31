# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render the FIRE validation figures from the cached minimization arrays."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: I001
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arrays", type=Path, help="arrays.npz from the FIRE minimization comparison")
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


def load_arrays(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as data:
        n_atoms = data["_n_atoms"]
        energies = {
            "RDKit MMFF94 (200)": data["energy::rdkit"],
            "nvMolKit BFGS (200)": data["energy::nvmolkit_bfgs"],
            "nvMolKit FIRE (200)": data["energy::nvmolkit_fire"],
            "nvMolKit FIRE (400)": data["energy::nvmolkit_fire_400"],
        }
    return n_atoms, energies


def plot_distributions(n_atoms: np.ndarray, energies: dict[str, np.ndarray], output_path: Path) -> None:
    reference = energies["RDKit MMFF94 (200)"]
    deltas = {
        name: (values - reference) / n_atoms for name, values in energies.items() if name != "RDKit MMFF94 (200)"
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    colors = ("#76b900", "#1f77b4", "#ff7f0e")
    joint = np.concatenate([values[np.isfinite(values)] for values in deltas.values()])
    full_limit = float(np.quantile(np.abs(joint), 0.99))
    zoom_joint = np.concatenate(
        [
            deltas["nvMolKit FIRE (200)"][np.isfinite(deltas["nvMolKit FIRE (200)"])],
            deltas["nvMolKit FIRE (400)"][np.isfinite(deltas["nvMolKit FIRE (400)"])],
        ]
    )
    zoom_limit = float(np.quantile(np.abs(zoom_joint), 0.90))

    for ax, limit, title in (
        (axes[0], full_limit, "Central 99%: all nvMolKit arms"),
        (axes[1], zoom_limit, "Central 90%: FIRE convergence"),
    ):
        bins = np.linspace(-limit, limit, 100)
        selected = (
            deltas.items() if ax is axes[0] else ((name, values) for name, values in deltas.items() if "FIRE" in name)
        )
        for color, (name, values) in zip(colors, selected):
            finite = values[np.isfinite(values)]
            ax.hist(
                finite,
                bins=bins,
                histtype="step",
                linewidth=1.7,
                color=color,
                label=name,
            )
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
        ax.set_xlim(-limit, limit)
        ax.set_xlabel(r"$(E_{\mathrm{arm}}-E_{\mathrm{RDKit}})/N_{\mathrm{atoms}}$ (kcal/mol/atom)")
        ax.set_ylabel("Conformers")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_bfgs_fire_scatter(energies: dict[str, np.ndarray], output_path: Path) -> None:
    bfgs = energies["nvMolKit BFGS (200)"]
    fire_arms = ("nvMolKit FIRE (200)", "nvMolKit FIRE (400)")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.0), sharex=True, sharey=True)

    finite_all = np.concatenate([bfgs, *(energies[name] for name in fire_arms)])
    finite_all = finite_all[np.isfinite(finite_all)]
    lo, hi = np.quantile(finite_all, [0.001, 0.999])

    for ax, name in zip(axes, fire_arms):
        fire = energies[name]
        mask = np.isfinite(bfgs) & np.isfinite(fire)
        x = bfgs[mask]
        y = fire[mask]
        correlation = float(np.corrcoef(x, y)[0, 1])
        median_delta = float(np.median(y - x))

        ax.scatter(x, y, s=5, alpha=0.25, edgecolors="none", rasterized=True)
        ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1.0)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("nvMolKit BFGS energy (kcal/mol)")
        ax.set_ylabel(f"{name} energy (kcal/mol)")
        ax.set_title(f"{name}\nr={correlation:.6f}, median ΔE={median_delta:+.3g}")
        ax.grid(alpha=0.25)

    fig.suptitle("Pairwise final MMFF94 energies from identical starting conformers")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    n_atoms, energies = load_arrays(args.arrays)
    plot_distributions(n_atoms, energies, args.output_dir / "fire_validation_energy_distributions.png")
    plot_bfgs_fire_scatter(energies, args.output_dir / "fire_validation_bfgs_fire_scatter.png")


if __name__ == "__main__":
    main()

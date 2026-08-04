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
            "RDKit MMFF94": data["energy::rdkit"],
            "BFGS": data["energy::nvmolkit_bfgs"],
            "FIRE": data["energy::nvmolkit_fire"],
        }
    return n_atoms, energies


def plot_distributions(n_atoms: np.ndarray, energies: dict[str, np.ndarray], output_path: Path) -> None:
    reference = energies["RDKit MMFF94"]
    deltas = {name: (values - reference) / n_atoms for name, values in energies.items() if name != "RDKit MMFF94"}

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    limit = 1.0e-4
    bins = np.linspace(-limit, limit, 51)
    colors = ("#76b900", "#1f77b4")
    for color, (name, values) in zip(colors, deltas.items()):
        finite = values[np.isfinite(values)]
        ax.hist(
            finite,
            bins=bins,
            histtype="step",
            linewidth=1.9,
            color=color,
            label=name,
        )
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlim(-limit, limit)
    ax.set_xlabel(r"$(E_{\mathrm{arm}}-E_{\mathrm{RDKit}})/N_{\mathrm{atoms}}$ (kcal/mol/atom)")
    ax.set_ylabel("Conformers")
    ax.set_title("Final MMFF94 energy differences at 200 steps")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_bfgs_fire_scatter(energies: dict[str, np.ndarray], output_path: Path) -> None:
    bfgs = energies["BFGS"]
    fire = energies["FIRE"]
    fig, ax = plt.subplots(figsize=(6.2, 5.4))

    finite_all = np.concatenate([bfgs, fire])
    finite_all = finite_all[np.isfinite(finite_all)]
    lo, hi = np.quantile(finite_all, [0.001, 0.999])

    mask = np.isfinite(bfgs) & np.isfinite(fire)
    x = bfgs[mask]
    y = fire[mask]
    median_delta = float(np.median(y - x))

    ax.scatter(x, y, s=5, alpha=0.32, edgecolors="none", rasterized=True)
    ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1.0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("BFGS energy (kcal/mol)")
    ax.set_ylabel("FIRE energy (kcal/mol)")
    ax.set_title(f"200 steps; median ΔE={median_delta:+.3g} kcal/mol")
    ax.grid(alpha=0.25)

    fig.suptitle("Pairwise final MMFF94 energies")
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

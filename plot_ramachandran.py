#!/usr/bin/env python
"""Plot Ramachandran diagram for alanine dipeptide training data.

This script loads a tensor containing samples in internal coordinates and
converts them back to Cartesian coordinates using the coordinate
transformation defined in :mod:`fab.target_distributions.aldp`.  The
$φ$ and $ψ$ dihedral angles are then computed with `mdtraj` and a
2D histogram of these angles is saved as a Ramachandran plot.

The default location of the training data is ``datasets/aldp/train.pt``
relative to the repository root.  The output image is stored as
``ramachandran.png``.
"""

import argparse
import os
from typing import Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import mdtraj
from openmmtools.testsystems import AlanineDipeptideVacuum

from fab.target_distributions.aldp import AldpBoltzmann


DEFAULT_DATA_PATH = os.path.join("datasets", "aldp", "train.pt")
DEFAULT_OUTPUT = "ramachandran.png"
TRANSFORM_REF = os.path.join("experiments", "aldp", "data", "position_min_energy.pt")


def load_cartesian(data_path: str) -> torch.Tensor:
    """Load internal coordinates and convert them to Cartesian positions."""
    z = torch.load(data_path).double()
    # Build coordinate transform using the reference structure shipped with
    # the repository.  We do not need the full Boltzmann distribution for
    # visualisation, only its coordinate transformation.
    aldp = AldpBoltzmann(data_path=TRANSFORM_REF, temperature=300, transform="internal", env="vacuum")
    x, _ = aldp.coordinate_transform(z)
    return x


def compute_phi_psi(x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """Compute phi and psi dihedral angles from Cartesian coordinates."""
    aldp = AlanineDipeptideVacuum(constraints=None)
    topology = mdtraj.Topology.from_openmm(aldp.topology)
    traj = mdtraj.Trajectory(x.cpu().numpy().reshape(-1, 22, 3), topology)
    psi = mdtraj.compute_psi(traj)[1].reshape(-1)
    phi = mdtraj.compute_phi(traj)[1].reshape(-1)
    mask = ~(np.isnan(phi) | np.isnan(psi))
    return phi[mask], psi[mask]


def plot_ramachandran(phi: np.ndarray, psi: np.ndarray, output_path: str) -> None:
    """Create and save a Ramachandran plot."""
    plt.figure(figsize=(8, 8))
    plt.hist2d(phi, psi, bins=64, norm=mpl.colors.LogNorm(),
               range=[[-np.pi, np.pi], [-np.pi, np.pi]])
    plt.xlabel(r"$\phi$")
    plt.ylabel(r"$\psi$")
    plt.xlim(-np.pi, np.pi)
    plt.ylim(-np.pi, np.pi)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Ramachandran diagram for alanine dipeptide training data")
    parser.add_argument("--path", default=DEFAULT_DATA_PATH, help="Path to training data (.pt)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output image file")
    args = parser.parse_args()

    x = load_cartesian(args.path)
    phi, psi = compute_phi_psi(x)
    plot_ramachandran(phi, psi, args.output)


if __name__ == "__main__":
    main()

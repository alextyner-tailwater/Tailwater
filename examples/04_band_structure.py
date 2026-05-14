"""Bulk band structure along a Brillouin-zone path.

Two ways to invoke `bulk_band_structure`:

  1. Manual path: pass `k_points` (and optionally `k_labels`) and a `spacing`.
  2. Auto path:   pass `auto=True` + `structure=<pymatgen.Structure>`.
                  The high-symmetry path is determined by seekpath.

Both modes return either the matplotlib Figure (default) or the full
BandStructureResult (with raw arrays + figure) when `return_raw=True`.
"""

import numpy as np

from tailwater import bulk_band_structure, tb_model


def manual_example():
    model = tb_model.load("outputs/wannier90_hr.hdf5")

    fig = bulk_band_structure(
        model,
        k_points    = [[0, 0.5, 0], [0, 0, 0], [0.333, 0.333, 0], [0, 0, 0]],
        k_labels    = ["M", r"$\Gamma$", "K", r"$\Gamma$"],
        spacing     = 0.01,
        fermi_level = 0.0,
        e_range     = (-3.0, 3.0),
    )
    fig.savefig("bands_manual.png", dpi=150)


def auto_example():
    from pymatgen.core import Structure

    structure = Structure.from_file("MyMaterial.cif")
    model     = tb_model.load("outputs/wannier90_hr.hdf5")

    # `auto=True` -> seekpath inspects `structure` and picks a standard
    # high-symmetry path with proper labels.
    result = bulk_band_structure(
        model,
        auto       = True,
        structure  = structure,
        spacing    = 0.02,
        e_range    = (-5.0, 5.0),
        return_raw = True,
    )
    result.figure.savefig("bands_auto.png", dpi=150)
    print(f"Saved {result.eigenvalues.shape[0]} k-points "
          f"x {result.eigenvalues.shape[1]} bands")
    np.savez("bands_auto.npz", **result.as_dict())


if __name__ == "__main__":
    manual_example()
    # auto_example()           # requires the seekpath optional dep

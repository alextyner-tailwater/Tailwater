"""Surface charge-density heat maps of a (hkl) slab from a Wannier Hamiltonian.

`surface_charge_density` takes ANY Wannier tight-binding model — a Tailwater
prediction or a DFT-generated Wannier90 Hamiltonian — re-expresses H(R) in a
general (hkl) slab, integrates |psi|^2 of the occupied states over the
surface BZ, and renders the real-space charge density as two heat maps
(top view down the normal + a side cross-section).

The `model` argument accepts a tbmodels.Model, a path to a tbmodels HDF5,
a Wannier90 `*_hr.dat` (DFT output), or the dict from `load_hr`.
"""

import numpy as np

from tailwater import surface_charge_density, load_hr, supercell_self_check


def main():
    HR_PATH = "outputs/wannier90_hr.hdf5"   # or a Wannier90 *_hr.dat from DFT
    MILLER  = (0, 0, 1)                      # surface Miller index; try (1, 1, 1)
    SIZE    = 4                              # slab thickness in unit cells

    # 0) Correctness gate — the general-(hkl) integer supercell remap must
    #    reproduce the bulk spectrum to machine precision (~1e-13 eV) before
    #    any slab built from it is trustworthy.
    model = load_hr(HR_PATH)
    err = supercell_self_check(model, MILLER)
    print(f"supercell self-check {MILLER}: {err:.2e} eV  ->  "
          f"{'OK' if err < 1e-8 else 'FAIL'}")

    # 1) Full occupation (E < mu). Returns rho, top_img, side_img, the
    #    slab/supercell, and the matplotlib `fig`.
    res = surface_charge_density(
        model, MILLER, SIZE,
        mu=0.0,          # Fermi level (eV); 0 = Tailwater training convention
        nk=12,           # nk x nk surface-BZ mesh
        sigma=0.6,       # Gaussian radius (Angstrom) per Wannier centre
        tile=3,          # in-plane unit-cell repetitions in the top view
        show=False,
        savepath="surface_charge_001.png",
    )
    np.savez("surface_charge_001.npz",
             rho=res["rho"], top_img=res["top_img"], side_img=res["side_img"],
             depth=res["depth"], layer=res["layer"])
    print(f"slab orbitals: {res['slab']['Ns']}   top_img: {res['top_img'].shape}")

    # 2) Topological surface state — restrict the occupation to a small window
    #    around E_F so only near-Fermi (surface) states contribute.
    surface_charge_density(
        model, MILLER, SIZE,
        energy_window=(-0.1, 0.1),
        show=False,
        savepath="surface_charge_001_tss.png",
    )


if __name__ == "__main__":
    main()

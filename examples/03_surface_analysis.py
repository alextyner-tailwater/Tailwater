"""Bulk DOS, surface Greens-function, and Fermi-arc analyses on the HDF5 hr-model.

All four post-processing classes accept either an HDF5 path or an
in-memory tbmodels.Model (via `tb_model.load`). Each call returns a
typed Result with NumPy arrays AND matplotlib Figures.
"""

import numpy as np

from tailwater import (
    BulkDOS,
    FermiArcMap,
    SurfaceGreensFunction,
    SurfaceSpectralDensity,
    tb_model,
)


def main():
    model = tb_model.load("outputs/wannier90_hr.hdf5")

    # 1) Bulk DOS — k-mesh averaged KPM
    dos_result = BulkDOS(
        model, k_mesh=(8, 8, 8), energies=(-4.0, 4.0),
        NC=2048, NV=4, device="cpu",
    ).run()
    dos_result.figure.savefig("bulk_dos.png")
    np.savez("bulk_dos.npz", **dos_result.as_dict())

    # 2) Surface spectral density along Γ→M→K→Γ (KPM, top + bottom)
    skpm = SurfaceSpectralDensity(
        model, surface=np.eye(3), LZ=5,
        energies=(-1.0, 1.0),
        k_path=[[0, 0.5, 0], [0, 0, 0], [0.333, 0.333, 0]],
        k_labels=["M", r"$\Gamma$", "K"],
        N_path=101, NC=2 ** 12, NV=4, device="cpu",
    ).run()
    skpm.figure_top   .savefig("surface_kpm_top.png")
    skpm.figure_bottom.savefig("surface_kpm_bottom.png")

    # 3) Surface Green's function (Lopez-Sancho)
    sgf = SurfaceGreensFunction(
        model, surface=np.eye(3),
        energies=np.linspace(-1.0, 1.0, 201),
        k_path=[[0, 0.5, 0], [0, 0, 0], [0.333, 0.333, 0]],
        k_labels=["M", r"$\Gamma$", "K"],
        N_path=101, thickness=6, NN=5, eps=0.005, device="cpu",
    ).run()
    sgf.figure_top   .savefig("surface_gf_top.png")
    sgf.figure_bottom.savefig("surface_gf_bottom.png")
    np.savez("surface_gf.npz", **sgf.as_dict())

    # 4) 2D Fermi-arc map at E = 0
    arc = FermiArcMap(
        model, surface=np.eye(3), energy=0.0,
        Nx=40, Ny=40, thickness=6, NN=5, eps=0.005, device="cpu",
    ).run()
    arc.figure_top_interpolated.savefig("fermi_arc_top.png")
    arc.figure_bottom_interpolated.savefig("fermi_arc_bottom.png")


if __name__ == "__main__":
    main()

"""Intrinsic spin Hall conductivity via WannierBerri on a Tailwater hr-model.

The Bi2Se3 family (and other strong-SOC topological insulators) is
*defined* by a large intrinsic spin Hall conductivity (SHC), so this
calculation is the natural follow-on to example 06's longitudinal
conductivity. Two pieces have to come together:

  * A WannierBerri ``System`` that carries position matrix elements
    (``berry=True``) AND spin matrix elements (``SS_R``). The latter
    is NOT auto-populated when you build a System from a tbmodels
    model — tbmodels stores only the Hamiltonian.

  * The spin matrix elements themselves. For the Tailwater 18-orbital
    basis we know analytically that each Wannier function is a S_z
    eigenstate, so we can build SS_R(R=0) from the (atom, spatial,
    spin) structure of the model — no ab-initio re-projection needed.

``tailwater.wb_system_with_spin(...)`` packages both steps. It infers
the sigma_z eigenstate doublets from the model's atomic-position
topology by default, so a plain call

    sys = wb_system_with_spin(model)

is enough — no basis JSON or hand-built mask needed.

Wall-clock at the defaults below is ~10 s on a laptop CPU for the
124-orbital Bi2Se3 hr-model. Bump NK_DIV to 16+ for production runs.

Requires:
    pip install tailwater wannierberri numba
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import wannierberri as wb

from tailwater import tb_model, wb_system_with_spin


# ----------------------------------------------------------------------
# Sweep parameters
# ----------------------------------------------------------------------
NK_DIV = 8                                         # k-mesh size per direction
NK_FFT = 4                                         # FFT mesh size per direction
EFERMI = np.linspace(-2.0, 2.0, 41)                # Fermi-energy sweep (eV)


def main():
    # ------------------------------------------------------------------
    # 1)  Load the hr-model and build a spin-aware WannierBerri System
    # ------------------------------------------------------------------
    model = tb_model.load("outputs/wannier90_hr.hdf5")

    # Default: infer sigma_z eigenstate pairs from the model's atomic
    # topology (works for compact subspace projections too). If you have
    # a basis.json from `subspace_projection`, pass
    # `basis_json_path="...basis.json"` instead.
    sys_wb = wb_system_with_spin(model)

    # ------------------------------------------------------------------
    # 2)  k-mesh
    # ------------------------------------------------------------------
    grid = wb.Grid(sys_wb, NK=(NK_DIV,) * 3, NKFFT=(NK_FFT,) * 3)

    # ------------------------------------------------------------------
    # 3)  Calculators
    # ------------------------------------------------------------------
    # The 'simple' spin-current type uses only SS_R (which we just set);
    # the 'ryoo'/'qiao' types would additionally need SR_R / SHR_R, which
    # we don't have without an ab-initio re-projection.
    calculators = {
        "shc": wb.calculators.static.SHC(
            Efermi          = EFERMI,
            kwargs_formula  = {"spin_current_type": "simple"},
        ),
        "dos": wb.calculators.static.DOS(Efermi=EFERMI, tetra=True),
    }

    # ------------------------------------------------------------------
    # 4)  Run the sweep
    # ------------------------------------------------------------------
    print(f"\nIntegrating on NK={NK_DIV}^3 / NKFFT={NK_FFT}^3 ...")
    t0 = time.time()
    result = wb.run(
        sys_wb, grid=grid, calculators=calculators,
        parallel=False, symmetrize=False,
        print_Kpoints=False, dump_results=False,
        fout_name="wb_shc",
    )
    print(f"wb.run finished in {time.time() - t0:.1f} s")

    shc_arr = np.asarray(result.results["shc"].data)      # (Nef, 3, 3, 3) in (ℏ/e)·S/cm
    dos_arr = np.asarray(result.results["dos"].data)      # (Nef,) in states/eV

    # The four canonical SHC components for a hexagonal/trigonal crystal:
    sigma_xy_z = shc_arr[:, 0, 1, 2]                       # σ^z_xy (the "Hall" piece)
    sigma_yz_x = shc_arr[:, 1, 2, 0]
    sigma_zx_y = shc_arr[:, 2, 0, 1]

    # ------------------------------------------------------------------
    # 5)  Save raw results and plot
    # ------------------------------------------------------------------
    np.savez(
        "wb_shc.npz",
        efermi=EFERMI, dos=dos_arr, shc=shc_arr,
        NK=NK_DIV, NKFFT=NK_FFT,
    )

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    axes[0].plot(EFERMI, dos_arr, "k-", lw=1.5)
    axes[0].set_ylabel("DOS (states / eV)")
    axes[0].axvline(0, ls=":", color="gray", lw=0.5)
    axes[0].set_title("Density of states")

    axes[1].plot(EFERMI, sigma_xy_z, lw=1.8, label=r"$\sigma^{z}_{xy}$")
    axes[1].plot(EFERMI, sigma_yz_x, lw=1.5, ls="--", label=r"$\sigma^{x}_{yz}$")
    axes[1].plot(EFERMI, sigma_zx_y, lw=1.5, ls=":",  label=r"$\sigma^{y}_{zx}$")
    axes[1].axvline(0, ls=":", color="gray", lw=0.5)
    axes[1].axhline(0, ls=":", color="gray", lw=0.5)
    axes[1].set_xlabel("Fermi energy (eV)")
    axes[1].set_ylabel(r"$\sigma^{\gamma}_{\alpha\beta}$  ($\hbar/e \cdot $S/cm)")
    axes[1].set_title(
        "Intrinsic spin Hall conductivity "
        "(plateau in the gap is the topological signature)"
    )
    axes[1].legend(frameon=False, loc="best")

    fig.tight_layout()
    fig.savefig("wb_shc.png", dpi=160)
    print("\nWrote wb_shc.png and wb_shc.npz")


if __name__ == "__main__":
    main()

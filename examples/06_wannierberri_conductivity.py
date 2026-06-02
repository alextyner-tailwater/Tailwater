"""Conductivity via WannierBerri on a Tailwater hr-model.

Takes a Tailwater Wannier hr-model (e.g. the API's Bi2Se3 output),
hands it to ``wannierberri.system.System_R.from_tbmodels`` (no file
I/O — the tbmodels.Model goes in directly), and sweeps three
quantities versus Fermi energy on a uniform Monkhorst-Pack k-mesh:

  1. Density of states              n(E)        [states / eV]
  2. Ohmic conductivity tensor      sigma(E)    [S/m, per 1 fs relax. time]
  3. Anomalous Hall conductivity    sigma^A(E)  [S/cm]  (Berry-curvature integral)

The Ohmic tensor's diagonal (sigma_xx, sigma_yy, sigma_zz) plays the
classic insulating-vs-metallic role: it drops to zero in the bulk gap
and lights up at the band edges.  The AHC is identically zero for a
T-reversal-invariant material (like Bi2Se3) but is included here as a
template — the same calculator handles real Berry-curvature transport
once you point it at a material that breaks T.

Runtime scales as O(num_wann^3 * NK^3).  Defaults below (NK=8,
NKFFT=4) take roughly a minute on a laptop CPU for the 124-orbital
Bi2Se3 model.  Bump NK to 16-20 for production runs.

Requires:
    pip install tailwater
    pip install wannierberri numba                    # numba is needed by
                                                       # tetrahedron integration
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import wannierberri as wb

from tailwater import tb_model


# ----------------------------------------------------------------------
# Sweep parameters
# ----------------------------------------------------------------------
NK_DIV     = 8                                     # k-mesh size per direction
NK_FFT     = 4                                     # FFT mesh size per direction
EFERMI     = np.linspace(-3.0, 3.0, 61)            # Fermi-energy sweep (eV)


def main():
    # ------------------------------------------------------------------
    # 1)  Load the hr-model and hand it to wannierberri
    # ------------------------------------------------------------------
    print("Loading hr-model ...")
    model = tb_model.load("outputs/wannier90_hr.hdf5")

    # `berry=True` builds the position matrix elements <0,n|r|R,m> from
    # the Wannier centres carried by `model.pos`.  These are what the
    # AHC / Berry-curvature calculators need.
    print("Building wannierberri System ...")
    sys_wb = wb.system.System_R.from_tbmodels(model, berry=True)
    print(f"  {sys_wb.num_wann} Wannier functions")

    # ------------------------------------------------------------------
    # 2)  k-grid: uniform mesh, with a smaller inner FFT mesh for speed.
    #     `wb.run` automatically uses the symmetry group attached to the
    #     System (None by default) to reduce the integration to the
    #     irreducible wedge.
    # ------------------------------------------------------------------
    grid = wb.Grid(sys_wb, NK=(NK_DIV,) * 3, NKFFT=(NK_FFT,) * 3)

    # ------------------------------------------------------------------
    # 3)  Calculators
    # ------------------------------------------------------------------
    calculators = {
        "dos":   wb.calculators.static.DOS           (Efermi=EFERMI, tetra=True),
        "ohmic": wb.calculators.static.Ohmic_FermiSea(Efermi=EFERMI),
        "ahc":   wb.calculators.static.AHC           (Efermi=EFERMI),
    }

    # ------------------------------------------------------------------
    # 4)  Run the sweep
    # ------------------------------------------------------------------
    print(f"\nIntegrating on NK={NK_DIV}^3 / NKFFT={NK_FFT}^3 ...")
    t0 = time.time()
    result = wb.run(
        sys_wb,
        grid       = grid,
        calculators= calculators,
        fout_name  = "wb_conductivity",                # writes wb_conductivity-*.dat
        parallel   = False,                            # set True for multi-core
        symmetrize = False,
        print_Kpoints = False,
        dump_results  = False,
    )
    print(f"wb.run finished in {time.time()-t0:.1f} s")

    dos_arr   = np.asarray(result.results["dos"  ].data)        # (Nef,)
    ohmic_arr = np.asarray(result.results["ohmic"].data)        # (Nef, 3, 3)
    ahc_arr   = np.asarray(result.results["ahc"  ].data)        # (Nef, 3)

    # ------------------------------------------------------------------
    # 5)  Save raw results and plot
    # ------------------------------------------------------------------
    np.savez(
        "wb_conductivity.npz",
        efermi = EFERMI,
        dos    = dos_arr,
        ohmic  = ohmic_arr,
        ahc    = ahc_arr,
        NK     = NK_DIV,
        NKFFT  = NK_FFT,
    )

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    # (a) DOS
    axes[0].plot(EFERMI, dos_arr, "k-", lw=1.5)
    axes[0].set_ylabel("DOS (states / eV)")
    axes[0].axvline(0, ls=":", color="gray", lw=0.5)
    axes[0].set_title("Density of states")

    # (b) Ohmic sigma (diagonal of sigma_alpha,beta)
    axes[1].plot(EFERMI, ohmic_arr[:, 0, 0], label=r"$\sigma_{xx}$", lw=1.5)
    axes[1].plot(EFERMI, ohmic_arr[:, 1, 1], label=r"$\sigma_{yy}$", lw=1.5, ls="--")
    axes[1].plot(EFERMI, ohmic_arr[:, 2, 2], label=r"$\sigma_{zz}$", lw=1.5, ls=":")
    axes[1].axvline(0, ls=":", color="gray", lw=0.5)
    axes[1].set_ylabel(r"$\sigma_{\alpha\alpha}$ (S/m, $\tau=1\,$fs)")
    axes[1].set_title("Ohmic (longitudinal) conductivity")
    axes[1].legend(frameon=False)

    # (c) AHC components (sigma^A_x, sigma^A_y, sigma^A_z in the {yz,zx,xy}
    #     channels). Identically zero for time-reversal invariant materials,
    #     so this serves as a sanity check on the integration mesh.
    axes[2].plot(EFERMI, ahc_arr[:, 0], label=r"$\sigma^A_{yz}$", lw=1.5)
    axes[2].plot(EFERMI, ahc_arr[:, 1], label=r"$\sigma^A_{zx}$", lw=1.5, ls="--")
    axes[2].plot(EFERMI, ahc_arr[:, 2], label=r"$\sigma^A_{xy}$", lw=1.5, ls=":")
    axes[2].axvline(0, ls=":", color="gray", lw=0.5)
    axes[2].axhline(0, ls=":", color="gray", lw=0.5)
    axes[2].set_xlabel("Fermi energy (eV)")
    axes[2].set_ylabel(r"$\sigma^A$ (S/cm)")
    axes[2].set_title(
        "Anomalous Hall conductivity "
        "(identically zero for T-reversal symmetric materials)"
    )
    axes[2].legend(frameon=False)

    fig.tight_layout()
    fig.savefig("wb_conductivity.png", dpi=160)
    print("\nWrote wb_conductivity.png and wb_conductivity.npz")


if __name__ == "__main__":
    main()

"""Quantum geometry, cumulative DOS, and nonlinear Drude on a Tailwater model.

Three more WannierBerri quantities on top of the longitudinal /
spin-Hall / anomalous-Hall conductivities already covered in
examples 06 and 07, all sharing a single Fermi-energy sweep:

  1. Cumulative DOS                    N(E_F)        [states / unit cell]
     -- integral of n(E) from -inf to E_F.  Use this to convert
        "I want a target carrier density n_e" into the corresponding
        E_F shift in the model.

  2. Quantum metric (Fermi-sea integral)
                                        g_alpha,beta(E_F)  [Angstrom^2]
     -- the symmetric (Re) part of the quantum geometric tensor,
        integrated over occupied bands.  Controls a growing list of
        observables: nonlinear Hall and shift-current cross-sections,
        the bound-state contribution to superfluid stiffness in flat
        bands, the Fubini-Study geometry on the Bloch manifold.
        T-even, so nonzero for Bi2Se3 (unlike AHC).

  3. Nonlinear Drude conductivity      sigma^(2)_xyz(E_F)  [(S/m)/V]
     -- second-order intraband current response.  Vanishes for
        centrosymmetric materials (so identically zero for Bi2Se3),
        but the same calculator gives meaningful answers on
        non-centrosymmetric TRS materials (BiTeI, Te, WTe2, ...).
        Included here as a template + sanity check on the integration
        mesh.

Runtime ~1-2 min at the defaults below; bump NK to 16+ for production.

Requires:
    pip install tailwater wannierberri numba
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import wannierberri as wb

from tailwater import tb_model


NK_DIV  = 6
NK_FFT  = 3
EFERMI  = np.linspace(-3.0, 3.0, 25)


def main():
    model  = tb_model.load("outputs/wannier90_hr.hdf5")
    sys_wb = wb.system.System_R.from_tbmodels(model, berry=True)
    grid   = wb.Grid(sys_wb, NK=(NK_DIV,) * 3, NKFFT=(NK_FFT,) * 3)

    calculators = {
        "dos":     wb.calculators.static.DOS                  (Efermi=EFERMI, tetra=True),
        "cumdos":  wb.calculators.static.CumDOS               (Efermi=EFERMI, tetra=True),
        "qm":      wb.calculators.static.QuantumMetric_FermiSea(Efermi=EFERMI),
        "nldrude": wb.calculators.static.NLDrude_FermiSea     (Efermi=EFERMI),
    }

    print(f"Integrating on NK={NK_DIV}^3 / NKFFT={NK_FFT}^3 ...")
    t0 = time.time()
    result = wb.run(
        sys_wb, grid=grid, calculators=calculators,
        parallel=False, symmetrize=False,
        print_Kpoints=False, dump_results=False,
        fout_name="wb_geometry",
    )
    print(f"wb.run finished in {time.time() - t0:.1f} s")

    dos    = np.asarray(result.results["dos"    ].data)    # (Nef,)
    cumdos = np.asarray(result.results["cumdos" ].data)    # (Nef,)
    qm     = np.asarray(result.results["qm"     ].data)    # (Nef, 3, 3) — symmetric
    nld    = np.asarray(result.results["nldrude"].data)    # (Nef, 3, 3, 3)

    np.savez(
        "wb_geometry.npz",
        efermi=EFERMI, dos=dos, cumdos=cumdos, qm=qm, nldrude=nld,
        NK=NK_DIV, NKFFT=NK_FFT,
    )

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)

    # (a) DOS — same panel as 06, included for context.
    axes[0, 0].plot(EFERMI, dos, "k-", lw=1.5)
    axes[0, 0].axvline(0, ls=":", color="gray", lw=0.5)
    axes[0, 0].set_ylabel("DOS (states / eV)")
    axes[0, 0].set_title("Density of states")

    # (b) Cumulative DOS — read this to set E_F for a given carrier
    #     count.  At E_F=0 the value tells you how many states sit
    #     below the bulk gap.
    axes[0, 1].plot(EFERMI, cumdos, "C2-", lw=1.5)
    axes[0, 1].axvline(0, ls=":", color="gray", lw=0.5)
    axes[0, 1].set_ylabel("CumDOS (states / unit cell)")
    axes[0, 1].set_title(r"Cumulative DOS, $N(E_F) = \int^{E_F} n(E)\,dE$")

    # (c) Quantum metric — trace and the three Cartesian diagonals.
    qm_trace = qm[:, 0, 0] + qm[:, 1, 1] + qm[:, 2, 2]
    axes[1, 0].plot(EFERMI, qm_trace, "k-", lw=1.8, label=r"$\mathrm{tr}\,g$")
    axes[1, 0].plot(EFERMI, qm[:, 0, 0], "C0--", lw=1.0, label=r"$g_{xx}$")
    axes[1, 0].plot(EFERMI, qm[:, 1, 1], "C1--", lw=1.0, label=r"$g_{yy}$")
    axes[1, 0].plot(EFERMI, qm[:, 2, 2], "C2--", lw=1.0, label=r"$g_{zz}$")
    axes[1, 0].axvline(0, ls=":", color="gray", lw=0.5)
    axes[1, 0].set_xlabel("Fermi energy (eV)")
    axes[1, 0].set_ylabel(r"Quantum metric ($\AA^2$)")
    axes[1, 0].set_title("Quantum metric (Fermi-sea integral)")
    axes[1, 0].legend(frameon=False, loc="upper right", ncol=2)

    # (d) Nonlinear Drude — three representative diagonal channels.
    #     Magnitudes around ~1e-18 below indicate the centrosymmetric
    #     zero of Bi2Se3; on a non-centrosymmetric TRS material you'd
    #     see structure of order 1 (Angstrom^2 / V) inside the bands.
    axes[1, 1].plot(EFERMI, nld[:, 0, 0, 0], lw=1.0, label=r"$\sigma^{(2)}_{xxx}$")
    axes[1, 1].plot(EFERMI, nld[:, 1, 1, 1], lw=1.0, label=r"$\sigma^{(2)}_{yyy}$")
    axes[1, 1].plot(EFERMI, nld[:, 2, 2, 2], lw=1.0, label=r"$\sigma^{(2)}_{zzz}$")
    axes[1, 1].axvline(0, ls=":", color="gray", lw=0.5)
    axes[1, 1].axhline(0, ls=":", color="gray", lw=0.5)
    axes[1, 1].set_xlabel("Fermi energy (eV)")
    axes[1, 1].set_ylabel(r"Nonlinear Drude ($\sigma^{(2)}$ / V)")
    axes[1, 1].set_title(
        "Nonlinear Drude — zero by inversion symmetry on Bi$_2$Se$_3$"
    )
    axes[1, 1].legend(frameon=False, loc="best")

    fig.tight_layout()
    fig.savefig("wb_geometry.png", dpi=160)
    print("\nWrote wb_geometry.png and wb_geometry.npz")


if __name__ == "__main__":
    main()

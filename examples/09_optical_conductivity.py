"""Optical conductivity sigma(omega) on a Tailwater hr-model.

The interband Kubo formula yields the complex optical conductivity
tensor sigma_alpha,beta(omega) at a fixed chemical potential.  Real
part is dissipative (the canonical "absorption spectrum"); imaginary
part is dispersive.  T-even, so meaningful for Bi2Se3.

For a hexagonal/trigonal crystal like Bi2Se3 the tensor has two
independent diagonal channels:

    sigma_xx = sigma_yy  (in-plane)
    sigma_zz             (out-of-plane, along the c-axis)

This script sweeps both versus omega for a single E_F (= 0, the
Tailwater Fermi reference) and plots Re sigma + Im sigma.

Wall-clock at the defaults below (NK=8^3, 51 omega points) is
~3-5 min on a laptop CPU; bump NK to 14-16 for production. The
dynamic calculator does (Nomega * 3 * 3) work per k, so omega-grid
density is the main cost knob.

Requires:
    pip install tailwater wannierberri numba
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import wannierberri as wb

from tailwater import tb_model


NK_DIV = 8
NK_FFT = 4
OMEGA  = np.linspace(0.05, 5.0, 51)       # photon energy (eV); skip omega=0 (Drude pole)
EFERMI = np.array([0.0])                  # single chemical potential
SMR    = 0.05                             # Lorentzian smearing (eV) -- broadens delta peaks


def main():
    model  = tb_model.load("outputs/wannier90_hr.hdf5")
    sys_wb = wb.system.System_R.from_tbmodels(model, berry=True)
    grid   = wb.Grid(sys_wb, NK=(NK_DIV,) * 3, NKFFT=(NK_FFT,) * 3)

    calculators = {
        "optical": wb.calculators.dynamic.OpticalConductivity(
            Efermi          = EFERMI,
            omega           = OMEGA,
            smr_fixed_width = SMR,
            smr_type        = "Lorentzian",
        ),
    }

    print(f"Integrating on NK={NK_DIV}^3 / NKFFT={NK_FFT}^3, {len(OMEGA)} omega points ...")
    t0 = time.time()
    result = wb.run(
        sys_wb, grid=grid, calculators=calculators,
        parallel=False, symmetrize=False,
        print_Kpoints=False, dump_results=False,
        fout_name="wb_optical",
    )
    print(f"wb.run finished in {time.time() - t0:.1f} s")

    # (1, Nomega, 3, 3) -> (Nomega, 3, 3)
    sigma = np.asarray(result.results["optical"].data)[0]

    np.savez(
        "wb_optical.npz",
        omega=OMEGA, efermi=EFERMI, sigma=sigma,
        NK=NK_DIV, NKFFT=NK_FFT, smr_width=SMR,
    )

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    # (a) Real part of sigma -- absorption.  sigma_xx vs sigma_zz exposes
    #     the trigonal anisotropy (in-plane vs out-of-plane).
    axes[0].plot(OMEGA, sigma[:, 0, 0].real, "C0-", lw=1.8, label=r"$\mathrm{Re}\,\sigma_{xx}$")
    axes[0].plot(OMEGA, sigma[:, 2, 2].real, "C3-", lw=1.8, label=r"$\mathrm{Re}\,\sigma_{zz}$")
    axes[0].axhline(0, ls=":", color="gray", lw=0.5)
    axes[0].set_ylabel(r"$\mathrm{Re}\,\sigma_{\alpha\alpha}$ (S/cm)")
    axes[0].set_title(r"Optical conductivity: absorptive part")
    axes[0].legend(frameon=False, loc="best")

    # (b) Imaginary part -- dispersive.  Kramers-Kronig partner of (a).
    axes[1].plot(OMEGA, sigma[:, 0, 0].imag, "C0-", lw=1.8, label=r"$\mathrm{Im}\,\sigma_{xx}$")
    axes[1].plot(OMEGA, sigma[:, 2, 2].imag, "C3-", lw=1.8, label=r"$\mathrm{Im}\,\sigma_{zz}$")
    axes[1].axhline(0, ls=":", color="gray", lw=0.5)
    axes[1].set_xlabel(r"$\hbar\omega$ (eV)")
    axes[1].set_ylabel(r"$\mathrm{Im}\,\sigma_{\alpha\alpha}$ (S/cm)")
    axes[1].set_title(r"Optical conductivity: dispersive part")
    axes[1].legend(frameon=False, loc="best")

    fig.tight_layout()
    fig.savefig("wb_optical.png", dpi=160)
    print("\nWrote wb_optical.png and wb_optical.npz")


if __name__ == "__main__":
    main()

"""Quantum-wire scattering: Tailwater hr-model → Kwant transport.

Takes a Tailwater Wannier hr-model (e.g. the API's Bi2Se3 output),
exports it to a Kwant Builder via ``model.to_kwant()``, carves out
a finite scattering region, attaches two semi-infinite leads, and
sweeps the two-terminal conductance G(E) = (e^2/h) * T(E) across an
energy window.

The geometry is a 3D Wannier model `bulk`, with:

    transport direction      = a_1  (first row of the lattice, in-plane)
    finite cross-section     = Ly cells along a_2 x Lz cells along a_3

The scattering region is a rectangular block of `Lx * Ly * Lz` unit
cells; both leads share the same `Ly * Lz` cross-section and extend
to +/-infinity along a_1. Kwant computes the scattering matrix via
the wave-function method.

Run-time scales as O((Ly * Lz)^3 * Lx) for each energy. The default
parameters below (Lx=4, Ly=2, Lz=2) take ~2 minutes on a laptop CPU.
For publication-quality results you typically want a thicker
cross-section (e.g. Ly=Lz=6+) so finite-size quantization doesn't
mask the bulk band edges, and a denser energy grid; expect the cost
to grow accordingly.

Requires:
    pip install tailwater
    conda install -c conda-forge kwant         # kwant is heavy; use conda
"""

import time

import numpy as np
import matplotlib.pyplot as plt

import kwant
from tailwater import tb_model


# ----------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------
# Keep these small for a quick demo. The matrix size of the scattering
# region scales as Lx * Ly * Lz * num_wann; for the 124-orbital Bi2Se3
# hr-model that is ~2k sites at the defaults below.
LX, LY, LZ = 4, 2, 2

# Energy grid for the G(E) sweep, in eV (zero is the API's reference,
# typically Fermi-shifted). Adjust as needed.
ENERGIES = np.linspace(-3.0, 3.0, 21)


def in_scattering_region(site, Lx, Ly, Lz):
    """A finite Lx * Ly * Lz block of unit cells."""
    x, y, z = site.tag
    return (0 <= x < Lx) and (0 <= y < Ly) and (0 <= z < Lz)


def in_lead_cross_section(site, Ly, Lz):
    """The lead's transverse cross-section: a Ly * Lz tile."""
    _, y, z = site.tag
    return (0 <= y < Ly) and (0 <= z < Lz)


def main():
    # ------------------------------------------------------------------
    # 1)  Load the hr-model and export to Kwant
    # ------------------------------------------------------------------
    model = tb_model.load("outputs/wannier90_hr.hdf5")
    bulk  = model.to_kwant()
    print(f"bulk Builder:     {sum(1 for _ in bulk.sites())} orbitals "
          f"in the 3D-periodic fundamental domain")

    # Kwant returns its lattice periods wrapped in a custom array
    # class — cast to a regular ndarray before indexing.
    periods = np.asarray(bulk.symmetry.periods)         # (3, 3) Cartesian rows

    # ------------------------------------------------------------------
    # 2)  Carve out a finite scattering region from the bulk template.
    # ------------------------------------------------------------------
    syst = kwant.Builder()                              # no symmetry — finite
    syst.fill(
        bulk,
        shape=lambda site: in_scattering_region(site, LX, LY, LZ),
        start=(0, 0, 0),
    )
    print(f"scattering region: {sum(1 for _ in syst.sites())} sites "
          f"({LX}x{LY}x{LZ} cells)")

    # ------------------------------------------------------------------
    # 3)  Build a lead — 1D-periodic along a_1 (transport direction),
    #     finite along the other two.
    # ------------------------------------------------------------------
    sym_lead = kwant.TranslationalSymmetry(periods[0])
    lead     = kwant.Builder(sym_lead)
    lead.fill(
        bulk,
        shape=lambda site: in_lead_cross_section(site, LY, LZ),
        start=(0, 0, 0),
    )
    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())
    print(f"lead cross-section: {sum(1 for _ in lead.sites())} sites")

    # ------------------------------------------------------------------
    # 4)  Finalise and sweep G(E)
    # ------------------------------------------------------------------
    t0 = time.time()
    fsyst = syst.finalized()
    print(f"finalize:           {time.time() - t0:.1f} s "
          f"({fsyst.graph.num_nodes} total nodes)")

    print()
    print("Sampling G(E) ...")
    print(f"  {'E (eV)':>8}  {'T(E)':>8}  {'time':>6}")
    transmissions = np.zeros_like(ENERGIES)
    for k, E in enumerate(ENERGIES):
        t0 = time.time()
        try:
            smatrix         = kwant.smatrix(fsyst, energy=float(E))
            transmissions[k] = smatrix.transmission(1, 0)
            ok               = True
        except Exception as exc:                        # noqa: BLE001
            transmissions[k] = np.nan
            ok               = False
            print(f"  {E:>+8.3f}    FAIL   {type(exc).__name__}")
            continue
        if ok:
            print(f"  {E:>+8.3f}  {transmissions[k]:>8.3f}  "
                  f"{time.time() - t0:>5.1f}s")

    # ------------------------------------------------------------------
    # 5)  Save the raw data and plot G(E)
    # ------------------------------------------------------------------
    np.savez(
        "kwant_conductance.npz",
        energies=ENERGIES,
        transmission=transmissions,
        Lx=LX, Ly=LY, Lz=LZ,
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ENERGIES, transmissions, "k-o", lw=1.5, ms=4)
    ax.axhline(0, ls=":", color="gray", lw=0.5)
    ax.axvline(0, ls=":", color="gray", lw=0.5)        # E_F reference
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel(r"Two-terminal conductance, $G$ ($e^2/h$)")
    ax.set_title(
        f"Quantum-wire G(E) from Tailwater hr-model "
        f"({LX}x{LY}x{LZ}-cell scattering region)"
    )
    fig.tight_layout()
    fig.savefig("kwant_conductance.png", dpi=160)
    print()
    print("Wrote kwant_conductance.png and kwant_conductance.npz")


if __name__ == "__main__":
    main()

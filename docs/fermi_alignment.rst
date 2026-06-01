Fermi alignment for semiconductors and insulators
==================================================

For any material with a band gap — semiconductors, insulators,
topological insulators — it's natural to anchor the energy scale so
the **valence band maximum (VBM)** sits at zero. The 0.4.0 release
adds two helpers for exactly this:

* :func:`~tailwater.compute_band_edges` — measures where the VBM,
  CBM, and gap fall on a uniform k-mesh.
* :func:`~tailwater.align_to_vbm` — returns a new model with on-site
  energies shifted so VBM = 0.

After ``align_to_vbm``, every downstream calculator (``BulkDOS``,
``SurfaceSpectralDensity``, ``SurfaceGreensFunction``,
``FermiArcMap``, ``BandStructure``) automatically sees the band edge
at zero — no per-calculator flags needed.


Why this matters
----------------

The Tailwater training data is Fermi-shifted so the DFT-chosen
:math:`E_F` sits at 0 eV in every training sample. For non-metals
this puts the band gap straddling :math:`E = 0`: the VBM lands
just below zero and the CBM just above. That's a reasonable
convention but inconvenient when you want band-edge plots, since
the natural physical reference is the band edge itself, not a
mid-gap point.

``align_to_vbm`` re-anchors a single model so the band edge IS the
zero. Every eigenvalue at every k is shifted by exactly the same
constant — gap-preserving, Hermiticity-preserving.


Quick start
-----------

For a non-metal like Bi\ :sub:`2`\ Se\ :sub:`3` (~0.18 eV bulk gap):

.. code-block:: python

    from tailwater import tb_model, align_to_vbm, BulkDOS, bulk_band_structure

    model = tb_model.load("wannier90_hr.hdf5")
    model = align_to_vbm(model)               # VBM is now exactly at E = 0

    # All downstream calculators inherit the new zero — no extra args needed:
    dos = BulkDOS(model, energies=(-3, 3), k_mesh=(8, 8, 8)).run()
    dos.figure.savefig("bulk_dos.png")

    fig = bulk_band_structure(
        model,
        k_points=[[0, 0, 0], [0.5, 0.5, 0], [0, 0, 0]],
        k_labels=[r"$\Gamma$", "M", r"$\Gamma$"],
        e_range=(-3, 3),
    )
    fig.savefig("bands.png")


Inspecting the band edges first
-------------------------------

If you'd like to see the gap before deciding to align:

.. code-block:: python

    from tailwater import compute_band_edges

    edges = compute_band_edges(model)
    # -> {"vbm": -0.139, "cbm": 0.040, "gap": 0.179, "is_metal": False}

    print(f"VBM = {edges['vbm']:+.4f} eV    CBM = {edges['cbm']:+.4f} eV")
    print(f"gap = {edges['gap']:.4f} eV    metal = {edges['is_metal']}")

Detection runs on a uniform 4×4×4 k-mesh by default; pass
``k_mesh=(8, 8, 8)`` (or any tuple) for a denser grid.


How VBM is identified
---------------------

The heuristic is intentionally simple. Assuming the model's existing
zero is somewhere inside the gap (the Tailwater training
convention), ``compute_band_edges`` diagonalises :math:`H(\mathbf{k})`
on the k-mesh, collects every eigenvalue across every k-point, then
reports:

* ``vbm = max(eigs < 0)`` — the negative eigenvalue closest to zero,
* ``cbm = min(eigs > 0)`` — the positive eigenvalue closest to zero,
* ``gap = cbm - vbm``,
* ``is_metal = gap <= 0``.

This avoids needing to know how many bands are occupied — a number
that's awkward to source for Wannier-projected models, especially
under spin–orbit coupling.


Metals
------

For a metal (bands crossing :math:`E = 0`), the heuristic still
returns numbers but ``is_metal`` is ``True`` and the VBM/gap aren't
physically meaningful. By default, ``align_to_vbm`` recognises this
and **emits a** ``RuntimeWarning`` **and returns the unshifted
model** so downstream code keeps working:

.. code-block:: python

    >>> model = tb_model.load("some_metal.hdf5")
    >>> aligned = align_to_vbm(model)
    RuntimeWarning: align_to_vbm: no clean gap around E=0 on the (4, 4, 4)
    k-mesh (vbm=-0.001, cbm=+0.002, gap=0.003). Consistent with a metal
    (or a non-metal whose current zero isn't in the gap). Returning
    unshifted model.

Override via the ``if_metal`` argument:

==================  =================================================================
``if_metal=...``    Behavior
==================  =================================================================
``"warn"``          *(default)* emit ``RuntimeWarning`` and return unshifted model.
``"raise"``         raise ``RuntimeError`` — fail loudly.
``"skip"``          silently return the unshifted model — useful in batch processing
                    where you'd rather log + continue without warnings.
==================  =================================================================


Overriding the auto-detection
-----------------------------

If you already know the Fermi level (from a DFT calculation, a
self-consistent integration, or just preference for a different
reference like the CBM or mid-gap), pass it explicitly:

.. code-block:: python

    # Put your chosen E_F at the new zero — shift every eigenvalue by -fermi_level:
    aligned = align_to_vbm(model, fermi_level=-0.123)

This bypasses both the k-mesh detection and the metal check — it
simply adds a constant offset to all on-site energies.


Worked patterns
---------------

**Bulk DOS, VBM-aligned:**

.. code-block:: python

    from tailwater import tb_model, align_to_vbm, BulkDOS

    model = align_to_vbm(tb_model.load("wannier90_hr.hdf5"))
    dos = BulkDOS(model, energies=(-3, 3), k_mesh=(12, 12, 12)).run()
    dos.figure.savefig("bulk_dos.png")

**Surface Green's function near the band edge:**

.. code-block:: python

    import numpy as np
    from tailwater import tb_model, align_to_vbm, SurfaceGreensFunction

    model = align_to_vbm(tb_model.load("wannier90_hr.hdf5"))
    sgf = SurfaceGreensFunction(
        model,
        surface=np.eye(3),
        energies=np.linspace(-0.5, +0.5, 201),     # ±0.5 eV around the VBM
        k_path=[[0, 0.5, 0], [0, 0, 0], [0.333, 0.333, 0]],
        k_labels=["M", r"$\Gamma$", "K"],
        n_jobs=-1,                                 # see :doc:`performance`
    ).run()
    sgf.figure_top.savefig("surface_top.png")

**Just inspecting, no shift:**

.. code-block:: python

    edges = compute_band_edges(model, k_mesh=(8, 8, 8))
    if edges["is_metal"]:
        print("metal — skipping VBM alignment")
    else:
        print(f"VBM = {edges['vbm']:+.4f} eV   "
              f"CBM = {edges['cbm']:+.4f} eV   "
              f"gap = {edges['gap']:.4f} eV")


Implementation note
-------------------

The on-site shift is applied to the ``(0, 0, 0)`` block of the
``tbmodels`` ``HopDict``. Because ``tbmodels`` includes both the
stored ``+R`` matrix and its Hermitian conjugate when building
:math:`H(\mathbf{k})`, the R=(0,0,0) block contributes to
:math:`H(\mathbf{k})` twice. ``align_to_vbm`` therefore adds
:math:`\frac{1}{2}\, \text{shift} \cdot I` to ``hop[(0,0,0)]`` so
the eigenvalue shift comes out to exactly the requested value (this
is verified numerically — the per-band shift std is at the
``1e-14`` level across random k-points).


API reference
-------------

The canonical entries live in :doc:`api/wannier_wizard`. Reproduced here
for convenience (``:no-index:`` to avoid double-indexing):

.. autofunction:: tailwater.wannier_wizard.compute_band_edges
   :no-index:

.. autofunction:: tailwater.wannier_wizard.align_to_vbm
   :no-index:

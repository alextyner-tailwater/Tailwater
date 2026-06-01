Fermi arcs and 2D surface spectral maps
========================================

:class:`~tailwater.FermiArcMap` computes the surface spectral
function on a **2D slice of the surface Brillouin zone at a single
energy**. The classic use is mapping out Fermi arcs on a topological
semimetal — open contours that connect surface projections of bulk
Weyl/Dirac nodes — but the same calculator is just as useful for any
constant-energy surface plot: e.g. checking topological-surface-state
warping at the band edge of a TI like Bi\ :sub:`2`\ Se\ :sub:`3`.

It uses the same Lopez-Sancho machinery as
:class:`~tailwater.SurfaceGreensFunction`, just laid out over a 2D
(``k_x``, ``k_y``) grid instead of a 1D k-path × energy axis.


When to use ``FermiArcMap`` vs ``SurfaceGreensFunction``
--------------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - You want…
     - Use
   * - "What does the surface spectrum look like along Γ–M–K, over
       a window of energies?"
     - :class:`~tailwater.SurfaceGreensFunction` — 1D k-path × Nw
       energies.
   * - "What does the surface look like at exactly E = E\ :sub:`F`
       (or any single energy)?"
     - :class:`~tailwater.FermiArcMap` — 2D k-grid at 1 energy.

Both calculators produce identical surface spectral densities for
the (k, w) values they share — they're just different slices.


Quick start
-----------

For a slab with the surface normal along the c-axis:

.. code-block:: python

    import numpy as np
    from tailwater import tb_model, FermiArcMap

    model = tb_model.load("wannier90_hr.hdf5")

    arc = FermiArcMap(
        model,
        surface=np.eye(3),      # 3x3 basis change — see "Choosing surface" below
        energy=0.0,             # 1 eV from E_F, or wherever the arcs live
        Nx=40, Ny=40,           # k-grid resolution
        thickness=6,            # number of unit cells in the slab
        NN=5,                   # Lopez-Sancho iterations
        eps=0.005,              # imaginary broadening, eV
        device="cpu",
        n_jobs=-1,              # parallelize the (Nx*Ny) k-grid over CPU cores
    ).run()

    arc.figure_top.savefig("arc_top.png")                  # raw map
    arc.figure_top_interpolated.savefig("arc_top_interp.png")  # smoothed
    np.savez("arc.npz", **arc.as_dict())                   # raw arrays


What you get back
-----------------

``FermiArcMap.run()`` returns a :class:`~tailwater.FermiArcMapResult`
dataclass with **four matplotlib figures** plus the raw arrays:

================================  =====================================================
Attribute                         What it shows
================================  =====================================================
``figure_top``                    Raw spectral density of the **top** surface on the
                                  fractional ``(k_x, k_y)`` grid.
``figure_bottom``                 Same for the **bottom** surface.
``figure_top_interpolated``       Top-surface map plotted in **Cartesian** ``(k_x,
                                  k_y)`` coordinates with ``scipy.griddata`` smoothing.
                                  This is the one to put in a paper.
``figure_bottom_interpolated``    Same for the bottom surface.
``spectral_top``                  ``(Nx, Ny)`` numpy array of the raw spectral density.
``spectral_bottom``               Same for the bottom.
``kx_grid``, ``ky_grid``          ``(Nx,)`` / ``(Ny,)`` fractional k-grids.
``pos_x``, ``pos_y``              ``(Nx*Ny,)`` Cartesian k-coordinates for each grid
                                  point — used by the interpolated figures.
================================  =====================================================

Use the raw figures for quick checks; use the interpolated ones for
publication-quality output. Both come "for free" out of one
``.run()``.


Choosing the energy
-------------------

The ``energy`` argument sets where the constant-energy cut lives, in
the model's energy zero (typically ``E_F = 0`` for Tailwater-trained
hr-models). For semiconductors / insulators it's natural to align
the VBM to zero first with :func:`~tailwater.align_to_vbm` (see
:doc:`fermi_alignment`) and then pick a small offset:

.. code-block:: python

    from tailwater import align_to_vbm
    model = align_to_vbm(model)        # VBM is now exactly 0

    # 50 meV above the VBM — well inside any conduction-band activity
    arc = FermiArcMap(model, surface=np.eye(3),
                      energy=0.050, Nx=60, Ny=60, n_jobs=-1).run()

For Weyl/Dirac semimetals where the arcs are at the chemical
potential, ``energy=0.0`` is usually correct.


Choosing the surface
--------------------

The ``surface`` argument is a ``3×3`` matrix whose rows give the new
basis vectors in terms of the original lattice. The slab is then
periodic along rows 0 and 1, terminated along row 2. The default
``np.eye(3)`` keeps the c-axis as the surface normal — correct for
hexagonal Bi\ :sub:`2`\ Se\ :sub:`3` 0001, MoS\ :sub:`2` 0001, etc.

For a (1, 1, 1) cubic surface (e.g. TaAs along [111]):

.. code-block:: python

    # Surface basis: u, v in the surface plane; w along the [111] normal
    surface_111 = np.array([
        [ 1,  -1,   0],
        [ 1,   1,  -2],
        [ 1,   1,   1],
    ], dtype=float)

    arc = FermiArcMap(model, surface=surface_111,
                      energy=0.0, Nx=60, Ny=60, n_jobs=-1).run()

The library re-orients the model internally; you don't need to
hand-build a slab.


Performance tips
----------------

Always set ``n_jobs=-1``. The 2D k-grid has ``Nx * Ny`` independent
Lopez-Sancho recursions — perfectly parallel. See :doc:`performance`
for the full story; for a typical ``Nx=Ny=40`` grid the speedup is
4–8× on a desktop CPU.

If you hit a memory error on a thick slab (high ``thickness``),
lower ``chunk_size`` (default ``128``) — that controls how many
k-grid points share a batched LAPACK call at a time. ``chunk_size=32``
is a safe fallback.


End-to-end example: Bi\ :sub:`2`\ Se\ :sub:`3` Dirac cone at the VBM
--------------------------------------------------------------------

Get a hr-model from the API, align the VBM to zero, then map the
surface spectral function 30 meV above the VBM — where the Bi-Se
topological surface state has visible hexagonal warping:

.. code-block:: python

    import numpy as np
    from pymatgen.core import Structure
    from tailwater import tw_api_call, tb_model, align_to_vbm, FermiArcMap

    structure = Structure.from_file("Bi2Se3.cif")
    paths = tw_api_call(structure, "user", "pw", "./out", "Bi2Se3")

    model = tb_model.load(paths["hdf5"])
    model = align_to_vbm(model)

    arc = FermiArcMap(
        model, surface=np.eye(3),
        energy=0.030,                   # 30 meV above VBM
        Nx=60, Ny=60,
        thickness=8, NN=8, eps=0.003,
        n_jobs=-1,
    ).run()

    arc.figure_top_interpolated.savefig("Bi2Se3_arc.png", dpi=200)
    np.savez("Bi2Se3_arc.npz", **arc.as_dict())


API reference
-------------

.. autoclass:: tailwater.wannier_wizard.FermiArcMap
   :members:
   :no-index:

.. autoclass:: tailwater.wannier_wizard.FermiArcMapResult
   :members:
   :no-index:

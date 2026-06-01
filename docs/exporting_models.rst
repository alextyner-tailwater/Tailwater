Exporting models: ``_hr.dat`` and pybinding ``Lattice``
========================================================

The API returns a tight-binding Hamiltonian as an HDF5 file, which
:func:`tailwater.tb_model.load` reads into a ``tbmodels.Model``.
From there you have two common downstream needs:

1. **Write the model to a Wannier90-style** ``_hr.dat`` **file** — so
   it can be consumed by external tools (``Z2Pack``, ``WannierTools``,
   downstream DFT pipelines, custom analysis scripts that expect the
   plain-text Wannier90 format).
2. **Convert the model to a pybinding** ``Lattice`` — so you can use
   pybinding's solvers, KPM routines, eigenvalue plotters, and
   transport tools on top of the Tailwater Hamiltonian.

Both are one-liners.


Writing an ``_hr.dat`` file
---------------------------

Two equivalent entry points:

.. code-block:: python

    from tailwater import tb_model, write_hr_output

    model = tb_model.load("wannier90_hr.hdf5")

    # Option A — tbmodels' native method, attached to the loaded model:
    model.to_hr_file("wannier90_hr.dat")

    # Option B — Tailwater's thin wrapper, useful when the format is a
    # runtime choice rather than hard-coded:
    write_hr_output(model, "wannier90_hr.dat", fmt="hr_dat")
    write_hr_output(model, "wannier90_hr.hdf5", fmt="hdf5")     # re-emit HDF5

Both produce the standard Wannier90 column layout::

    Rx Ry Rz   i  j   Re(H)   Im(H)

with the unit-cell weights written in the same convention Wannier90
uses, so the result drops into any tool that already reads
``wannier90_hr.dat``. The HDF5 round-trip is bit-identical.

No optional dependencies are needed for the hr export — it's pure
tbmodels under the hood.


Converting to a pybinding ``Lattice``
-------------------------------------

Every model returned by :func:`tailwater.tb_model.load` carries a
``.to_pb()`` instance method:

.. code-block:: python

    from tailwater import tb_model

    model = tb_model.load("wannier90_hr.hdf5")
    lat   = model.to_pb()

    # `lat` is a pb.Lattice — use it like any other pybinding lattice:
    import pybinding as pb
    bz = lat.brillouin_zone()
    solver = pb.solver.lapack(pb.Model(lat, pb.translational_symmetry()))
    bands  = solver.calc_bands(*bz)
    bands.plot()

What ``.to_pb()`` does:

* Reads the on-site energies off the diagonal of the ``R = (0, 0, 0)``
  hop block (works regardless of which tbmodels version exposes
  ``.on_site`` directly).
* Adds one ``Sublattice`` per Wannier orbital, anchored at the
  orbital's Cartesian position.
* Iterates the full hopping dict and adds each ``(R, i, j)`` to
  ``lat.add_one_hopping``. pybinding implies the Hermitian conjugate
  automatically, so any duplicate ``(-R, j, i)`` returned by
  tbmodels is silently skipped.

This is **lossless** for the model's Hamiltonian: every band and
every k-point eigenvalue you can compute through the ``pb.Lattice``
matches what ``tbmodels.Model.hamilton(k)`` returns, to machine
precision.


Overriding the lattice vectors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

API-produced HDF5 files don't always carry the unit cell as
``model.uc`` (it's deliberately left as ``None`` so the same model
works under any choice of lattice convention). ``.to_pb()`` falls
back to the identity ``np.eye(3)`` in that case, which is fine for
algebraic work but wrong for any plot in physical k-coordinates.

Pass the real lattice explicitly:

.. code-block:: python

    import numpy as np

    a, c = 4.143, 28.636            # Bi2Se3 hexagonal in Å
    LM = np.array([
        [ a,             0,  0],
        [-a/2,  a*np.sqrt(3)/2, 0],
        [ 0,             0,  c],
    ])

    lat = model.to_pb(lattice_vectors=LM)

After this, the pybinding ``brillouin_zone()`` returns the correct
hexagonal BZ and any band plot is in physical Å\ :sup:`-1`.


Filtering tiny hops
~~~~~~~~~~~~~~~~~~~

For exploratory analysis or visualisation you may want to drop the
smallest hops to keep ``lat`` lightweight:

.. code-block:: python

    lat = model.to_pb(hop_threshold=1e-6)         # drop |H_ij(R)| < 1e-6 eV

Keep ``hop_threshold`` low — it's there to clean up sparse-storage
zeros, **not** to act as a physical cutoff. The right place to set
the physical threshold is the inference / hr-build step (default
0.01 eV), not here.


Round-trip: HDF5 → pybinding → HDF5
-----------------------------------

A pybinding ``Lattice`` is not directly serialisable to HDF5, but
you don't need to round-trip through pybinding — just keep the
original ``tbmodels.Model`` alongside:

.. code-block:: python

    from tailwater import tb_model, write_hr_output

    model = tb_model.load("wannier90_hr.hdf5")
    lat   = model.to_pb(lattice_vectors=LM)

    # ... do pybinding work with `lat` ...

    # When you want to persist or share the Hamiltonian, write the
    # original tbmodels model — same content, two interchangeable
    # serialisations:
    write_hr_output(model, "Bi2Se3.hdf5",  fmt="hdf5")
    write_hr_output(model, "Bi2Se3_hr.dat", fmt="hr_dat")


Building a model from raw head predictions (advanced)
-----------------------------------------------------

The standard workflow uses the HDF5 the API ships back. If you
instead need to assemble the tight-binding model on the client side
from the raw dense head outputs — e.g. to experiment with a
different ``hop_threshold`` — use
:func:`~tailwater.build_hr_model_fast`:

.. code-block:: python

    from tailwater import build_hr_model_fast, write_hr_output

    hr_model = build_hr_model_fast(
        edge_pred   = edge_pred,           # [num_edges, 18, 18, 2] from API/heads
        onsite_pred = onsite_pred,         # [num_atoms, 18, 18, 2]
        gdata       = gdata,               # PyG Data the model consumed
        LM          = lattice_matrix,      # 3x3 real lattice (Å)
        atoms       = [(sym, xyz), ...],   # Cartesian per-atom positions
        hop_threshold = 0.01,              # drop |H_ij(R)| <= this (eV)
    )

    write_hr_output(hr_model, "wannier90_hr.dat", fmt="hr_dat")

``build_hr_model_fast`` is byte-identical to ``build_hr_model`` at
~100-300× the speed; prefer it unless you're debugging a build
discrepancy. Both require :command:`pip install pybinding-dev` — see
:doc:`installation`.


API reference
-------------

.. autofunction:: tailwater.client._to_pb_method
   :no-index:

.. autofunction:: tailwater.hr_export.write_hr_output
   :no-index:

.. autofunction:: tailwater.hr_export.build_hr_model
   :no-index:

.. autofunction:: tailwater.hr_export.build_hr_model_fast
   :no-index:

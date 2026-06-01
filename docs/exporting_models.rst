Exporting models: ``_hr.dat``, pybinding, and PythTB
=====================================================

The API returns a tight-binding Hamiltonian as an HDF5 file, which
:func:`tailwater.tb_model.load` reads into a ``tbmodels.Model``.
From there you have three common downstream needs:

1. **Write the model to a Wannier90-style** ``_hr.dat`` **file** — so
   it can be consumed by external tools (``Z2Pack``, ``WannierTools``,
   downstream DFT pipelines, custom analysis scripts that expect the
   plain-text Wannier90 format).
2. **Convert the model to a pybinding** ``Lattice`` — so you can use
   pybinding's solvers, KPM routines, eigenvalue plotters, and
   transport tools on top of the Tailwater Hamiltonian.
3. **Convert the model to a PythTB** ``tb_model`` — so you can use
   PythTB's band-path helpers, slab/wire builders, Berry-phase /
   Wannier-charge-centre routines, and the body of literature that
   targets PythTB.

All three are one-liners.


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

    from tailwater import tb_model, k_cart_from_frac
    import pybinding as pb

    model = tb_model.load("wannier90_hr.hdf5")
    lat   = model.to_pb()

    # Build a pybinding model and sample H(k) at any fractional k:
    pmod = pb.Model(lat, pb.translational_symmetry())
    pmod.set_wave_vector(k_cart_from_frac([0.0, 0.0, 0.0], model.uc))    # Gamma
    eig  = np.sort(np.linalg.eigvalsh(pmod.hamiltonian.todense()))

These eigenvalues match ``np.sort(np.linalg.eigvalsh(model.hamilton([0,0,0])))``
to float32 precision (~1e-6 eV).

.. important::

   Pybinding's :meth:`pb.Model.set_wave_vector` expects k in
   **Cartesian (rad/length)** — *not* fractional. Use
   :func:`tailwater.k_cart_from_frac` to convert from the fractional
   convention :class:`tbmodels.Model.hamilton` uses. Passing fractional
   k directly to ``set_wave_vector`` is the most common source of
   "the pybinding bands don't match the tbmodels bands" reports.

What ``.to_pb()`` does internally:

* Reads the on-site energies off the diagonal of the ``R = (0, 0, 0)``
  hop block (doubling them, see "Convention notes" below).
* Adds one ``Sublattice`` per Wannier orbital, with the position
  converted from fractional → Cartesian via ``pos_cart = pos_frac @ LM``.
* Iterates the hopping dict and adds each ``(R, i, j)`` to
  ``lat.add_one_hopping``. pybinding implies the Hermitian conjugate
  automatically, so duplicates returned by tbmodels are silently
  skipped.

Computing a full band structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    from tailwater import tb_model, k_cart_from_frac
    import pybinding as pb

    model = tb_model.load("wannier90_hr.hdf5")
    pmod  = pb.Model(model.to_pb(), pb.translational_symmetry())

    # Gamma -> M -> K -> Gamma on the (k_x, k_y) plane
    k_frac_path = np.array([
        [0.000, 0.000, 0],
        [0.500, 0.000, 0],
        [0.333, 0.333, 0],
        [0.000, 0.000, 0],
    ])

    bands = []
    for kf in k_frac_path:
        pmod.set_wave_vector(k_cart_from_frac(kf, model.uc))
        bands.append(np.sort(np.linalg.eigvalsh(pmod.hamiltonian.todense())))
    bands = np.array(bands)                # shape (Npts, num_wann)

Comparing this against ``[np.sort(np.linalg.eigvalsh(model.hamilton(kf))) for kf in k_frac_path]``
gives matching curves to ~1e-6 eV — both routes diagonalise the same
Hamiltonian, just with different per-orbital phase conventions on the
eigenvectors.

Convention notes
~~~~~~~~~~~~~~~~

These are the two conventions ``to_pb`` quietly handles for you; the
``CHANGELOG`` covers the same story for upgraders from 0.4.2 or
earlier.

* **On-site doubling.** ``tbmodels.Model.hamilton(k)`` constructs the
  Hamiltonian via ``Σ_R stored[R] e^{ikR}`` followed by ``H += H.c.``
  This H.c. step supplies the missing -R half for R ≠ 0, but at
  R=(0,0,0) it *doubles* the stored block. tbmodels therefore stores
  exactly **half** the physical on-site block. ``to_pb`` multiplies
  ``hop[(0,0,0)]`` by 2 before feeding it to pybinding so pybinding's
  H(k) recovers the full physical on-site contribution.
* **Position basis.** ``tbmodels.Model.pos`` is in fractional
  coordinates; pybinding's ``add_one_sublattice`` expects Cartesian.
  ``to_pb`` does the conversion ``pos_cart = pos_frac @ LM`` so the
  resulting pybinding ``Lattice`` has physically meaningful positions
  (its Brillouin-zone routines, real-space LDOS plotters, etc. all
  see the right geometry).


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


Converting to a PythTB ``tb_model``
-----------------------------------

Every model returned by :func:`tailwater.tb_model.load` also carries
a ``.to_pythtb()`` instance method:

.. code-block:: python

    from tailwater import tb_model

    model    = tb_model.load("wannier90_hr.hdf5")
    py_model = model.to_pythtb()

    # Sample H(k) at any fractional k:
    eig_gamma = py_model.solve_one([0.0, 0.0, 0.0])      # Γ
    eig_m     = py_model.solve_one([0.5, 0.0, 0.0])      # M (hex zone)

These eigenvalues match
``np.linalg.eigvalsh(model.hamilton([0,0,0]))`` to **float64
precision** (~5×10⁻¹⁴ eV) — a much tighter agreement than the
pybinding path, which is float32 (~10⁻⁶ eV).

The PythTB path is generally the easier of the two:

* PythTB takes orbital positions in **fractional** coordinates, the
  same convention ``tbmodels.Model.pos`` uses — no Cartesian
  conversion needed.
* PythTB's ``solve_one(k)`` accepts **fractional k directly** — no
  analogue of :func:`tailwater.k_cart_from_frac` is needed.
* PythTB ships rich first-class helpers for band paths, slabs/wires
  (``cut_piece``), supercells (``make_supercell``), and
  Wannier-centre / Berry-phase analyses.

Computing a band structure with PythTB's built-in helper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    from tailwater import tb_model
    import matplotlib.pyplot as plt

    model    = tb_model.load("wannier90_hr.hdf5")
    py_model = model.to_pythtb()

    # PythTB does the path interpolation for you:
    k_path, k_dist, k_node = py_model.k_path(
        [[0,0,0], [0.5,0,0], [0.333,0.333,0], [0,0,0]],
        nk=101, report=False,
    )
    bands = py_model.solve_all(k_path)                    # (num_wann, nk)

    fig, ax = plt.subplots()
    for band in bands:
        ax.plot(k_dist, band, lw=0.7, color="k")
    ax.set_xticks(k_node, [r"$\Gamma$", "M", "K", r"$\Gamma$"])
    ax.set_ylabel("E (eV)")
    fig.savefig("bands_pythtb.png", dpi=150)

Slabs and wires
~~~~~~~~~~~~~~~

PythTB's ``cut_piece`` makes a 1D / 2D slab from the bulk model. For
a 6-layer Bi\ :sub:`2`\ Se\ :sub:`3` slab terminated along the c-axis:

.. code-block:: python

    py_slab = py_model.cut_piece(num=6, fin_dir=2, glue_edgs=False)
    print(py_slab.get_num_orbitals())                 # 6 * 124 = 744

The resulting model is 2D-periodic (in-plane) and 0D along the
surface-normal direction — solve it the same way:

.. code-block:: python

    eig_2d = py_slab.solve_one([0.0, 0.0])            # Γ of the surface BZ

Both ``model.to_pythtb()`` and ``model.to_pb()`` produce models with
identical bulk Hamiltonians; pick whichever ecosystem (PythTB,
pybinding, or both) fits your downstream analysis.


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

.. autofunction:: tailwater.client._to_pythtb_method
   :no-index:

.. autofunction:: tailwater.client.k_cart_from_frac
   :no-index:

.. autofunction:: tailwater.hr_export.write_hr_output
   :no-index:

.. autofunction:: tailwater.hr_export.build_hr_model
   :no-index:

.. autofunction:: tailwater.hr_export.build_hr_model_fast
   :no-index:

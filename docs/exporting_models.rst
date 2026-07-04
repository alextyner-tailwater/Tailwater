Exporting models: sparse ``.npz``, HDF5, ``_hr.dat``, pybinding, PythTB, Kwant
==============================================================================

By **default the API returns the tight-binding Hamiltonian in sparse
form** — a ``wannier90_hr.npz`` (a :class:`tailwater.SparseHR`) that
stores only the non-zero hoppings, so it is O(N) in memory and file size
rather than O(N²). :func:`tailwater.tw_api_call` handles the two size
regimes for you:

* **Small systems** (< 30 atoms) — the ``.npz`` is **automatically
  converted to a dense tbmodels HDF5** on your machine and returned under
  ``paths["hdf5"]`` (the ``.npz`` is kept too, under ``paths["npz"]``), so
  every dense recipe on this page — ``tb_model.load("...hdf5")`` onward —
  works unchanged.
* **Large systems** — the result stays sparse (``paths["npz"]``), and a
  note is printed pointing at the conversions here. For big cells you
  should keep it sparse (see :ref:`why-sparsity`).

Force either format with ``tw_api_call(..., output_format="hdf5")`` or
``"sparse"``.

Importantly, the conversion functions below accept **either** a sparse
``.npz`` / :class:`~tailwater.SparseHR` **or** a dense HDF5 /
``tbmodels.Model``, so the same one-liner works whatever you're holding.


The sparse ``.npz`` format (``SparseHR``)
-----------------------------------------

Load a ``.npz`` with :class:`tailwater.SparseHR`:

.. code-block:: python

    from tailwater import SparseHR

    shr = SparseHR.load("wannier90_hr.npz")
    shr.num_wann      # number of Wannier orbitals (the H(k) matrix dimension)
    shr.nnz           # number of stored hoppings

Internally the ``.npz`` holds the Hamiltonian as a **COO sparse list of
hoppings** plus the on-site diagonal and (optionally) the geometry — a
dense ``[num_wann, num_wann]`` matrix is never formed:

=================  ======================================================
``on_site``        real on-site energy per orbital, shape ``[num_wann]``
``rows`` / ``cols``  orbital indices ``i``, ``j`` of each stored hopping
``Rs``             lattice vector ``R`` per hopping, shape ``[nnz, 3]``
``vals``           complex hopping amplitude ``H_ij(R)``
``cell``           3×3 lattice vectors (Å), when geometry was recovered
``positions``      per-orbital Cartesian positions, when available
=================  ======================================================

Only the *forward* half of each ``±R`` pair is stored (the Hermitian
conjugate at ``-R`` is implied) and the ``R = 0`` diagonal lives in
``on_site``. That is what makes it O(N): a ``.npz`` for a 14,000-orbital
moiré cell is a few MB, where the dense HDF5 would be tens of GB.


Convert the ``.npz`` to any format (one call, auto-detecting the input)
-----------------------------------------------------------------------

The top-level converters accept a :class:`~tailwater.SparseHR` / ``.npz``
path **or** a ``tbmodels.Model`` / ``.hdf5`` / ``_hr.dat`` and dispatch
automatically:

.. code-block:: python

    from tailwater import (as_tbmodels, to_hdf5, to_hr_dat,
                           to_pb, to_pythtb, to_kwant)

    npz = "wannier90_hr.npz"

    model      = as_tbmodels(npz)               # tbmodels.Model (dense)
    to_hdf5(npz,   "wannier90_hr.hdf5")         # tbmodels HDF5
    to_hr_dat(npz, "wannier90_hr.dat")          # Wannier90 _hr.dat
    pb_lattice = to_pb(npz)                      # pybinding.Lattice
    py_model   = to_pythtb(npz)                 # pythtb model
    syst, lat  = to_kwant(npz)                  # kwant (Builder, lattice)

    # the identical calls work on a dense HDF5 / _hr.dat, too:
    to_hr_dat("wannier90_hr.hdf5", "wannier90_hr.dat")

``as_tbmodels(npz)`` is the bridge to the rest of this page: once you have
the ``tbmodels.Model`` (or the auto-converted HDF5 for a small system),
every pybinding / PythTB / Kwant / WannierBerri recipe below applies.

.. note::

   ``_hr.dat`` and HDF5 are **dense** on-disk formats — size grows as
   ``num_R · num_wann²``. They are guarded for very large systems; pass
   ``max_wann=`` to :meth:`SparseHR.to_hr_dat` / :meth:`SparseHR.to_hdf5`
   to override the guard if you really intend to write a huge file.


Staying sparse: pybinding, Kwant, and built-in solvers
------------------------------------------------------

For large systems you usually do **not** want to densify at all.
:class:`~tailwater.SparseHR` builds pybinding and Kwant models **straight
from the COO list** (no dense matrix is ever formed) and carries its own
sparse solvers:

.. code-block:: python

    from tailwater import SparseHR

    shr = SparseHR.load("wannier90_hr.npz")

    # --- built-in sparse spectra (large num_wann OK) ---
    Hk = shr.Hk([0.0, 0.0, 0.0])                   # scipy sparse H(k) at Γ
    w  = shr.eigsh_near_fermi([0, 0, 0], e_fermi=0.0, num=20)  # 20 states near E_F
    Rd = shr.hr_dict()                             # {R: scipy.sparse.csr_matrix}

    # --- hand the sparse model to pybinding / Kwant (built from the COO) ---
    pb_lattice = shr.to_pb()                       # pybinding.Lattice
    syst, lat  = shr.to_kwant()                    # kwant (Builder, lattice)

``eigsh_near_fermi`` uses a shift-invert sparse eigensolver, so you can
get just the handful of bands nearest the Fermi level for a Hamiltonian
far larger than a dense ``H(k)`` could hold. ``hr_dict`` returns the
``H(R)`` blocks as scipy sparse matrices to feed your own KPM /
Green's-function / transport code. ``to_pb`` and ``to_kwant`` are
sparse-native and scale to large ``num_wann`` — pybinding and Kwant are
themselves sparse solvers, so the whole pipeline stays O(N).


.. _why-sparsity:

Why maintaining sparsity matters for large systems
--------------------------------------------------

A **dense** Hamiltonian stores every ``[num_wann, num_wann]`` block for
every lattice vector ``R``: memory and file size scale as
**O(num_R · num_wann²)**. But a physical tight-binding Hamiltonian is
*sparse* — each orbital hops only to a bounded number of neighbours — so
the number of non-zeros scales just as **O(num_wann)**. For large cells
the difference is decisive:

* A twisted-bilayer moiré cell with ~14,000 orbitals is a **few MB** as a
  ``.npz`` but **tens of GB** as a dense ``_hr.dat`` / HDF5 — the dense
  form often cannot be written, let alone loaded into RAM.
* Diagonalising a dense ``H(k)`` is O(num_wann³) and needs the entire
  matrix resident; the sparse shift-invert path
  (:meth:`SparseHR.eigsh_near_fermi`) touches only the non-zeros and
  returns just the near-Fermi bands you ask for.
* pybinding and Kwant are sparse-native, so converting via
  :meth:`SparseHR.to_pb` / :meth:`SparseHR.to_kwant` keeps memory and
  compute O(N) end to end.

**Rule of thumb:** for small systems, let ``tw_api_call`` convert to HDF5
and use the dense recipes below. For large systems, **keep the ``.npz``
sparse** — write ``_hr.dat`` / HDF5 only if an external tool demands it,
and prefer the sparse pybinding / Kwant / built-in solvers above.


Working with the dense ``tbmodels.Model``
-----------------------------------------

The sections below operate on a dense ``tbmodels.Model`` — the HDF5 you
get for a small system (``tb_model.load("wannier90_hr.hdf5")``), or
``as_tbmodels("wannier90_hr.npz")`` for a converted sparse model. From a
dense model there are four common downstream needs, each a one-liner:

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
4. **Convert the model to a Kwant** ``Builder`` — so you can use
   Kwant's transport machinery (leads, ``smatrix``, ``greens_function``),
   the wraparound trick for bulk H(k), or any of Kwant's
   sample-builder utilities.


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


Converting to a Kwant ``Builder``
---------------------------------

Every model returned by :func:`tailwater.tb_model.load` also carries
a ``.to_kwant()`` instance method:

.. code-block:: python

    import numpy as np, kwant
    from tailwater import tb_model

    model   = tb_model.load("wannier90_hr.hdf5")
    builder = model.to_kwant()              # kwant.Builder, 3D periodic

    # For bulk H(k) sampling, wrap-around and finalise:
    syst = kwant.wraparound.wraparound(builder).finalized()

    # Kwant's wraparound takes 2π·k_frac as k_x, k_y, k_z (the
    # per-cell Bloch phase) — NOT Cartesian rad/length like pybinding.
    k_frac = np.array([0.5, 0.0, 0.0])
    phase  = 2 * np.pi * k_frac
    H      = syst.hamiltonian_submatrix(
        params=dict(k_x=phase[0], k_y=phase[1], k_z=phase[2]),
    )
    eig    = np.sort(np.linalg.eigvalsh(H))

This matches ``np.linalg.eigvalsh(model.hamilton([0.5, 0, 0]))`` to
**float64 precision** (~10⁻¹³ eV) — Kwant is double-precision
internally.

.. important::

   Kwant's :class:`kwant.wraparound` is the one place in this
   ecosystem where k is **neither** fractional **nor** Cartesian
   rad/length. The ``k_x``/``k_y``/``k_z`` parameters are
   ``2π · k_frac`` — the per-cell Bloch phase, independent of cell
   size. (Use :func:`tailwater.k_cart_from_frac` only for pybinding's
   ``set_wave_vector``.) Passing a Cartesian k to Kwant is the most
   common source of "the Kwant bands don't match the tbmodels bands"
   reports.

What ``.to_kwant()`` returns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It returns an **unfinalised** ``kwant.Builder`` with a 3D
``kwant.TranslationalSymmetry``, one site per Wannier orbital. From
there the user has two natural next steps:

* **Bulk H(k)** — wrap the Builder via
  ``kwant.wraparound.wraparound`` (as in the snippet above) and
  finalise.
* **Transport / scattering** — build a finite scattering region on
  top of the bulk Builder, attach leads via
  ``kwant.Builder(kwant.TranslationalSymmetry(...))`` for each lead,
  and call ``kwant.smatrix`` / ``kwant.greens_function``.

  A complete end-to-end recipe (quantum wire of a Bi\ :sub:`2`\
  Se\ :sub:`3` hr-model + two semi-infinite leads + G(E) sweep) lives
  at ``examples/05_kwant_scattering.py``. The full Kwant tutorial is
  at https://kwant-project.org/doc/.

  One Kwant API quirk to know: ``bulk.symmetry.periods`` is a custom
  Kwant array class that doesn't support direct integer indexing.
  Cast to a plain NumPy array first::

    periods = np.asarray(bulk.symmetry.periods)        # (3, 3)
    sym_x   = kwant.TranslationalSymmetry(periods[0])  # transport axis

Computing a band structure with Kwant
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np, kwant
    import matplotlib.pyplot as plt
    from tailwater import tb_model

    model = tb_model.load("wannier90_hr.hdf5")
    syst  = kwant.wraparound.wraparound(model.to_kwant()).finalized()

    # Gamma -> M -> K -> Gamma  (linearly sampled fractional path)
    nodes = np.array([[0,0,0], [0.5,0,0], [0.333,0.333,0], [0,0,0]])
    path  = np.vstack([np.linspace(nodes[i], nodes[i+1], 33)
                       for i in range(len(nodes) - 1)])

    bands = []
    for k_frac in path:
        phase = 2 * np.pi * k_frac
        H = syst.hamiltonian_submatrix(
            params=dict(k_x=phase[0], k_y=phase[1], k_z=phase[2]),
        )
        bands.append(np.sort(np.linalg.eigvalsh(H)))
    bands = np.array(bands)                       # (Npts, num_wann)

    fig, ax = plt.subplots()
    ax.plot(bands, lw=0.7, color="k")
    ax.set_ylabel("E (eV)")
    fig.savefig("bands_kwant.png", dpi=150)

The bands are identical to those from ``model.hamilton(k_frac)`` or
``model.to_pythtb().solve_one(k_frac)`` at every k — Kwant just
gives you the rest of the transport ecosystem for free.


Using the model with WannierBerri
---------------------------------

WannierBerri (Berry-curvature integrator for transport quantities,
optical responses, Wannier-charge centres, etc.) reads
``tbmodels.Model`` directly — no ``model.to_X()`` method needed.
Just pass the loaded model:

.. code-block:: python

    import wannierberri as wb
    from tailwater import tb_model

    model  = tb_model.load("wannier90_hr.hdf5")
    sys_wb = wb.system.System_R.from_tbmodels(model, berry=True)

    grid = wb.Grid(sys_wb, NK=(8, 8, 8), NKFFT=(4, 4, 4))

    Efermi = np.linspace(-3, 3, 61)
    result = wb.run(
        sys_wb, grid=grid,
        calculators={
            "dos":   wb.calculators.static.DOS           (Efermi=Efermi, tetra=True),
            "ohmic": wb.calculators.static.Ohmic_FermiSea(Efermi=Efermi),
            "ahc":   wb.calculators.static.AHC           (Efermi=Efermi),
        },
        parallel=False, symmetrize=False, dump_results=False,
    )

``berry=True`` tells WannierBerri to construct the position matrix
elements ``<0,n|r|R,m>`` from ``model.pos`` — these are what every
Berry-curvature-derived calculator (AHC, optical conductivity,
Wannier charge centres) needs.

A complete worked recipe with DOS, longitudinal conductivity, and
AHC vs. Fermi energy is at ``examples/06_wannierberri_conductivity.py``.

Spin Hall conductivity (``SHC`` and friends) need an extra ingredient:
the spin matrix elements ``<n,0 | S^alpha | m,R>``. WannierBerri's
``from_tbmodels(..., spin=True)`` is documented but doesn't actually
populate these — tbmodels carries only the Hamiltonian. Tailwater
synthesises them from the *known* sigma_z-eigenstate structure of the
18-orbital Wannier basis via
:func:`tailwater.wb_system_with_spin`:

.. code-block:: python

    import numpy as np, wannierberri as wb
    from tailwater import tb_model, wb_system_with_spin

    model = tb_model.load("wannier90_hr.hdf5")
    sys   = wb_system_with_spin(model)        # SS_R populated; berry=True too

    Efermi = np.linspace(-2.0, 2.0, 41)
    grid   = wb.Grid(sys, NK=(8, 8, 8), NKFFT=(4, 4, 4))
    result = wb.run(
        sys, grid=grid,
        calculators={
            "shc": wb.calculators.static.SHC(
                Efermi=Efermi,
                kwargs_formula={"spin_current_type": "simple"},
            ),
        },
        parallel=False, symmetrize=False, dump_results=False,
    )
    shc = np.asarray(result.results["shc"].data)   # (Nef, 3, 3, 3) in (ℏ/e)·S/cm

The function infers the sigma_z eigenstate doublets from the model's
geometric structure by default — it walks the atomic positions to
group Wannier functions by atom, then pairs consecutive Kramers
partners and verifies their on-site energies match. A worked
end-to-end recipe with Bi\ :sub:`2`\ Se\ :sub:`3` lives at
``examples/07_spin_hall_conductivity.py``; the in-gap plateau in
:math:`\sigma^{z}_{xy}` is the topological signature.

WannierBerri brings its own optional dependencies — install them
explicitly when needed:

.. code-block:: bash

    pip install wannierberri numba   # numba: tetrahedron integration


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

Sparse Hamiltonian + format-detecting converters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: tailwater.SparseHR
   :members: load, save, Hk, eigvals_grid, eigsh_near_fermi, hr_dict,
             to_tbmodels, to_hdf5, to_hr_dat, to_pb, to_kwant
   :no-index:

.. autofunction:: tailwater.convert.as_tbmodels
   :no-index:

.. autofunction:: tailwater.convert.to_hdf5
   :no-index:

.. autofunction:: tailwater.convert.to_hr_dat
   :no-index:

.. autofunction:: tailwater.convert.to_pb
   :no-index:

.. autofunction:: tailwater.convert.to_pythtb
   :no-index:

.. autofunction:: tailwater.convert.to_kwant
   :no-index:

Dense (tbmodels) converters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: tailwater.client._to_pb_method
   :no-index:

.. autofunction:: tailwater.client._to_pythtb_method
   :no-index:

.. autofunction:: tailwater.client._to_kwant_method
   :no-index:

.. autofunction:: tailwater.client.k_cart_from_frac
   :no-index:

.. autofunction:: tailwater.hr_export.write_hr_output
   :no-index:

.. autofunction:: tailwater.hr_export.build_hr_model
   :no-index:

.. autofunction:: tailwater.hr_export.build_hr_model_fast
   :no-index:

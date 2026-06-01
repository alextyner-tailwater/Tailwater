Speeding up surface-state calculations
======================================

``SurfaceGreensFunction`` and ``FermiArcMap`` solve a Lopez-Sancho
recursion at every k-point on every energy. For realistic problems
that's tens of thousands of dense complex-matrix factorizations — so
they're the most CPU-hungry calculators in the package. Two knobs
on each class control how that work is laid out:

* **``n_jobs``** — k-point parallelism across worker processes.
* **``chunk_size``** — energy-axis batch size on each k-point.

Both default to safe values, so existing code keeps working
unchanged. Setting ``n_jobs=-1`` is the single biggest win.


TL;DR
-----

Anywhere you call ``SurfaceGreensFunction`` or ``FermiArcMap``, just
add ``n_jobs=-1``:

.. code-block:: python

    sgf = SurfaceGreensFunction(
        model, surface=np.eye(3),
        energies=np.linspace(-1, 1, 201),
        k_path=[[0, 0.5, 0], [0, 0, 0], [0.333, 0.333, 0]],
        k_labels=["M", r"$\Gamma$", "K"],
        n_jobs=-1,                                 # <-- use every core
    ).run()

Results are bit-exact relative to ``n_jobs=1`` — no precision trade-off.


How much faster?
----------------

Measured on a Bi\ :sub:`2`\ Se\ :sub:`3` slab (124 Wannier orbitals,
slab dim 248), 16-core CPU:

.. list-table::
   :header-rows: 1
   :widths: 40 20 20 20

   * - Run
     - Serial
     - ``n_jobs=-1``
     - speedup
   * - SurfaceGF, ``Nk=21, Nw=11, NN=8``
     - 28.9 s
     - 9.3 s
     - 3.1×
   * - SurfaceGF, ``Nk=51, Nw=51, NN=10``
     - ~480 s
     - 79 s
     - 6×
   * - FermiArcMap, ``Nx=Ny=12, NN=8``
     - 14 s
     - 3.5 s
     - 4×

Bigger problems get bigger speedups because the fixed worker-startup
cost (~1 s/worker on macOS, faster on Linux) amortizes away. On a
32+ core Linux machine, expect 10-12× on the larger problems.


``n_jobs`` — k-point parallelism
--------------------------------

Each k-point of the surface Green's function is fully independent of
every other, so fanning them out across worker processes via
``joblib`` is essentially free correctness-wise.

==================  ===================================================
``n_jobs=...``      Behavior
==================  ===================================================
``1`` *(default)*   Serial. No worker processes spawned.
``-1``              Use every physical CPU core on the host.
``N`` (any int)     Use exactly ``N`` worker processes.
==================  ===================================================

Each worker pins itself to **one BLAS thread** to avoid
oversubscription — without that, ``N`` workers each spawning ``N``
BLAS threads would thrash the cache and run slower than serial.

A small caveat on macOS / Windows: ``joblib`` spawns workers fresh
(not fork), so each one re-imports torch on startup. That adds ~1 s
per worker, paid once per ``.run()`` call. For runs longer than ~10
s the overhead is negligible; for runs that already complete in 2-3
s, ``n_jobs=-1`` may not be worth it.


``chunk_size`` — energy-axis batching
-------------------------------------

For each k-point the Lopez-Sancho recursion is run as a single
batched LAPACK pass over (a chunk of) the energy grid. Larger
chunks ⇒ less Python dispatch overhead, more peak memory. The
defaults are chosen so memory stays bounded even for very dense
energy grids on thick slabs:

============================  ================  ================
Class                         Default chunk     What it batches
============================  ================  ================
``SurfaceGreensFunction``     ``256``           ``Nw`` energies
``FermiArcMap``               ``128``           ``Nx·Ny`` k-points
============================  ================  ================

Pass a smaller value (e.g. ``32``) if you hit memory pressure on a
thick slab, or a larger value if memory is plentiful and you want
every (k, w) pair in one shot.


Choosing what to set
--------------------

* **Always set ``n_jobs=-1``** for any nontrivial run. It's a 3-10×
  speedup with no precision cost.
* **Leave ``chunk_size`` alone** unless you hit a memory error.
  Lower it (``chunk_size=32`` or ``16``) to fix the error.
* **Don't combine ``n_jobs=-1`` with ``device="cuda"``** — the GPU
  is already a batched accelerator; spawning multiple host
  processes that each grab a CUDA context will fight over the GPU.
  Use one or the other.


Implementation note
-------------------

The Lopez-Sancho recursion uses two LAPACK ``solve`` calls per
iteration with the same coefficient matrix; these are combined into
a single multi-RHS solve so the LU factorization happens once. That
plus batching across the energy axis accounts for the ~25-35%
speedup at ``n_jobs=1``; the rest of the speedup at ``n_jobs > 1``
comes from real cross-process parallelism on independent k-points.

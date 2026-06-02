Examples
========

Runnable scripts live in the package's ``examples/`` directory. Each one
is self-contained and operates on artifacts produced by the API.

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - File
     - What it does
   * - ``examples/01_basic_api_call.py``
     - Default mode: upload a pymatgen ``Structure``, download a
       tbmodels HDF5 hr-model + parsed ``.win``.
   * - ``examples/02_subspace_projection.py``
     - One ``project=True`` API call → ``subspace_projection`` →
       projected ``_hr.dat`` + basis JSON.
   * - ``examples/03_surface_analysis.py``
     - ``BulkDOS`` / ``SurfaceSpectralDensity`` / ``SurfaceGreensFunction`` /
       ``FermiArcMap`` on the HDF5 hr-model.
   * - ``examples/04_band_structure.py``
     - ``bulk_band_structure`` in manual (custom k-points) and auto
       (seekpath-derived) modes.
   * - ``examples/05_kwant_scattering.py``
     - ``model.to_kwant()`` → finite scattering region + two
       semi-infinite leads → ``kwant.smatrix`` → two-terminal
       conductance G(E). End-to-end quantum-transport recipe.

Each script targets a single workflow stage so customers can pick the
slice they care about and run it in isolation. They share three
conventions worth knowing about:

#. **Single material per script.** Every example takes one
   ``Structure`` (from a ``.cif`` file) at the top and threads it
   through the pipeline. Multi-material batches are not in the
   examples by design — looping over a CIF directory is one line
   of Python on top.

#. **Default device is CPU.** Every post-processing class accepts a
   ``device="cuda"`` argument; the examples leave it at ``"cpu"`` so
   they run anywhere. Switching to GPU is a one-line change at the
   top of any of them.

#. **Results are saved as both figures AND raw arrays.** Every
   post-processing class returns a Result dataclass with
   ``.figure*`` matplotlib objects AND an ``.as_dict()`` for
   ``np.savez``. Examples demonstrate both so customers see the full
   surface.

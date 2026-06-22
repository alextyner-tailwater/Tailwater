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
   * - ``examples/06_wannierberri_conductivity.py``
     - ``wannierberri.system.System_R.from_tbmodels(model)`` → DOS,
       Ohmic (longitudinal) conductivity tensor, and anomalous Hall
       conductivity vs Fermi energy on a uniform MP k-mesh.
   * - ``examples/07_spin_hall_conductivity.py``
     - ``tailwater.wb_system_with_spin(model)`` builds an SHC-ready
       WannierBerri ``System`` (synthesises the spin matrix elements
       ``SS_R`` from the Tailwater orbital basis), then sweeps the
       intrinsic spin Hall conductivity vs Fermi energy. Plateau in
       the gap is the topological signature.
   * - ``examples/08_quantum_geometry.py``
     - Three more static WannierBerri quantities versus Fermi energy
       on a single sweep: cumulative DOS (carrier-count integral),
       quantum metric (Fermi-sea integral), and nonlinear Drude
       conductivity (a centrosymmetric-zero sanity check on Bi\
       :sub:`2`\ Se\ :sub:`3`).
   * - ``examples/09_optical_conductivity.py``
     - Frequency-dependent Kubo optical conductivity
       :math:`\sigma_{\alpha\beta}(\omega)` from
       ``wb.calculators.dynamic.OpticalConductivity``, plotted as
       Re/Im parts of the in-plane :math:`\sigma_{xx}` and
       out-of-plane :math:`\sigma_{zz}` channels.
   * - ``examples/10_multi_material_finetune.py``
     - Heads-only fine-tune on a SET of (API-embedding,
       user-Wannier-Hamiltonian) pairs via
       ``prepare_finetune_target`` + ``finetune_heads_multi``.
       Subspace eigenvalue loss masked outside a per-run energy
       window; validation loss reported every N epochs; best-val
       checkpoint kept alongside the final one.
   * - ``examples/11_surface_charge_density.py``
     - ``surface_charge_density`` — general ``(hkl)`` slab from any
       Wannier ``H(R)`` (Tailwater HDF5 *or* DFT ``wannier90_hr.dat``)
       → real-space surface charge-density heat maps (top view +
       side cross-section). ``energy_window=`` isolates
       near-\ :math:`E_F` states to image topological surface states;
       ``supercell_self_check`` verifies the supercell remap to machine
       precision first.

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

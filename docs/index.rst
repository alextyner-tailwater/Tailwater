tailwater
=========

**Client + post-processing toolkit for the Tailwater Wannier-Hamiltonian inference API.**

``tailwater`` lets you upload a crystal structure to the Tailwater API,
receive a tight-binding Hamiltonian, optionally fine-tune the output
heads on customer-side targets, and run band-structure / DOS /
surface-state analyses locally — all from one pip-installable package.

.. code-block:: bash

    pip install tailwater

----

Three workflow layers
---------------------

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Layer
     - What you get
   * - :doc:`HTTP client <api/client>`
     - ``tw_api_call`` / ``SparseHR`` / ``tb_model`` — upload a pymatgen
       ``Structure`` and receive the tight-binding model (a sparse
       ``.npz`` by default; small systems auto-converted to HDF5) +
       parsed ``.win`` file, or any intermediate inference artifact.
   * - :doc:`Subspace projection <api/finetune_heads>`
     - ``subspace_projection`` — fine-tune the output heads to reproduce
       the predicted Hamiltonian's eigenvalues in a narrow near-Fermi
       window (a compact, downfolded model).
   * - :doc:`Post-processing <api/wannier_wizard>`
     - ``BulkDOS`` / ``SurfaceGreensFunction`` / ``FermiArcMap`` /
       ``BandStructure`` — band-structure, DOS, surface-state, and
       Fermi-arc analyses on the HDF5 model.
   * - :doc:`Surface charge density <api/surface_charge>`
     - ``surface_charge_density`` — real-space surface charge-density
       heat maps of a general ``(hkl)`` slab, from any Wannier
       ``H(R)`` (Tailwater HDF5 or DFT ``wannier90_hr.dat``).

----

Quick start
-----------

.. code-block:: python

    from pymatgen.core import Structure
    from tailwater import tw_api_call, subspace_projection, as_tbmodels, SurfaceGreensFunction

    structure = Structure.from_file("MyMaterial.cif")

    # 1) One API call, one credit — embeddings.pt + sparse wannier90_hr.npz
    paths = tw_api_call(structure, "user", "pw", "./outputs", "my_mat",
                        project=True)

    # 2) Fine-tune heads to the Hamiltonian's eigenvalues in [-2, 2] eV of E_F
    subspace_projection(
        start_lr=1e-4, end_lr=1e-5, num_epochs=20,
        energy_range=(-2.0, 2.0), decay_sigma=0.5,
        device="cpu",
        save_path="./projection_out",
        embed_path=paths["embeddings"],
        hr_npz_path=paths["npz"],
    )

    # 3) Surface Green's function (Lopez-Sancho).
    #    n_jobs=-1 fans the k-points across every CPU core for a 3-10x
    #    speedup; see :doc:`performance` for the full story.
    import numpy as np
    model  = as_tbmodels(paths["npz"])   # dense tbmodels.Model from the sparse .npz
    result = SurfaceGreensFunction(
        model, surface=np.eye(3),
        energies=np.linspace(-1, 1, 201),
        k_path=[[0, 0.5, 0], [0, 0, 0], [0.333, 0.333, 0]],
        k_labels=["M", r"$\Gamma$", "K"],
        n_jobs=-1,
    ).run()
    result.figure_top.savefig("surface_top.png")


.. toctree::
   :maxdepth: 2
   :caption: Getting started
   :hidden:

   installation
   quickstart
   examples


.. toctree::
   :maxdepth: 2
   :caption: Guides
   :hidden:

   fermi_alignment
   fermi_arcs
   exporting_models
   performance


.. toctree::
   :maxdepth: 2
   :caption: API reference
   :hidden:

   api/client
   api/finetune_heads
   api/hr_export
   api/wannier_wizard
   api/surface_charge
   api/constants


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

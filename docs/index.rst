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
     - ``tw_api_call`` / ``tb_model`` — upload a pymatgen ``Structure``
       and receive an HDF5 tight-binding model + parsed ``.win`` file,
       or any of the intermediate inference artifacts.
   * - :doc:`Subspace projection <api/finetune_heads>`
     - ``subspace_projection`` — fine-tune the output heads on
       supplier-provided embeddings to project predictions into a
       narrow near-Fermi energy window.
   * - :doc:`Post-processing <api/wannier_wizard>`
     - ``BulkDOS`` / ``SurfaceGreensFunction`` / ``FermiArcMap`` /
       ``BandStructure`` — band-structure, DOS, surface-state, and
       Fermi-arc analyses on the HDF5 model.

----

Quick start
-----------

.. code-block:: python

    from pymatgen.core import Structure
    from tailwater import tw_api_call, subspace_projection, tb_model, SurfaceGreensFunction

    structure = Structure.from_file("MyMaterial.cif")

    # 1) One API call, one credit — receive every artifact for downstream work
    paths = tw_api_call(structure, "user", "pw", "./outputs", "my_mat",
                        project=True)

    # 2) Fine-tune heads + project to [-2, 2] eV around E_F
    subspace_projection(
        start_lr=5e-5, end_lr=5e-7, num_epochs=20,
        energy_range=(-2.0, 2.0), decay_sigma=1.0,
        device="cpu",
        save_path="./projection_out",
        embed_path=paths["embeddings"],
        graph_output_path=paths["graph_output"],
    )

    # 3) Surface Green's function (Lopez-Sancho).
    #    n_jobs=-1 fans the k-points across every CPU core for a 3-10x
    #    speedup; see :doc:`performance` for the full story.
    import numpy as np
    model  = tb_model.load(paths["hdf5"])
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
   performance


.. toctree::
   :maxdepth: 2
   :caption: API reference
   :hidden:

   api/client
   api/finetune_heads
   api/wannier_wizard
   api/constants


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Quick start
===========

End-to-end: get a tight-binding Hamiltonian from the API, project it
into a near-Fermi subspace, and run a surface Green's function on
the result.

.. note::

   This guide assumes you have ``tailwater`` installed
   (``pip install tailwater``) and a username/password issued by the
   Tailwater team. The client talks to the hosted API at
   ``https://api.tailwater.io`` automatically — no configuration needed.

1. Get the artifacts from the API
---------------------------------

``project=True`` returns all four bundle artifacts in one credit-billed call:

.. code-block:: python

    from pymatgen.core import Structure
    from tailwater import tw_api_call

    structure = Structure.from_file("MyMaterial.cif")
    paths = tw_api_call(
        structure   = structure,
        user        = "acme-research",
        password    = "...",
        output_path = "./outputs",
        filename    = "my_material",
        project     = True,
    )
    # paths = {"hdf5": "...", "embeddings": "...",
    #          "graph_output": "...", "win": "..."}

The returned dict always contains a ``"win"`` key — the parsed
``wannier90.win`` file the server actually ran inference on, useful
for tracing graph-construction differences across API and offline
runs.

2. (Optional) Fine-tune the heads to fit a near-Fermi window
------------------------------------------------------------

.. code-block:: python

    from tailwater import subspace_projection

    subspace_projection(
        start_lr          = 5e-5,
        end_lr            = 5e-7,
        num_epochs        = 20,
        energy_range      = (-2.0, 2.0),       # eV, relative to E_F
        decay_sigma       = 1.0,
        device            = "cpu",
        save_path         = "./projection_out",
        embed_path        = paths["embeddings"],
        graph_output_path = paths["graph_output"],
        loss_mode         = "subspace",         # default
    )

After training, ``./projection_out/`` contains a fine-tuned heads
checkpoint, a projected (subspace-restricted) ``_hr.dat``, and a
``.basis.json`` describing the orbital basis of the projection.

3. Analyze the model
--------------------

.. code-block:: python

    import numpy as np
    from tailwater import tb_model, SurfaceGreensFunction, BulkDOS, bulk_band_structure

    model = tb_model.load(paths["hdf5"])

    # Bulk DOS (KPM, k-mesh averaged)
    dos = BulkDOS(model, k_mesh=(8, 8, 8), energies=(-4, 4)).run()
    dos.figure.savefig("bulk_dos.png")

    # Surface Green's function (Lopez-Sancho)
    sgf = SurfaceGreensFunction(
        model, surface=np.eye(3),
        energies=np.linspace(-1, 1, 201),
        k_path=[[0, 0.5, 0], [0, 0, 0], [0.333, 0.333, 0]],
        k_labels=["M", r"$\Gamma$", "K"],
    ).run()
    sgf.figure_top.savefig("surface_top.png")

    # Bulk band structure (auto k-path via seekpath, if installed)
    fig = bulk_band_structure(model, auto=True, structure=structure,
                              spacing=0.02, e_range=(-3, 3))
    fig.savefig("bands.png")

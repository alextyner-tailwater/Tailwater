Installation
============

Core install
------------

.. code-block:: bash

    pip install tailwater

The base install pulls in everything needed for the three workflow
layers — HTTP client, subspace projection, and post-processing —
including the heavy ML stack (``torch``, ``torch-geometric``,
``e3nn``, ``tbmodels``) and the structural plumbing (``pymatgen``,
``scipy``, ``matplotlib``, ``tqdm``, ``h5py``).

Optional extras
---------------

A few small features depend on extra packages that aren't pulled in
by default. Install with the bracket syntax:

.. code-block:: bash

    pip install "tailwater[scatter]"     # torch-scatter, only if missing
    pip install "tailwater[seekpath]"    # auto k-path for bulk_band_structure
    pip install "tailwater[dev]"         # pytest, ruff, build, twine

.. list-table::
   :widths: 20 30 50
   :header-rows: 1

   * - Extra
     - Adds
     - When you need it
   * - ``scatter``
     - ``torch-scatter>=2``
     - Your torch / torch-geometric install doesn't already ship
       ``torch_scatter`` and you hit an ``ImportError`` during
       fine-tuning.
   * - ``seekpath``
     - ``seekpath>=2``
     - You want ``bulk_band_structure(..., auto=True, structure=...)``
       to auto-derive the high-symmetry k-path.
   * - ``dev``
     - ``pytest``, ``ruff``, ``build``, ``twine``
     - You're working on ``tailwater`` itself.

pybinding (separate package, not an extra)
-------------------------------------------

If you want to use ``subspace_projection`` (which writes a projected
hr-model HDF5 after fine-tuning) or ``tb_model.load(...).to_pb()``,
install ``pybinding-dev`` directly — it is **not** a ``tailwater`` extra,
just a regular PyPI package:

.. code-block:: bash

    pip install pybinding-dev

Calling those features without it installed raises a clear ``ImportError``
that points you back at this command.

Supported Python
----------------

Python 3.9 – 3.12. The package is pure Python; all heavy compute is
delegated to the dependencies.

Verifying the install
---------------------

.. code-block:: python

    import tailwater
    print(tailwater.__version__)

    from tailwater import tw_api_call, subspace_projection, BulkDOS
    print(tw_api_call.__doc__.splitlines()[0])


Getting API access
------------------

By default, ``tw_api_call(...)`` talks to the hosted Tailwater API at
**https://api.tailwater.io** — no configuration required beyond a username
and password issued by the Tailwater team. Authentication is HTTP Basic;
every successful inference call decrements your credit balance by one.

To request an account, email the Tailwater team. To check your balance
without spending a credit:

.. code-block:: python

    from tailwater import remaining_credits
    print(remaining_credits("user", "pw"))     # -> int

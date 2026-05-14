Post-processing: bands, DOS, surface states, Fermi arcs
=======================================================

Calculator classes
------------------

Each calculator accepts either an HDF5 path (``str``) or an in-memory
``tbmodels.Model``; the ``.run()`` method returns a typed Result
dataclass with raw NumPy arrays AND matplotlib Figures.

.. autoclass:: tailwater.wannier_wizard.BulkDOS
   :members:
   :show-inheritance:

.. autoclass:: tailwater.wannier_wizard.SurfaceSpectralDensity
   :members:
   :show-inheritance:

.. autoclass:: tailwater.wannier_wizard.SurfaceGreensFunction
   :members:
   :show-inheritance:

.. autoclass:: tailwater.wannier_wizard.FermiArcMap
   :members:
   :show-inheritance:

.. autoclass:: tailwater.wannier_wizard.BandStructure
   :members:
   :show-inheritance:


Convenience function
--------------------

.. autofunction:: tailwater.wannier_wizard.bulk_band_structure


Result dataclasses
------------------

.. autoclass:: tailwater.wannier_wizard.BulkDOSResult
   :members:
.. autoclass:: tailwater.wannier_wizard.SurfaceSpectralDensityResult
   :members:
.. autoclass:: tailwater.wannier_wizard.SurfaceGreensFunctionResult
   :members:
.. autoclass:: tailwater.wannier_wizard.FermiArcMapResult
   :members:
.. autoclass:: tailwater.wannier_wizard.BandStructureResult
   :members:


k-path helper
-------------

.. autofunction:: tailwater.wannier_wizard.generate_k_path

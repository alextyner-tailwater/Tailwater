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

   See :doc:`../fermi_arcs` for a worked Bi\ :sub:`2`\ Se\ :sub:`3`
   example, surface-basis guidance, and tips for choosing the energy
   and grid resolution.

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


Fermi / band-edge alignment
---------------------------

For semiconductors and insulators, anchor the energy zero to the
valence band maximum so all calculators downstream share a physically
natural reference — see :doc:`../fermi_alignment` for the full guide.

.. autofunction:: tailwater.wannier_wizard.compute_band_edges

.. autofunction:: tailwater.wannier_wizard.align_to_vbm

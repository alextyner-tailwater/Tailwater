Model assembly and hr-file I/O
==============================

Helpers for building a ``tbmodels.Model`` from the API's dense head
predictions and writing it out as HDF5 or a Wannier90-style
``_hr.dat`` file.

See :doc:`../exporting_models` for the customer-facing walkthrough,
including the ``model.to_pb()`` pybinding-conversion path.

.. autofunction:: tailwater.hr_export.write_hr_output

.. autofunction:: tailwater.hr_export.build_hr_model

.. autofunction:: tailwater.hr_export.build_hr_model_fast

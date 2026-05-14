tbmodels assembly (hr_export)
=============================

Converts the model's dense head output into a ``tbmodels.Model``.
The fast variant is a vectorized rewrite of the reference loop that
preserves byte-exact equivalence with the notebook-style nested-loop
assembly — see the module docstring for the gotchas.

.. automodule:: tailwater.hr_export
   :members:
      build_hr_model,
      build_hr_model_fast,
      write_hr_output
   :member-order: bysource

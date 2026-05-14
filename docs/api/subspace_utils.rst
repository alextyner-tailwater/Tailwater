Subspace loss helpers
=====================

These are the loss-function building blocks ``subspace_projection`` uses
under the hood; expose them directly if you want to plug a custom
training loop on top.

.. automodule:: tailwater.subspace_utils
   :members:
      Subspace_H_MSE_Loss,
      Subspace_EigLoss,
      Eigenvalue_Only_Loss,
      make_eigenvalue_only_data,
      build_subspace_active_mask,
      write_subspace_basis_file,
      SPATIAL_BASIS_LABELS,
      SPIN_BASIS_LABELS
   :member-order: bysource

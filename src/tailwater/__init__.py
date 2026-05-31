"""Tailwater — client and post-processing toolkit for the Tailwater Wannier-Hamiltonian inference API.

Three workflow layers:

1. HTTP CLIENT — talk to the inference API
        from tailwater import tw_api_call
        paths = tw_api_call(structure, user, password, "./out", "mat", project=True)

2. SUBSPACE PROJECTION — fine-tune the output heads on supplier-provided
   embeddings to project predictions into a near-Fermi energy window
        from tailwater import subspace_projection
        subspace_projection(start_lr, end_lr, num_epochs, energy_range,
                            decay_sigma, device, save_path,
                            embed_path, graph_output_path)

3. POST-PROCESSING — load the HDF5 tight-binding model and run bulk DOS,
   surface spectral density, surface Greens-function (Lopez-Sancho),
   or Fermi-arc analyses
        from tailwater import tb_model, BulkDOS, SurfaceGreensFunction
        model  = tb_model.load("wannier90_hr.hdf5")
        result = SurfaceGreensFunction(model, ...).run()
        result.figure_top.savefig(...)

The package is self-contained — it does NOT require the proprietary
backbone weights or training code. Only the customer-shippable head
checkpoint (HeadsOnly.pth) and HDF5 / .pt artifacts produced by the
API are needed.
"""

__version__ = "0.4.1"

# ---- HTTP client + HDF5 loader ----
from .client import (
    tw_api_call,
    tb_model,
    remaining_credits,
)

# ---- Heads-only inference model ----
from .heads_only_model import (
    HeadsOnly,
    CovariantOnsiteHead,
    CovariantEdgeHead,
    load_heads_only_checkpoint,
    save_heads_only_checkpoint,
)

# ---- Subspace fine-tuning ----
from .finetune_heads import subspace_projection

# ---- Subspace loss helpers (advanced — used by subspace_projection internally) ----
from .subspace_utils import (
    Subspace_H_MSE_Loss,
    Subspace_EigLoss,
    Eigenvalue_Only_Loss,
    make_eigenvalue_only_data,
    build_subspace_active_mask,
    write_subspace_basis_file,
    SPATIAL_BASIS_LABELS,
    SPIN_BASIS_LABELS,
)

# ---- tbmodels assembly from raw head output ----
from .hr_export import (
    build_hr_model,
    build_hr_model_fast,
    write_hr_output,
)

# ---- Post-processing (KPM / Lopez-Sancho / Fermi-arc / bands) ----
from .wannier_wizard import (
    BulkDOS,
    SurfaceSpectralDensity,
    SurfaceGreensFunction,
    FermiArcMap,
    BandStructure,
    BulkDOSResult,
    SurfaceSpectralDensityResult,
    SurfaceGreensFunctionResult,
    FermiArcMapResult,
    BandStructureResult,
    generate_k_path,
    bulk_band_structure,
    compute_band_edges,
    align_to_vbm,
)

# ---- Constants (rarely needed directly; exposed for advanced users) ----
from .constants import NeighBrs, NUM_ELEMENTS


__all__ = [
    "__version__",
    # client
    "tw_api_call", "tb_model", "remaining_credits",
    # heads-only
    "HeadsOnly", "CovariantOnsiteHead", "CovariantEdgeHead",
    "load_heads_only_checkpoint", "save_heads_only_checkpoint",
    # subspace
    "subspace_projection",
    "Subspace_H_MSE_Loss", "Subspace_EigLoss", "Eigenvalue_Only_Loss",
    "make_eigenvalue_only_data", "build_subspace_active_mask",
    "write_subspace_basis_file",
    "SPATIAL_BASIS_LABELS", "SPIN_BASIS_LABELS",
    # tbmodels assembly
    "build_hr_model", "build_hr_model_fast", "write_hr_output",
    # post-processing
    "BulkDOS", "SurfaceSpectralDensity", "SurfaceGreensFunction",
    "FermiArcMap", "BandStructure",
    "BulkDOSResult", "SurfaceSpectralDensityResult",
    "SurfaceGreensFunctionResult", "FermiArcMapResult",
    "BandStructureResult",
    "generate_k_path", "bulk_band_structure",
    "compute_band_edges", "align_to_vbm",
    # constants
    "NeighBrs", "NUM_ELEMENTS",
]

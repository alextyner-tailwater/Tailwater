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

   The loaded model also carries three converters to other tight-binding
   libraries — ``model.to_pb()`` for pybinding (use with
   ``k_cart_from_frac`` for k conversion), ``model.to_pythtb()`` for
   PythTB (fractional k natively), and ``model.to_kwant()`` for Kwant
   (use ``2π·k_frac`` with the wraparound ``k_x/k_y/k_z`` params).

4. SPARSE HAMILTONIANS — when the optimized backend returns a sparse
   ``wannier90_hr.npz`` (large systems; O(N) egress), ``tw_api_call``
   auto-converts small systems (<30 atoms) to HDF5 and keeps large ones
   sparse. Load one with ``SparseHR.load("wannier90_hr.npz")``. The
   format-detecting converters ``to_pb`` / ``to_pythtb`` / ``to_kwant`` /
   ``to_hr_dat`` / ``to_hdf5`` / ``as_tbmodels`` accept EITHER a sparse
   ``.npz`` / ``SparseHR`` or a dense ``.hdf5`` / ``_hr.dat`` /
   ``tbmodels.Model`` and dispatch automatically, so the same call works
   regardless of format:
        from tailwater import to_hr_dat
        to_hr_dat("wannier90_hr.npz", "wannier90_hr.dat")

The package is self-contained — it does NOT require the proprietary
backbone weights or training code. Only the customer-shippable head
checkpoint (HeadsOnly.pth) and HDF5 / .pt artifacts produced by the
API are needed.
"""

__version__ = "0.9.0"

# ---- HTTP client + HDF5 loader ----
from .client import (
    tw_api_call,
    tb_model,
    remaining_credits,
    k_cart_from_frac,
)

# ---- WannierBerri helpers (optional, lazy-imports wannierberri) ----
from .wb_helpers import (
    wb_system_with_spin,
    spin_pairs_from_basis_json,
    spin_pairs_from_model_topology,
    build_ss_r0,
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

# ---- Multi-material fine-tuning against user-supplied Wannier targets ----
from .multi_finetune import (
    prepare_finetune_target,
    prepare_finetune_targets_from_directory,
    finetune_heads_multi,
    build_active_mask,
    build_edge_targets_from_hr,
    active_orbitals_from_win,
    infer_active_orbitals_from_hr,
    parse_win_projections,
    parse_win_atoms,
    parse_win_lattice,
    parse_win_fermi_energy,
    structure_from_win,
    SPATIAL_LABEL_TO_INDEX,
)

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

# ---- Sparse H(R) (optimized-inference .npz) + format-detecting converters ----
from .sparse import SparseHR
from .convert import (
    to_pb,
    to_pythtb,
    to_kwant,
    to_hr_dat,
    to_hdf5,
    as_tbmodels,
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

# ---- Surface charge-density heat maps (general (hkl) slab from any H(R)) ----
from .surface_charge import (
    surface_charge_density,
    load_hr,
    supercell_self_check,
)

# ---- Constants (rarely needed directly; exposed for advanced users) ----
from .constants import NeighBrs, NUM_ELEMENTS


__all__ = [
    "__version__",
    # client
    "tw_api_call", "tb_model", "remaining_credits", "k_cart_from_frac",
    # wannierberri helpers
    "wb_system_with_spin", "spin_pairs_from_basis_json",
    "spin_pairs_from_model_topology", "build_ss_r0",
    # heads-only
    "HeadsOnly", "CovariantOnsiteHead", "CovariantEdgeHead",
    "load_heads_only_checkpoint", "save_heads_only_checkpoint",
    # subspace
    "subspace_projection",
    # multi-material finetune
    "prepare_finetune_target", "prepare_finetune_targets_from_directory",
    "finetune_heads_multi",
    "build_active_mask", "build_edge_targets_from_hr",
    "active_orbitals_from_win", "infer_active_orbitals_from_hr",
    "parse_win_projections", "parse_win_atoms",
    "parse_win_lattice", "parse_win_fermi_energy", "structure_from_win",
    "SPATIAL_LABEL_TO_INDEX",
    "Subspace_H_MSE_Loss", "Subspace_EigLoss", "Eigenvalue_Only_Loss",
    "make_eigenvalue_only_data", "build_subspace_active_mask",
    "write_subspace_basis_file",
    "SPATIAL_BASIS_LABELS", "SPIN_BASIS_LABELS",
    # tbmodels assembly
    "build_hr_model", "build_hr_model_fast", "write_hr_output",
    # sparse H(R) + format-detecting converters
    "SparseHR",
    "to_pb", "to_pythtb", "to_kwant", "to_hr_dat", "to_hdf5", "as_tbmodels",
    # post-processing
    "BulkDOS", "SurfaceSpectralDensity", "SurfaceGreensFunction",
    "FermiArcMap", "BandStructure",
    "BulkDOSResult", "SurfaceSpectralDensityResult",
    "SurfaceGreensFunctionResult", "FermiArcMapResult",
    "BandStructureResult",
    "generate_k_path", "bulk_band_structure",
    "compute_band_edges", "align_to_vbm",
    # surface charge density
    "surface_charge_density", "load_hr", "supercell_self_check",
    # constants
    "NeighBrs", "NUM_ELEMENTS",
]

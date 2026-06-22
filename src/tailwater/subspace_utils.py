"""Subspace fine-tuning utilities.

Implements the "energy-window downfolding" workflow used by
finetune_subspace.py:

  1. For each material, identify which orbital channels at each atom have
     on-site energy inside the target window [E_LO, E_HI] (relative to
     the Fermi level convention E_F = 0 used elsewhere in the codebase).
  2. Build masks that zero out:
       - on-site matrix elements outside the active subspace, and
       - hopping matrix elements that connect at least one orbital
         outside the active subspace.
  3. Build a reduced orbital index mapping that lets the existing
     construct_Hk_vectorized routine produce a subspace H(k) instead of
     the full one.
  4. Provide a freeze_backbone helper so only the output heads stay
     trainable during fine-tuning.

Why use the TARGET on-site diagonal (not the predicted one) for the
energy criterion: the subspace must be a fixed reference that doesn't
co-move with the model's predictions. Using the prediction's diagonal
would let the model trivially escape the loss by predicting onsite
energies outside the window. The target's diagonal is what gives the
fine-tune procedure a stable, well-defined "where the physics lives"
definition.
"""

import math
from math import pi

import torch
import e3nn


# =====================================================
# CONSTANTS  (shared with train_wider.py / domain.py)
# =====================================================

import numpy as np

NeighBrs = np.array([
    [ 0,  0,  0], [ 0,  0,  1], [ 0,  1,  0], [ 1,  0,  0],
    [ 0,  1, -1], [ 0,  1,  1], [ 1, -1,  0], [ 1,  0, -1],
    [ 1,  0,  1], [ 1,  1,  0], [ 1, -1, -1], [ 1, -1,  1],
    [ 1,  1, -1], [ 1,  1,  1], [ 0,  0,  2], [ 0,  2,  0],
    [ 2,  0,  0],
])


# =====================================================
# 18-CHANNEL BASIS LABELS  (must match _build_18x18_basis in model_lite.py)
# =====================================================
# These labels describe what each row/column of the model's 18x18
# Hamiltonian block represents. The ordering convention is
#     orbital_index = spatial_index * 2 + spin_index
# i.e. spatial outer, spin inner. Spatial orbitals are in Wannier90
# canonical order (s, pz, px, py, dz2, dxz, dyz, dx2-y2, dxy) — this is
# the order produced by the permutation `[0, 2, 3, 1, 6, 7, 5, 8, 4]`
# applied to the natural e3nn (py, pz, px) and (dxy, dyz, dz2, dxz,
# dx2-y2) orderings inside _build_18x18_basis.
SPATIAL_BASIS_LABELS = [
    "s", "pz", "px", "py",
    "dz2", "dxz", "dyz", "dx2-y2", "dxy",
]
SPIN_BASIS_LABELS = ["up", "down"]


def orbital_index_to_labels(orbital_index: int):
    """Map an integer in 0..17 to its (spatial, spin) labels.

    Convention: orbital_index = spatial_index * 2 + spin_index.
    Returns ("s"|"pz"|..., "up"|"down").
    """
    if not (0 <= orbital_index < 18):
        raise ValueError(f"orbital_index must be in [0, 18), got {orbital_index}")
    spatial = orbital_index // 2
    spin    = orbital_index % 2
    return SPATIAL_BASIS_LABELS[spatial], SPIN_BASIS_LABELS[spin]


def write_subspace_basis_file(
    out_path,
    active_mask,
    atoms,
    LM,
    energy_window=None,
    onsite_energies=None,
    extra_metadata=None,
):
    """Write a JSON file describing the basis of the subspace Hamiltonian.

    Each row/column of the subspace H corresponds to one
    (atom_index, orbital_index_in_18_basis) pair from the full per-atom
    basis. This file is the authoritative mapping from subspace integer
    indices to physical labels — necessary because the _hr.dat / HDF5
    written by tbmodels carries only integer indices.

    Args
    ----
    out_path        : path to write the JSON file (extension `.basis.json`
                      recommended; not enforced).
    active_mask     : BoolTensor or ndarray of shape [num_atoms, 18].
                      True for slots that survived the subspace mask.
                      The SAME mask used to build the subspace H — order
                      matters because subspace_idx is assigned by a
                      row-major iteration over (atom, orbital_in_18).
    atoms           : list of (symbol, [x, y, z]) — the per-atom species
                      and Cartesian positions, as produced by
                      structure_io.process_win.
    LM              : 3x3 lattice matrix (NumPy array or list-of-lists).
    energy_window   : optional (e_lo, e_hi) tuple. Echoed into the file
                      for traceability.
    onsite_energies : optional [num_atoms, 18] real array of on-site
                      diagonal energies, OR [num_sub] array of energies
                      already in subspace order. If None, the field is
                      omitted from the file. Useful for downstream
                      band-window sanity checks.
    extra_metadata  : optional dict — anything else worth recording
                      (model version, checkpoint path, decay sigma, etc.).

    Returns
    -------
    out_path : the path that was written (echoed for convenience).
    """
    import json

    if isinstance(active_mask, torch.Tensor):
        active_mask = active_mask.detach().cpu().numpy()
    active_mask = np.asarray(active_mask, dtype=bool)
    if active_mask.shape[1] != 18:
        raise ValueError(
            f"active_mask must have shape [num_atoms, 18], got {active_mask.shape}"
        )

    num_atoms = active_mask.shape[0]
    num_sub   = int(active_mask.sum())

    # Normalize onsite_energies into a [num_atoms, 18] grid for lookup,
    # so callers can pass either layout without us caring.
    energies_per_slot = None
    if onsite_energies is not None:
        oe = np.asarray(onsite_energies)
        if oe.shape == (num_atoms, 18):
            energies_per_slot = oe
        elif oe.shape == (num_sub,):
            # Caller already collapsed to subspace order — re-distribute
            # back into the [num_atoms, 18] grid for uniform lookup.
            grid = np.full((num_atoms, 18), np.nan)
            sub_cnt = 0
            for a in range(num_atoms):
                for o in range(18):
                    if active_mask[a, o]:
                        grid[a, o] = oe[sub_cnt]
                        sub_cnt += 1
            energies_per_slot = grid
        else:
            raise ValueError(
                f"onsite_energies must be shape [num_atoms, 18] or "
                f"[num_sub], got {oe.shape}"
            )

    # ---- Iterate in the SAME order used by build_subspace_orbital_mapping
    # and build_subspace_tb_model. Row-major over (atom_idx, orbital_idx);
    # subspace_idx increments only when active_mask is True.
    atoms_block = []
    sub_idx = 0
    for atom_idx in range(num_atoms):
        symbol, position = atoms[atom_idx]
        active_orbitals = []
        for orbital_idx in range(18):
            if not active_mask[atom_idx, orbital_idx]:
                continue
            spatial_label, spin_label = orbital_index_to_labels(orbital_idx)
            entry = {
                "subspace_index":  sub_idx,
                "orbital_index":   orbital_idx,        # within the per-atom 18 basis
                "spatial":         spatial_label,
                "spin":            spin_label,
            }
            if energies_per_slot is not None:
                e_val = energies_per_slot[atom_idx, orbital_idx]
                if not np.isnan(e_val):
                    entry["onsite_energy_eV"] = float(e_val)
            active_orbitals.append(entry)
            sub_idx += 1

        if active_orbitals:
            atoms_block.append({
                "atom_index":      atom_idx,
                "element":         symbol,
                "position_cart":   [float(x) for x in position],
                "active_orbitals": active_orbitals,
            })

    # ---- Build the top-level document ----
    document = {
        "format_version":   "1.0",
        "schema":           "tailwater-subspace-basis",
        "per_atom_basis_size": 18,
        "spatial_basis":    SPATIAL_BASIS_LABELS,
        "spin_basis":       SPIN_BASIS_LABELS,
        "ordering_convention":
            "orbital_index = spatial_index * 2 + spin_index; "
            "subspace_index iterates row-major over (atom_index, orbital_index) "
            "and increments only when active_mask is True.",
        "subspace_size":    num_sub,
        "num_atoms":        num_atoms,
        "lattice_vectors_angstrom":
            np.asarray(LM).tolist() if LM is not None else None,
        "atoms":            atoms_block,
    }
    if energy_window is not None:
        document["energy_window_eV"] = [float(energy_window[0]),
                                        float(energy_window[1])]
    if extra_metadata is not None:
        document["extra_metadata"] = extra_metadata

    with open(out_path, "w") as f:
        json.dump(document, f, indent=2)
    return out_path


# =====================================================
# SUBSPACE MASK CONSTRUCTION
# =====================================================

def build_subspace_active_mask(node_features, onsite_target, e_lo, e_hi):
    """Return the per-(atom, orbital) boolean mask of the active subspace.

    An orbital channel is "active" iff
        (a) it is structurally present (node_features[:, 109:127] == 1), and
        (b) its TARGET on-site diagonal energy is in [e_lo, e_hi].

    The structural part (a) is the same convention used throughout the
    codebase (orbital-presence one-hot in cols 109..126 of node_features).
    The energy part (b) reads the diagonal of the per-atom 18x18 onsite
    block from the supplied target tensor; the diagonal is real by
    construction (on-site energies are real), so we look at only the
    real component.

    Args
    ----
    node_features : [num_atoms, 127]  (cols 109..126 are the orbital flags)
    onsite_target : [num_atoms, 18, 18, 2]  (per-atom on-site block, real+imag)
    e_lo, e_hi    : floats — energy window endpoints, same units as
                    the target Hamiltonian (eV in this codebase).

    Returns
    -------
    active : BoolTensor [num_atoms, 18]
    """
    structural_active = (node_features[:, 109:127] == 1)              # [N, 18]

    # Real part of the diagonal of each atom's 18x18 onsite block.
    # onsite_target is laid out as [..., real, imag] in the last dim.
    diag_E = torch.diagonal(onsite_target[..., 0], dim1=-2, dim2=-1)  # [N, 18]

    energy_active = (diag_E >= e_lo) & (diag_E <= e_hi)               # [N, 18]
    return structural_active & energy_active


def build_subspace_orbital_mapping(node_features, onsite_target, e_lo, e_hi):
    """Maps (atom, orbital) -> SUBSPACE global matrix index.

    Drop-in analog of build_orbital_mapping in train_wider.py, but
    inactive orbitals (either structurally OR by the energy window) get
    -1 and are excluded from the subspace H(k).

    Returns
    -------
    mapping  : LongTensor [num_atoms, 18] — subspace index (or -1)
    num_sub  : int — number of active orbitals in the subspace
    active   : BoolTensor [num_atoms, 18] — the underlying mask
    """
    active = build_subspace_active_mask(node_features, onsite_target,
                                        e_lo, e_hi)
    num_sub = int(active.sum().item())

    mapping = torch.full((node_features.shape[0], 18), -1,
                         dtype=torch.long, device=node_features.device)
    if num_sub > 0:
        mapping[active] = torch.arange(num_sub, device=node_features.device)
    return mapping, num_sub, active


def build_onsite_edge_subspace_masks(gdata, e_lo, e_hi):
    """Build per-element BoolTensor masks for the H MSE loss on the subspace.

    Returns
    -------
    M_onsite : FloatTensor [num_atoms, 18, 18]  — 1 where both orbital
               indices of the on-site block are inside the subspace,
               0 elsewhere.
    M_edge   : FloatTensor [num_edges, 18, 18]  — 1 where the src orbital
               AND the dst orbital are both inside the subspace, 0
               otherwise. (Note: an edge between two atoms that BOTH have
               active orbitals is only contributing within the active
               sub-blocks of its 18x18; this is exactly what "couples
               these onsite terms" means in the subspace fit.)
    active   : BoolTensor [num_atoms, 18]
    """
    target = gdata.edge_targets[:, 0, :, :, :]                # [num_edges, 18, 18, 2]
    is_self_loop = (gdata.edge_vectors.norm(dim=-1) == 0)
    onsite_target = target[is_self_loop]                       # [num_atoms, 18, 18, 2]

    active = build_subspace_active_mask(gdata.node_features,
                                        onsite_target, e_lo, e_hi)  # [N, 18]

    # Onsite mask: 1 wherever both row and column are inside the subspace.
    M_onsite = (active.unsqueeze(2) & active.unsqueeze(1)).float()   # [N, 18, 18]

    # Edge mask: per-edge outer product of the src active mask and the dst
    # active mask. Self-loop edges naturally reduce to the onsite mask.
    edge_src = gdata.edge_index[0]
    edge_dst = gdata.edge_index[1]
    active_src = active[edge_src]                              # [E, 18]
    active_dst = active[edge_dst]                              # [E, 18]
    M_edge = (active_src.unsqueeze(2) & active_dst.unsqueeze(1)).float()  # [E, 18, 18]

    return M_onsite, M_edge, active


# =====================================================
# SUBSPACE H(k) CONSTRUCTION  (mirrors train_wider.py)
# =====================================================

def construct_Hk_vectorized_subspace(edge_cplx, inv_data, mapping, num_sub,
                                     kvec, NeighBrs_t, device, threshold=0.0):
    """Same vectorized H(k) builder as train_wider.py, parametrized by a
    subspace `mapping` (with -1 for inactive orbitals) and `num_sub`.

    Inactive rows/cols are filtered out by the existing
    (global_row != -1) & (global_col != -1) check; the only thing this
    routine cares about is that num_sub matches the count of non-(-1)
    entries in `mapping`.
    """
    num_k     = kvec.shape[0]
    num_edges = edge_cplx.shape[0]

    Hk = torch.zeros((num_k, num_sub, num_sub),
                     dtype=torch.complex64, device=device)

    atm1_flat = inv_data[:, 0].view(-1, 1, 1).expand(-1, 18, 18).flatten()
    atm2_flat = inv_data[:, 1].view(-1, 1, 1).expand(-1, 18, 18).flatten()
    k_flat    = inv_data[:, 2].view(-1, 1, 1).expand(-1, 18, 18).flatten()

    local_row = torch.arange(18, device=device).view(1, 18, 1).expand(num_edges, 18, 18).flatten()
    local_col = torch.arange(18, device=device).view(1, 1, 18).expand(num_edges, 18, 18).flatten()

    vals = edge_cplx.flatten()
    global_row = mapping[atm1_flat, local_row]
    global_col = mapping[atm2_flat, local_col]
    R_vecs     = NeighBrs_t[k_flat]

    valid_mask = (global_row != -1) & (global_col != -1)
    if threshold > 0.0:
        valid_mask &= (torch.abs(vals) > threshold)

    is_R_zero    = (k_flat == 0)
    is_diagonal  = (global_row == global_col)
    mask_not_diag  = ~(is_R_zero & is_diagonal)
    mask_R0_unique = (~is_R_zero) | (is_R_zero & (global_row < global_col))
    final_mask = valid_mask & mask_not_diag & mask_R0_unique

    g_row = global_row[final_mask]
    g_col = global_col[final_mask]
    R     = R_vecs[final_mask]
    V     = vals[final_mask]

    phase_angles = -2 * pi * torch.matmul(kvec, R.T.to(kvec.dtype))
    H_vals       = torch.exp(1j * phase_angles) * V.unsqueeze(0)

    Hk_flat       = Hk.view(num_k, -1)
    idx_forward   = (g_row * num_sub + g_col).unsqueeze(0).expand(num_k, -1)
    idx_backward  = (g_col * num_sub + g_row).unsqueeze(0).expand(num_k, -1)
    Hk_flat.scatter_add_(1, idx_forward,  H_vals)
    Hk_flat.scatter_add_(1, idx_backward, H_vals.conj())

    diag_mask = is_R_zero & is_diagonal & valid_mask
    g_diag    = global_row[diag_mask]
    V_diag    = vals[diag_mask].real.to(torch.complex64)
    idx_diag  = (g_diag * num_sub + g_diag).unsqueeze(0).expand(num_k, -1)
    Hk_flat.scatter_add_(1, idx_diag, V_diag.unsqueeze(0).expand(num_k, -1))

    return Hk


# =====================================================
# SUBSPACE LOSSES
# =====================================================

def Subspace_H_MSE_Loss(gdata, edge_pred, onsite_pred, e_lo, e_hi,
                        return_components=False):
    """Per-element H MSE restricted to the [e_lo, e_hi] subspace.

    Mathematically identical to the H MSE term in train_wider.py, but
    the per-element error is weighted by the subspace masks before being
    summed and normalized by the number of contributing (real + imag)
    elements.  Predictions outside the active subspace produce zero
    gradient, exactly matching the "freeze + mask" intent.
    """
    edge_pred   = edge_pred.reshape(gdata.edge_index.shape[1], 18, 18, 2)
    onsite_pred = onsite_pred.reshape(gdata.node_features.shape[0], 18, 18, 2)

    target       = gdata.edge_targets[:, 0, :, :, :]
    is_self_loop = (gdata.edge_vectors.norm(dim=-1) == 0)
    weights      = torch.sign(torch.abs(gdata.edge_vectors.norm(dim=1))).view(-1, 1, 1, 1)

    M_onsite, M_edge, _ = build_onsite_edge_subspace_masks(gdata, e_lo, e_hi)

    # Edge MSE (off-diagonal hoppings). The pre-existing `weights`
    # multiplier zeros out the self-loop slots in `edge_pred` before MSE,
    # mirroring train_wider.py.
    edge_err = (weights * (edge_pred - target)) * M_edge.unsqueeze(-1)     # [E, 18, 18, 2]
    n_edge   = (M_edge.sum() * 2).clamp(min=1.0)                            # *2 for real+imag
    mse_edge = (edge_err ** 2).sum() / n_edge

    # Onsite MSE.
    onsite_err = (onsite_pred - target[is_self_loop]) * M_onsite.unsqueeze(-1)
    n_onsite   = (M_onsite.sum() * 2).clamp(min=1.0)
    mse_onsite = (onsite_err ** 2).sum() / n_onsite

    loss = mse_onsite + 10.0 * mse_edge      # same edge/onsite weighting as train_wider.py
    if return_components:
        return loss, mse_onsite, mse_edge
    return loss


def make_eigenvalue_only_data(gdata, kvecs, eigs_per_k, e_lo, e_hi):
    """Attach customer-supplied eigenvalues + custom k-points to a Data object.

    Use this on the customer side to format their DFT eigenvalues into
    the shape that Eigenvalue_Only_Loss expects.

    Args
    ----
    gdata      : PyG Data with backbone embeddings already attached
                 (f_out, edge_feat) and the usual structural fields
                 (node_features, edge_index, edge_vectors, inv_data).
    kvecs      : array-like [num_k, 3] — fractional k-coordinates, in the
                 same convention used by construct_Hk_vectorized (i.e.
                 phase = exp(-2πi k·R)).
    eigs_per_k : sequence of length num_k. Each entry is a 1D iterable
                 of eigenvalues at the corresponding k-point (in eV,
                 relative to E_F = 0). Eigenvalues outside [e_lo, e_hi]
                 are discarded here so the loss never sees them.
    e_lo, e_hi : energy window (eV).

    Mutates `gdata` to add:
        .kpts             : [num_k, 3] float32
        .target_eigs      : [num_k, max_n] float32, -1e9 padding
        .target_eigs_mask : [num_k, max_n] bool, True where valid
    The padding sentinel is well outside any physical band energy, so
    if a downstream consumer ignores the mask the loss will at worst
    report an obvious garbage value rather than silently corrupt training.

    Notes on the count interpretation
    ---------------------------------
    The number of valid eigenvalues at each k is what defines the
    "subspace size" for that k in the eigenvalue-only loss — the loss
    will use exactly that many predicted bands per k, selected as the
    ones closest to the window center.
    """
    num_k = len(eigs_per_k)
    filtered = []
    for eigs_k in eigs_per_k:
        eigs_k = torch.as_tensor(eigs_k, dtype=torch.float32).flatten()
        in_window = (eigs_k >= e_lo) & (eigs_k <= e_hi)
        e_in = eigs_k[in_window]
        # Sort ascending; eigvalsh produces sorted output and we'll want
        # the index-by-index match against it.
        e_in, _ = torch.sort(e_in)
        filtered.append(e_in)

    max_n = max((f.shape[0] for f in filtered), default=0)
    target_eigs = torch.full((num_k, max(1, max_n)), -1e9, dtype=torch.float32)
    target_mask = torch.zeros((num_k, max(1, max_n)),  dtype=torch.bool)
    for i, f in enumerate(filtered):
        n = f.shape[0]
        if n > 0:
            target_eigs[i, :n] = f
            target_mask[i, :n] = True

    gdata.kpts             = torch.as_tensor(kvecs, dtype=torch.float32)
    gdata.target_eigs      = target_eigs
    gdata.target_eigs_mask = target_mask
    return gdata


def Eigenvalue_Only_Loss(gdata, edge_pred, onsite_pred, e_lo, e_hi,
                         decay_sigma=None, neighbrs_arr=None,
                         return_diagnostic=False, weight_center=None):
    """Eigenvalue-only fine-tuning loss — no target Hamiltonian required.

    The customer provides, per material:
        gdata.kpts              [num_k, 3]            custom k-coordinates
        gdata.target_eigs       [num_k, max_n]        padded target eigenvalues
        gdata.target_eigs_mask  [num_k, max_n] bool   validity mask
    (Build these with make_eigenvalue_only_data().)

    Procedure
    ---------
    For each k-point in `kpts`:
      1. Build the FULL predicted H(k) using every structurally-active
         orbital (no orbital mask is required because the customer's
         eigenvalues directly define what "in the subspace" means at
         this k).
      2. Diagonalize to get num_wann predicted eigenvalues at this k.
      3. Read n_target_k = mask[k].sum() target eigenvalues, already
         sorted ascending and in-window.
      4. Pick the n_target_k predicted eigenvalues whose energies are
         CLOSEST to the window center — this is the "effective subspace
         at k" implied by the customer's data, with size exactly equal
         to the number of provided target eigenvalues.
      5. Sort those predicted selections ascending and pair them with
         the target eigenvalues index-by-index.
      6. Weight each pair by a Gaussian centered at the window midpoint
         and reduce: loss = sum(w * |Δeig|) / sum(w).

    Selecting "closest-to-center" rather than "lowest" is the right
    choice for windows that straddle E_F = 0: it correctly picks bands
    from both sides of the gap proportionally to how close they sit to
    the window's central energy.

    Returns
    -------
    loss : scalar tensor.
    (Optionally: (loss, num_wann, n_target_total, n_match_total) when
    return_diagnostic=True — n_match_total is how many predicted bands
    actually got paired with target bands across the full k-set, useful
    to confirm n_target_k <= num_wann at every k.)
    """
    device = edge_pred.device
    center = 0.5 * (e_lo + e_hi)
    # `wcenter` controls ONLY the Gaussian weighting focus; the band
    # *selection* below still uses the window midpoint `center` so the
    # matched band set tracks the window. Pin wcenter on the Fermi level
    # (or VBM) to emphasize near-Fermi bands inside a wide clean window.
    wcenter = center if weight_center is None else float(weight_center)
    sigma  = decay_sigma if decay_sigma is not None else (e_hi - e_lo) / 4.0
    inv_two_sigma2 = 1.0 / (2.0 * sigma * sigma)

    # ---- Customer-supplied k-points + target eigenvalues ----
    if not (hasattr(gdata, "kpts") and hasattr(gdata, "target_eigs")
            and hasattr(gdata, "target_eigs_mask")):
        raise RuntimeError(
            "Eigenvalue_Only_Loss requires gdata.kpts, gdata.target_eigs, "
            "gdata.target_eigs_mask. Call make_eigenvalue_only_data(...) "
            "to attach them to your Data objects before training."
        )
    kvec        = gdata.kpts.to(device).float()
    target_eigs = gdata.target_eigs.to(device).float()
    target_mask = gdata.target_eigs_mask.to(device).bool()

    if neighbrs_arr is None:
        neighbrs_arr = NeighBrs
    NeighBrs_t = torch.tensor(neighbrs_arr, dtype=torch.float32, device=device) \
        if not torch.is_tensor(neighbrs_arr) else neighbrs_arr.to(device).float()

    # ---- Build the FULL predicted edge tensor (same as the other losses) ----
    edge_pred_r   = edge_pred.reshape(gdata.edge_index.shape[1], 18, 18, 2)
    is_self_loop  = (gdata.edge_vectors.norm(dim=-1) == 0)
    weights_h     = torch.sign(torch.abs(gdata.edge_vectors.norm(dim=1))).view(-1, 1, 1, 1)
    edge_pred_full = weights_h * edge_pred_r
    edge_pred_full[is_self_loop] = onsite_pred
    pred_cplx = edge_pred_full[..., 0] + 1j * edge_pred_full[..., 1]

    # ---- Full orbital mapping (no energy filter) ----
    full_mapping, num_wann = _build_full_orbital_mapping(gdata.node_features)
    if num_wann < 2:
        zero = torch.tensor(0.0, device=device, requires_grad=True)
        if return_diagnostic:
            return zero, num_wann, 0, 0
        return zero

    # ---- Predicted H(k) at the customer's custom k-points ----
    Hk_pred_full = construct_Hk_vectorized_subspace(
        pred_cplx, gdata.inv_data, full_mapping, num_wann,
        kvec, NeighBrs_t, device, threshold=0.01,
    )
    eig_pred_full = torch.linalg.eigvalsh(Hk_pred_full)        # [num_k, num_wann]

    # ---- Per-k matching ----
    num_k = eig_pred_full.shape[0]
    total_weighted_err = torch.tensor(0.0, device=device)
    total_weight       = torch.tensor(0.0, device=device)
    n_target_total = 0
    n_match_total  = 0

    for k_idx in range(num_k):
        valid = target_mask[k_idx]
        n_target_k = int(valid.sum().item())
        if n_target_k == 0:
            continue
        n_target_total += n_target_k

        E_t_k = target_eigs[k_idx][valid]   # already sorted ascending by make_eigenvalue_only_data

        # The n_target_k predicted eigenvalues closest to the window center
        # define the effective subspace at this k. torch.topk(..., largest=False)
        # gives the smallest-distance entries — i.e., the closest values.
        E_p_full_k = eig_pred_full[k_idx]
        dist       = torch.abs(E_p_full_k - center)
        n_take     = min(n_target_k, num_wann)
        _, idx_closest = torch.topk(dist, n_take, largest=False)
        E_p_sub_k, _   = torch.sort(E_p_full_k[idx_closest])
        n_match_total += n_take

        n_pairs   = min(E_t_k.shape[0], E_p_sub_k.shape[0])
        E_t_paired = E_t_k[:n_pairs]
        E_p_paired = E_p_sub_k[:n_pairs]

        w_k = torch.exp(-((E_t_paired - wcenter) ** 2) * inv_two_sigma2)
        total_weighted_err = total_weighted_err + (w_k * torch.abs(E_p_paired - E_t_paired)).sum()
        total_weight       = total_weight + w_k.sum()

    if total_weight.item() == 0.0:
        zero = torch.tensor(0.0, device=device, requires_grad=True)
        if return_diagnostic:
            return zero, num_wann, n_target_total, n_match_total
        return zero

    loss = total_weighted_err / total_weight
    if return_diagnostic:
        return loss, num_wann, n_target_total, n_match_total
    return loss


def _build_full_orbital_mapping(node_features):
    """Maps (atom, orbital) -> FULL global matrix index. Structurally inactive
    orbitals (node_features[:, 109:127] == 0) get -1. No energy filter is
    applied — this is the mapping for the full-system target H(k).
    """
    active_mask = (node_features[:, 109:127] == 1)
    num_wann = int(active_mask.sum().item())
    mapping = torch.full((node_features.shape[0], 18), -1,
                         dtype=torch.long, device=node_features.device)
    if num_wann > 0:
        mapping[active_mask] = torch.arange(num_wann,
                                            device=node_features.device)
    return mapping, num_wann


def Subspace_EigLoss(gdata, edge_pred, onsite_pred, kvec, NeighBrs_arr,
                     e_lo, e_hi, decay_sigma=None,
                     return_evals=False, return_diagnostic=False,
                     symmetrize_targets: bool = False):
    """Eigenvalue loss with FULL-target-band reference, downfolded prediction.

    Physics
    -------
    The target eigenvalues come from diagonalizing the FULL target H(k)
    (only structurally inactive orbitals removed — no energy filter), then
    keeping only those that lie inside the energy window [e_lo, e_hi].
    The predicted eigenvalues come from diagonalizing the SUBSPACE predicted
    H(k), which is the model's prediction after the orbital mask is
    applied — i.e., the downfolded low-energy effective model.

    Pairs are matched index-by-index in sorted-ascending order, truncated
    to N_k = min(n_target_in_window(k), num_sub) at each k-point. This
    answers the "make the number of predicted bands match the number of
    target bands in the window" requirement: any predicted band beyond
    N_k is ignored at that k-point, and any target band beyond N_k is
    treated as "not represented by the chosen subspace" (reported in the
    diagnostic).

    Weighting
    ---------
    Each (target, predicted) pair is weighted by a Gaussian centered at
    the middle of the window:
        w(E_t) = exp( -(E_t - center)^2 / (2 * sigma^2) )
    where center = (e_lo + e_hi) / 2 and sigma defaults to (e_hi - e_lo)/4
    (so the window covers ~±2 sigma → ~95% of the Gaussian mass). Tune
    sigma via `decay_sigma` to make the weight fall off faster (sharper
    focus near E_F) or slower (more uniform across the window).

    Returns
    -------
    loss : scalar tensor — weighted mean absolute error across all k-points.
    (Optionally: eig_pred / eig_target, or (num_sub, n_target_total,
    n_match_total) diagnostic about how well the subspace size matches
    the band count in the window.)
    """
    device = edge_pred.device
    center = 0.5 * (e_lo + e_hi)
    sigma  = decay_sigma if decay_sigma is not None else (e_hi - e_lo) / 4.0
    inv_two_sigma2 = 1.0 / (2.0 * sigma * sigma)

    kvec_t = torch.tensor(kvec, dtype=torch.float32, device=device) \
        if not torch.is_tensor(kvec) else kvec.to(device).float()
    NeighBrs_t = torch.tensor(NeighBrs_arr, dtype=torch.float32, device=device) \
        if not torch.is_tensor(NeighBrs_arr) else NeighBrs_arr.to(device).float()

    # ---- Build the FULL predicted edge tensor (with onsite at self-loops) ----
    # This is the same prediction the model produced — we don't re-derive it;
    # we just route it through TWO different orbital mappings to get the two
    # H(k) we need (full target H(k) for the band reference, subspace pred H(k)
    # for the bands we're learning to fit).
    edge_pred_r   = edge_pred.reshape(gdata.edge_index.shape[1], 18, 18, 2)
    is_self_loop  = (gdata.edge_vectors.norm(dim=-1) == 0)
    weights_h     = torch.sign(torch.abs(gdata.edge_vectors.norm(dim=1))).view(-1, 1, 1, 1)
    edge_pred_full = weights_h * edge_pred_r
    edge_pred_full[is_self_loop] = onsite_pred
    pred_cplx = edge_pred_full[..., 0] + 1j * edge_pred_full[..., 1]

    # ---- Target ----
    target      = gdata.edge_targets[:, 0, :, :, :]
    target_cplx = target[..., 0] + 1j * target[..., 1]
    onsite_target = target[is_self_loop]

    # ---- Mappings ----
    # FULL mapping (only structurally inactive orbitals are -1) — for target H(k).
    full_mapping, num_wann = _build_full_orbital_mapping(gdata.node_features)
    # SUBSPACE mapping (structural + energy-window filter) — for predicted H(k).
    sub_mapping, num_sub, _ = build_subspace_orbital_mapping(
        gdata.node_features, onsite_target, e_lo, e_hi,
    )

    # Degenerate cases: too few active orbitals to produce eigenvalues.
    if num_sub < 2 or num_wann < 2:
        zero = torch.tensor(0.0, device=device, requires_grad=True)
        if return_evals:
            return zero, None, None
        if return_diagnostic:
            return zero, num_sub, 0, 0
        return zero

    # ---- Build H(k) ----
    # Target H(k) uses the FULL (non-downfolded) mapping -> bands from the
    # full system. Predicted H(k) uses the SUBSPACE mapping -> bands from
    # the downfolded effective model the user is fine-tuning toward.
    Hk_target_full = construct_Hk_vectorized_subspace(
        target_cplx, gdata.inv_data, full_mapping, num_wann,
        kvec_t, NeighBrs_t, device, threshold=0.01,
    )
    Hk_pred_sub = construct_Hk_vectorized_subspace(
        pred_cplx, gdata.inv_data, sub_mapping, num_sub,
        kvec_t, NeighBrs_t, device, threshold=0.01,
    )

    eig_target_full = torch.linalg.eigvalsh(Hk_target_full)   # [num_k, num_wann]
    eig_pred_sub    = torch.linalg.eigvalsh(Hk_pred_sub)      # [num_k, num_sub]

    # ---- Optionally Kramers-pair the target eigenvalues ----
    # When the crystal has PT or C₂ᶻT symmetry, every band must be
    # Kramers-doubly-degenerate, and the raw target predictions usually
    # break that by tens to hundreds of meV (model noise). Pair-averaging
    # consecutive ascending eigenvalues here replaces the targets with
    # their unique minimum-perturbation Kramers-paired version — the
    # eigenvalue-level equivalent of `light_symmetrize.kramers_fix`.
    # Caller MUST gate this on symmetry detection (the spectral fix is
    # wrong for non-PT / non-C2zT crystals — Rashba/Weyl splittings are
    # physical there and pair-averaging would corrupt the target).
    if symmetrize_targets:
        # num_wann is even for spinful models; if it's odd (no-spin or
        # incomplete pairing), fall through unchanged.
        n_bands = eig_target_full.shape[-1]
        if n_bands % 2 == 0:
            pair_avg = 0.5 * (eig_target_full[..., 0::2]
                              + eig_target_full[..., 1::2])
            eig_target_full = torch.stack(
                [pair_avg, pair_avg], dim=-1,
            ).reshape(*pair_avg.shape[:-1], 2 * pair_avg.shape[-1])

    # ---- Per-k matching ----
    # At each k, filter the FULL target eigenvalues to [e_lo, e_hi], then
    # pair the first N_k of them with the first N_k subspace eigenvalues,
    # where N_k = min(n_target_in_window(k), num_sub). eigvalsh returns
    # ascending order already; the in-window subset of an ascending list
    # is itself ascending.
    num_k     = eig_target_full.shape[0]
    in_window = (eig_target_full >= e_lo) & (eig_target_full <= e_hi)   # [num_k, num_wann]

    total_weighted_err = torch.tensor(0.0, device=device)
    total_weight       = torch.tensor(0.0, device=device)
    n_target_total = 0
    n_match_total  = 0

    for k_idx in range(num_k):
        mask_k = in_window[k_idx]
        E_t_window_k = eig_target_full[k_idx][mask_k]    # [n_target_k], variable per k
        n_target_k = int(E_t_window_k.shape[0])
        n_target_total += n_target_k

        n_match_k = min(n_target_k, num_sub)
        if n_match_k == 0:
            continue
        n_match_total += n_match_k

        E_t_k = E_t_window_k[:n_match_k]
        E_p_k = eig_pred_sub[k_idx][:n_match_k]

        # Exponential weighting centered at window midpoint, falling off
        # toward the edges. Weight uses the TARGET energy (fixed reference),
        # not the prediction.
        w_k = torch.exp(-((E_t_k - center) ** 2) * inv_two_sigma2)

        total_weighted_err = total_weighted_err + (w_k * torch.abs(E_p_k - E_t_k)).sum()
        total_weight       = total_weight + w_k.sum()

    if total_weight.item() == 0.0:
        zero = torch.tensor(0.0, device=device, requires_grad=True)
        if return_evals:
            return zero, eig_pred_sub, eig_target_full
        if return_diagnostic:
            return zero, num_sub, n_target_total, n_match_total
        return zero

    loss = total_weighted_err / total_weight

    if return_evals:
        return loss, eig_pred_sub, eig_target_full
    if return_diagnostic:
        return loss, num_sub, n_target_total, n_match_total
    return loss


# =====================================================
# BACKBONE FREEZE
# =====================================================

def freeze_backbone(model, head_names=("CovariantOnsiteHead",
                                       "CovariantEdgeHead")):
    """Freeze every parameter NOT in one of the output heads.

    Also forces every e3nn BatchNorm in the backbone into eval mode so its
    running stats stop updating during fine-tuning. This is critical for
    DDP — if a frozen module's BN running stats drift on different ranks,
    you'll see exactly the eigenvalue-divergence symptom you debugged
    earlier in train_wider.

    The output heads (which contain Linear -> Gate -> Linear and no BN)
    stay in train() mode and remain trainable.

    Returns the number of trainable parameters that survived the freeze.
    """
    base = model.module if hasattr(model, "module") else model

    n_trainable = 0
    for name, param in base.named_parameters():
        if any(name.startswith(h + ".") for h in head_names):
            param.requires_grad = True
            n_trainable += param.numel()
        else:
            param.requires_grad = False

    # Set every BatchNorm in the backbone to eval mode. The heads have no
    # BN, so this is safe to apply across all BN-like modules outside the
    # named heads.
    for module_name, module in base.named_modules():
        if isinstance(module, e3nn.nn.BatchNorm) or isinstance(
                module, (torch.nn.BatchNorm1d,
                         torch.nn.BatchNorm2d,
                         torch.nn.BatchNorm3d)):
            in_head = any(module_name.startswith(h + ".") for h in head_names)
            if not in_head:
                module.eval()

    return n_trainable

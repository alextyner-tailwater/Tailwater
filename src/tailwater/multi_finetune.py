"""Multi-material heads-only fine-tuning against user-supplied Wannier targets.

``subspace_projection`` (in ``finetune_heads.py``) fine-tunes the
heads on a single material using the *API's own* full-model output as
a self-distillation target. This module covers the complementary
workflow: fine-tune the heads on a **set** of materials where the
ground-truth Hamiltonians come from the **user's own** Wannier
calculations.

Workflow
--------

For each material in the user's set:

1. The user calls the API once to get the backbone embedding
   (``embed_path``, the ``.pt`` produced by
   ``/upload_structure_and_download_embeddings/``).
2. The user has their own Wannier Hamiltonian — an ``_hr.dat`` or
   ``_hr.hdf5`` file — and knows which spatial orbitals were
   projected per atom (e.g. ``[["s", "pz", "px", "py"]]`` for an
   atom with only s+p orbitals, vs. the full
   ``[["s", "pz", "px", "py", "dz2", "dxz", "dyz", "dx2-y2", "dxy"]]``
   for s+p+d).
3. Call :func:`prepare_finetune_target` once per material to merge
   the API embedding with the user-hr-derived targets into a single
   PyG ``Data``-with-targets object (cached as a ``.pt`` for
   reuse).
4. Call :func:`finetune_heads_multi` on the list of prepared targets
   (plus an optional held-out validation set) to fine-tune the heads.
   Loss is masked outside the user-specified energy window per
   material, exactly like ``subspace_projection``.

The user's hr-models need NOT have the full 18 orbitals per atom —
each atom can carry an arbitrary subset of
``{s, pz, px, py, dz2, dxz, dyz, dx2-y2, dxy}``. The active mask
(stored in ``gdata.node_features[:, 109:127]``) tells the loss which
orbital pairs to score; everything outside is zero-masked.

After training the function writes a ``HeadsFT_multi_final.pth``
checkpoint that can be loaded into :class:`tailwater.HeadsOnly` for
inference on new materials — pass it as the ``heads_checkpoint``
argument of any forward-pass workflow.
"""

import os
import time
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import tbmodels

from .constants import NeighBrs
from .heads_only_model import load_heads_only_checkpoint
from .subspace_utils import (
    Subspace_EigLoss,
    Subspace_H_MSE_Loss,
    Eigenvalue_Only_Loss,
)
from .finetune_heads import _DEFAULT_HEADS_CHECKPOINT, _build_kgrid


# ---------------------------------------------------------------------
# 18-orbital basis: spatial label → spatial index
# ---------------------------------------------------------------------
# Same convention as the rest of the package:
#     orbital_index_in_18 = spatial_index * 2 + spin_index
SPATIAL_LABEL_TO_INDEX: Dict[str, int] = {
    "s":      0,
    "pz":     1,
    "px":     2,
    "py":     3,
    "dz2":    4,
    "dxz":    5,
    "dyz":    6,
    "dx2-y2": 7,
    "dxy":    8,
}

# Wannier90 shell → ordered list of spatial-orbital labels in the
# canonical 18-basis order. Matches the row-major iteration order
# `process_win` uses on the server side, so the compact orbital
# indices in the user's hr-file line up with the (atom, orbital_in_18)
# slots we read out below.
_SHELL_TO_SPATIAL: Dict[str, List[str]] = {
    "s": ["s"],
    "p": ["pz", "px", "py"],
    "d": ["dz2", "dxz", "dyz", "dx2-y2", "dxy"],
}


# ---------------------------------------------------------------------
# Wannier90 .win parsing  (mirrors structure_io.process_win)
# ---------------------------------------------------------------------
def parse_win_projections(win_path: str) -> Dict[str, List[str]]:
    """Parse the ``begin_projections / end_projections`` block of a .win file.

    Returns a dict mapping each element symbol to its list of shell
    labels (``"s"``, ``"p"``, ``"d"``), in the order they appear on the
    projection line. Example:

        Bi : s, p, d
        Se : s, p

    becomes ``{"Bi": ["s", "p", "d"], "Se": ["s", "p"]}``.
    """
    with open(win_path, "r") as f:
        lines = f.readlines()
    inside  = False
    proj_lines: List[str] = []
    for line in lines:
        line = line.strip()
        if line.lower().startswith("begin_projections"):
            inside = True; continue
        if line.lower().startswith("end_projections"):
            break
        if inside and line:
            proj_lines.append(line)

    if not proj_lines:
        raise ValueError(
            f"No projections block found in {win_path!r}. Expected lines "
            f"between `begin_projections` and `end_projections`."
        )

    out: Dict[str, List[str]] = {}
    for ln in proj_lines:
        cleaned = ln.replace(" ", "")
        if ":" not in cleaned:
            raise ValueError(
                f"Malformed projection line in {win_path!r}: {ln!r}. "
                f"Expected `Element: shell, shell, ...`."
            )
        sym, orb_str = cleaned.split(":", 1)
        shells = [s for s in orb_str.split(",") if s]
        for sh in shells:
            if sh not in _SHELL_TO_SPATIAL:
                raise ValueError(
                    f"Unknown shell label {sh!r} for element {sym!r} in "
                    f"{win_path!r}. Supported: {sorted(_SHELL_TO_SPATIAL)}."
                )
        out[sym] = shells
    return out


def parse_win_atoms(win_path: str) -> List[Tuple[str, List[float]]]:
    """Parse the ``begin atoms_cart / end atoms_cart`` block.

    Returns ``[(element_symbol, [x, y, z]), ...]`` in the order the
    atoms appear in the file. This is the same order the compact
    orbital indices follow in the user's hr-file.
    """
    with open(win_path, "r") as f:
        lines = f.readlines()
    inside = False
    atoms: List[Tuple[str, List[float]]] = []
    for line in lines:
        line = line.strip()
        if line.lower().startswith("begin atoms_cart"):
            inside = True; continue
        if line.lower().startswith("end atoms_cart"):
            break
        if inside and line:
            parts = line.split()
            if len(parts) >= 4:
                atoms.append((parts[0], [float(x) for x in parts[1:4]]))
    if not atoms:
        raise ValueError(
            f"No atoms_cart block found in {win_path!r}. Expected lines "
            f"between `begin atoms_cart` and `end atoms_cart`."
        )
    return atoms


def active_orbitals_from_win(win_path: str) -> List[List[str]]:
    """Per-atom list of spatial-orbital labels for the 18-basis active mask.

    Combines :func:`parse_win_projections` and :func:`parse_win_atoms`,
    walking the atoms in .win order and expanding each shell label
    (``"s" / "p" / "d"``) into the canonical Tailwater spatial-orbital
    sequence (``["s"]`` / ``["pz", "px", "py"]`` / ``["dz2", "dxz",
    "dyz", "dx2-y2", "dxy"]``). Order matters: it determines the
    compact-orbital → ``(atom, orbital_in_18)`` mapping used by
    :func:`build_edge_targets_from_hr`.

    Returns
    -------
    list of list of str
        ``out[a]`` is the spatial-orbital list for atom ``a``,
        suitable as the ``active_orbitals`` argument to
        :func:`prepare_finetune_target` /
        :func:`build_active_mask` / :func:`build_edge_targets_from_hr`.
    """
    projections = parse_win_projections(win_path)
    atoms       = parse_win_atoms(win_path)

    out: List[List[str]] = []
    for sym, _pos in atoms:
        if sym not in projections:
            raise ValueError(
                f"Atom {sym!r} in {win_path!r} has no matching entry in the "
                f"projections block. Add it (e.g. `{sym}: s, p, d`) or remove "
                f"the atom."
            )
        spatial: List[str] = []
        for sh in projections[sym]:
            spatial.extend(_SHELL_TO_SPATIAL[sh])
        out.append(spatial)
    return out


# ---------------------------------------------------------------------
# Active-orbital mask construction
# ---------------------------------------------------------------------
def build_active_mask(active_orbitals: Sequence[Sequence[str]]) -> np.ndarray:
    """Convert per-atom spatial-orbital lists into the 18-flag active mask.

    Args
    ----
    active_orbitals : sequence of (sequence of str)
        ``active_orbitals[a]`` lists the spatial orbital labels
        present on atom ``a`` (subset of
        ``{"s", "pz", "px", "py", "dz2", "dxz", "dyz", "dx2-y2", "dxy"}``).
        Each spatial slot covers BOTH spin partners automatically.

    Returns
    -------
    np.ndarray of shape ``(num_atoms, 18)``, float
        ``1.0`` at orbital positions ``(spatial*2, spatial*2+1)`` for
        every spatial label in ``active_orbitals[a]``, ``0.0`` elsewhere.
        This is the array that lives at
        ``gdata.node_features[a, 109:127]``.
    """
    num_atoms = len(active_orbitals)
    mask = np.zeros((num_atoms, 18), dtype=np.float32)
    for a, spatial_list in enumerate(active_orbitals):
        for sp_label in spatial_list:
            if sp_label not in SPATIAL_LABEL_TO_INDEX:
                raise ValueError(
                    f"Unknown spatial orbital label {sp_label!r} for atom {a}. "
                    f"Valid labels: {sorted(SPATIAL_LABEL_TO_INDEX)}."
                )
            sp_idx = SPATIAL_LABEL_TO_INDEX[sp_label]
            mask[a, sp_idx * 2]     = 1.0
            mask[a, sp_idx * 2 + 1] = 1.0
    return mask


def _compact_orbital_map(
    active_orbitals: Sequence[Sequence[str]],
) -> Tuple[Dict[Tuple[int, int], int], int]:
    """Map ``(atom_idx, orbital_in_18) → compact_idx_in_user_hr``.

    Assumes the user's hr-model was built in row-major order over
    atoms; within each atom, the spatial orbitals appear in the
    order given by ``active_orbitals[a]``, and for each spatial
    orbital the up partner precedes the down partner.

    Returns
    -------
    mapping : dict
        ``{(atom_idx, orbital_in_18): compact_idx}``.
    total : int
        Total number of compact orbitals — must equal ``hr_model.size``.
    """
    mapping: Dict[Tuple[int, int], int] = {}
    cnt = 0
    for a, spatial_list in enumerate(active_orbitals):
        for sp_label in spatial_list:
            sp_idx = SPATIAL_LABEL_TO_INDEX[sp_label]
            for spin in (0, 1):
                mapping[(a, sp_idx * 2 + spin)] = cnt
                cnt += 1
    return mapping, cnt


# ---------------------------------------------------------------------
# Edge-target tensor from user's hr-model
# ---------------------------------------------------------------------
def build_edge_targets_from_hr(
    gdata,
    hr_model: tbmodels.Model,
    active_orbitals: Sequence[Sequence[str]],
    *,
    fermi_shift: float = 0.0,
) -> torch.Tensor:
    """Fill ``edge_targets[num_edges, 18, 18, 2]`` from the user's hr.

    For each PyG edge ``e = (atm1, atm2, R)`` and each orbital pair
    ``(orbital_in_18_i, orbital_in_18_j)`` that is active on its atom,
    looks up the corresponding matrix element in the user's hr-model
    and writes ``(Re, Im)`` into the target tensor. Inactive orbital
    pairs are left at zero.

    Handles the tbmodels (0,0,0)-doubling convention internally:
    ``H_physical(R=0) = 2 × hr_model.hop[(0,0,0)]`` because tbmodels'
    ``add_hop`` halves R=0 entries on input and ``hamilton(k)``
    re-doubles them on output. For R ≠ 0 the stored value is the
    physical hop.

    Args
    ----
    gdata : torch_geometric.data.Data
        PyG graph from the API embedding ``.pt``. Must carry
        ``edge_index``, ``edge_vectors``, ``inv_data``.
    hr_model : tbmodels.Model
        The user's Wannier Hamiltonian. ``hr_model.size`` must equal
        ``sum(2 * len(spatial_list) for spatial_list in active_orbitals)``.
    active_orbitals : sequence of (sequence of str)
        Per-atom spatial-orbital labels — see :func:`build_active_mask`.
    fermi_shift : float, default 0.0
        Constant subtracted from on-site diagonal entries before they
        become the target. Use this to align the user's hr-model
        Fermi reference with the Tailwater convention (E_F = 0). The
        shift is applied to *physical* on-site energies (after the 2×
        unfolding), so it works regardless of the user's input
        convention.

    Returns
    -------
    torch.Tensor of shape ``(num_edges, 18, 18, 2)``, float32
        ``[e, orbital_i, orbital_j, 0]`` and ``[..., 1]`` are the
        real and imaginary parts of the ``(orbital_i, orbital_j)``
        matrix element on edge ``e``.
    """
    # Resolve and sanity-check the orbital map.
    mapping, n_compact = _compact_orbital_map(active_orbitals)
    if n_compact != int(hr_model.size):
        raise ValueError(
            f"Orbital-map size ({n_compact}) does not match hr_model.size "
            f"({hr_model.size}). Check that `active_orbitals` matches the "
            f"projection scheme of the hr-file."
        )

    inv_data     = gdata.inv_data.detach().cpu().numpy()        # [num_edges, 3]
    edge_vectors = gdata.edge_vectors.detach().cpu().numpy()    # [num_edges, 3]
    num_edges    = int(inv_data.shape[0])

    # Per-atom active orbital_in_18 lists — precompute for inner loop speed.
    active_per_atom: List[List[int]] = []
    for spatial_list in active_orbitals:
        flat: List[int] = []
        for sp_label in spatial_list:
            sp_idx = SPATIAL_LABEL_TO_INDEX[sp_label]
            flat.append(sp_idx * 2)
            flat.append(sp_idx * 2 + 1)
        active_per_atom.append(flat)

    # Cache the R-blocks as dense complex arrays (one per stored R).
    hop_cache: Dict[Tuple[int, int, int], np.ndarray] = {}
    for R, blk in hr_model.hop.items():
        R_tup = tuple(int(x) for x in R)
        arr   = np.asarray(blk.toarray() if hasattr(blk, "toarray") else blk,
                           dtype=complex)
        hop_cache[R_tup] = arr

    targets = np.zeros((num_edges, 18, 18, 2), dtype=np.float32)

    for e in range(num_edges):
        atm1, atm2, R_idx = int(inv_data[e, 0]), int(inv_data[e, 1]), int(inv_data[e, 2])
        R_vec = tuple(int(x) for x in NeighBrs[R_idx])
        blk   = hop_cache.get(R_vec)
        if blk is None:
            continue                                # target stays zero
        is_self_loop = bool(np.linalg.norm(edge_vectors[e]) == 0)
        scale = 2.0 if R_vec == (0, 0, 0) else 1.0   # un-do tbmodels (0,0,0) halving

        for orbital_i in active_per_atom[atm1]:
            ci = mapping[(atm1, orbital_i)]
            for orbital_j in active_per_atom[atm2]:
                cj = mapping[(atm2, orbital_j)]
                val = scale * blk[ci, cj]
                if is_self_loop and orbital_i == orbital_j:
                    val = val - fermi_shift
                targets[e, orbital_i, orbital_j, 0] = float(val.real)
                targets[e, orbital_i, orbital_j, 1] = float(val.imag)

    return torch.from_numpy(targets)


# ---------------------------------------------------------------------
# Per-material preparation
# ---------------------------------------------------------------------
def prepare_finetune_target(
    embed_path: str,
    hr_path_or_model: Union[str, tbmodels.Model],
    win_path: Optional[str] = None,
    *,
    active_orbitals: Optional[Sequence[Sequence[str]]] = None,
    fermi_shift: float = 0.0,
    out_path: Optional[str] = None,
    name: Optional[str] = None,
) -> dict:
    """Build a per-material training/validation item.

    Merges the API-supplied embedding (PyG ``Data`` with ``f_out`` /
    ``edge_feat`` from the backbone) with user-supplied targets
    derived from the user's Wannier hr-model.

    The canonical usage is to point at the same ``.win`` file you fed
    Wannier90 — the projection block (e.g. ``Bi: s, p, d`` /
    ``Se: s, p``) and the ``atoms_cart`` block together fully
    determine the per-atom orbital layout, and the helper
    :func:`active_orbitals_from_win` reconstructs the per-atom spatial
    list automatically. Customers therefore only need to provide
    ``(embed_path, hr_path, win_path)`` for each material:

    .. code-block:: python

        item = prepare_finetune_target(
            embed_path="outputs/Bi2Se3_embeddings.pt",
            hr_path_or_model="wannier_data/Bi2Se3_hr.dat",
            win_path="wannier_data/Bi2Se3.win",
        )

    Args
    ----
    embed_path : str
        Path to the embedding ``.pt`` from the API.
    hr_path_or_model : str | tbmodels.Model
        The user's Wannier Hamiltonian — either a path readable by
        ``tbmodels.Model.from_hdf5_file`` / ``from_hr_file`` (chosen
        by extension), or an already-loaded ``tbmodels.Model``.
    win_path : str, optional
        Path to the matching ``.win`` file. When supplied,
        ``active_orbitals`` is derived from it automatically via
        :func:`active_orbitals_from_win`. Required unless you pass
        ``active_orbitals`` explicitly.
    active_orbitals : sequence of (sequence of str), optional
        Explicit per-atom spatial-orbital labels — see
        :func:`build_active_mask`. Overrides ``win_path`` if both are
        given. Useful when the user wants to fine-tune on a subspace
        smaller than the full Wannier projection.
    fermi_shift : float, default 0.0
        Subtract this from on-site diagonal entries to align the
        user's hr Fermi reference with the Tailwater convention.
    out_path : str, optional
        If given, the prepared item is also saved as a ``.pt`` file
        at this path for reuse in subsequent fine-tune runs.
    name : str, optional
        Free-form label for this material (echoed in training logs).
        Defaults to the embedding filename stem.

    Returns
    -------
    dict with the following keys:

      * ``"name"``       : str
      * ``"gdata"``      : PyG ``Data`` with ``edge_targets``
                            already attached and ``node_features``
                            updated with the (user- or .win-derived)
                            active mask.
      * ``"embed_path"`` : str (the input path, for traceability).
    """
    if not os.path.isfile(embed_path):
        raise FileNotFoundError(f"embed_path does not point to a file: {embed_path!r}")
    embed_pkg = torch.load(embed_path, map_location="cpu", weights_only=False)
    if not (isinstance(embed_pkg, dict) and "data" in embed_pkg):
        raise ValueError(
            f"embed_path must point to a dict-format .pt with a `data` key "
            f"(format produced by the API's "
            f"/upload_structure_and_download_embeddings/ endpoint). "
            f"Got top-level type {type(embed_pkg).__name__}."
        )
    gdata = embed_pkg["data"]

    # Resolve the hr-model
    if isinstance(hr_path_or_model, str):
        path = hr_path_or_model
        if path.lower().endswith((".hdf5", ".h5")):
            hr_model = tbmodels.Model.from_hdf5_file(path)
        else:
            hr_model = tbmodels.Model.from_hr_file(path)
    else:
        hr_model = hr_path_or_model

    # Resolve the per-atom active orbital layout.
    if active_orbitals is None:
        if win_path is None:
            raise ValueError(
                "Either `win_path` or `active_orbitals` must be supplied. "
                "Pass `win_path` to derive the layout from the Wannier90 "
                ".win file's projection + atoms_cart blocks automatically, "
                "or pass `active_orbitals` explicitly to override."
            )
        if not os.path.isfile(win_path):
            raise FileNotFoundError(f"win_path does not point to a file: {win_path!r}")
        active_orbitals = active_orbitals_from_win(win_path)

    # Sanity: atom count must match between API graph and the resolved layout.
    num_atoms_api = int(gdata.node_features.shape[0])
    if len(active_orbitals) != num_atoms_api:
        src = (f"`active_orbitals` (explicit)" if win_path is None
               else f"`win_path`={win_path!r}")
        raise ValueError(
            f"{src} describes {len(active_orbitals)} atoms but the API "
            f"embedding describes {num_atoms_api} atoms. They must match — "
            f"the .win file used to build the user's hr must be the same "
            f"structure as the one uploaded to the API."
        )

    # Build the (num_edges, 18, 18, 2) target tensor and unsqueeze the
    # leading K-axis the subspace losses expect.
    targets = build_edge_targets_from_hr(
        gdata, hr_model, active_orbitals, fermi_shift=fermi_shift,
    )
    gdata.edge_targets = targets.unsqueeze(1)        # (num_edges, 1, 18, 18, 2)

    # Stamp the user-supplied active mask into node_features[:, 109:127].
    active_mask = build_active_mask(active_orbitals)
    nf = gdata.node_features.clone()
    nf[:, 109:127] = torch.from_numpy(active_mask).to(nf.dtype)
    gdata.node_features = nf

    if name is None:
        name = os.path.splitext(os.path.basename(embed_path))[0]

    item = {
        "name":       name,
        "gdata":      gdata,
        "embed_path": embed_path,
    }
    if out_path is not None:
        torch.save(item, out_path)
    return item


# ---------------------------------------------------------------------
# Directory-based bulk discovery
# ---------------------------------------------------------------------
def _first_glob_match(
    directory: str,
    patterns: Sequence[str],
) -> Optional[str]:
    """Return the first file in ``directory`` matching any of ``patterns``.

    Patterns are case-insensitive; a deterministic sort within each
    pattern ensures reproducible behavior across runs / filesystems.
    """
    import fnmatch
    try:
        names = sorted(os.listdir(directory))
    except (FileNotFoundError, NotADirectoryError):
        return None
    names_lower = [n.lower() for n in names]
    for pat in patterns:
        plow = pat.lower()
        for n, nl in zip(names, names_lower):
            if fnmatch.fnmatch(nl, plow):
                return os.path.join(directory, n)
    return None


def prepare_finetune_targets_from_directory(
    root_dir: str,
    *,
    embed_patterns: Sequence[str] = ("*embeddings*.pt", "*embedding*.pt"),
    win_patterns:   Sequence[str] = ("*.win",),
    hr_patterns:    Sequence[str] = ("*_hr.dat", "*_hr.hdf5", "*_hr.h5"),
    out_dir:        Optional[str] = None,
    fermi_shift:    float = 0.0,
    strict:         bool  = False,
    sort_names:     bool  = True,
) -> List[dict]:
    """Auto-discover (embedding, hr, .win) triples from a tree of subdirectories.

    Layout convention::

        root_dir/
        ├── Bi2Se3/
        │   ├── embeddings.pt      <- from `/upload_structure_and_download_embeddings/`
        │   ├── wannier90.win      <- the user's
        │   └── wannier90_hr.dat   <- the user's (or `_hr.hdf5`)
        ├── Bi2Te3/
        │   ├── embeddings.pt
        │   ├── wannier90.win
        │   └── wannier90_hr.dat
        ├── ...
        └── SnTe/
            └── ...

    One subdirectory per material; the **subdirectory name becomes the
    material name** for training logs and the cached ``.pt`` filename.
    Filenames inside each subdirectory can vary — the patterns below
    cover the common conventions.

    The function calls :func:`prepare_finetune_target` once per
    discovered subdirectory and returns the list of prepared items
    ready to hand to :func:`finetune_heads_multi`.

    Args
    ----
    root_dir : str
        Path to the directory containing one subdirectory per material.
    embed_patterns : sequence of str, default ``("*embeddings*.pt",
                     "*embedding*.pt")``
        Glob patterns (case-insensitive) used to locate the API
        embedding file within each subdirectory. First match wins.
    win_patterns : sequence of str, default ``("*.win",)``
        Glob patterns for the Wannier90 .win file.
    hr_patterns : sequence of str, default ``("*_hr.dat", "*_hr.hdf5",
                  "*_hr.h5")``
        Glob patterns for the Wannier hr-model file. ``.dat`` is tried
        before HDF5; either is read transparently by
        ``tbmodels.Model``.
    out_dir : str, optional
        If given, each prepared item is also saved as
        ``out_dir/{name}_target.pt`` for reuse on subsequent runs.
    fermi_shift : float, default 0.0
        Forwarded to :func:`prepare_finetune_target` for every material.
        Use this to align the user's hr-model Fermi reference with the
        Tailwater convention if the hr-files are not already pre-shifted.
    strict : bool, default False
        If True, raise on the first subdirectory that's missing any of
        the three required files. If False (default), skip that
        subdirectory with a warning and continue.
    sort_names : bool, default True
        Process subdirectories in sorted-by-name order for reproducible
        training-set ordering.

    Returns
    -------
    list of dict
        One entry per successfully-prepared material — same dict
        format as :func:`prepare_finetune_target` (with the material
        name set to the subdirectory name). Ready to pass straight to
        :func:`finetune_heads_multi` as ``train_targets`` /
        ``val_targets``.

    Example
    -------
    .. code-block:: python

        from tailwater import (
            prepare_finetune_targets_from_directory,
            finetune_heads_multi,
        )

        train_items = prepare_finetune_targets_from_directory(
            "datasets/train",
            out_dir="finetune_out/cache",
        )
        val_items   = prepare_finetune_targets_from_directory(
            "datasets/val",
            out_dir="finetune_out/cache",
        )

        finetune_heads_multi(
            train_targets = train_items,
            val_targets   = val_items,
            start_lr      = 5e-5, end_lr = 5e-7, num_epochs = 50,
            energy_range  = (-2.0, 2.0),
            decay_sigma   = 1.0,
            device        = "cuda",
            save_path     = "finetune_out",
        )
    """
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"root_dir does not exist or is not a directory: {root_dir!r}")
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

    entries = sorted(os.listdir(root_dir)) if sort_names else os.listdir(root_dir)
    subdirs = [
        os.path.join(root_dir, name)
        for name in entries
        if os.path.isdir(os.path.join(root_dir, name))
    ]
    if not subdirs:
        raise ValueError(
            f"No subdirectories found in {root_dir!r}. Each material should "
            f"live in its own subdirectory containing the API embedding, the "
            f"user's .win, and the user's hr-file."
        )

    items: List[dict] = []
    skipped: List[Tuple[str, str]] = []
    print(f"Scanning {root_dir!r} for (embedding, .win, hr) triples in "
          f"{len(subdirs)} subdirectories ...")
    for sd in subdirs:
        name = os.path.basename(sd)
        emb  = _first_glob_match(sd, embed_patterns)
        win  = _first_glob_match(sd, win_patterns)
        hrp  = _first_glob_match(sd, hr_patterns)

        missing = []
        if emb is None: missing.append("embedding")
        if win is None: missing.append(".win")
        if hrp is None: missing.append("hr-file")
        if missing:
            msg = (f"{name}: missing {', '.join(missing)} "
                   f"(in {sd!r})")
            if strict:
                raise FileNotFoundError(msg)
            print(f"  [skip] {msg}")
            skipped.append((name, ", ".join(missing)))
            continue

        out_path = (os.path.join(out_dir, f"{name}_target.pt")
                    if out_dir is not None else None)
        try:
            item = prepare_finetune_target(
                embed_path        = emb,
                hr_path_or_model  = hrp,
                win_path          = win,
                fermi_shift       = fermi_shift,
                out_path          = out_path,
                name              = name,
            )
        except Exception as exc:
            if strict:
                raise
            print(f"  [skip] {name}: {type(exc).__name__}: {exc}")
            skipped.append((name, f"{type(exc).__name__}: {exc}"))
            continue

        edges = int(item["gdata"].edge_targets.shape[0])
        active = int(item["gdata"].node_features[:, 109:127].sum())
        print(f"  [ok]   {name}: {edges} edges, {active} active orbitals  "
              f"(embed={os.path.basename(emb)}, win={os.path.basename(win)}, "
              f"hr={os.path.basename(hrp)})")
        items.append(item)

    print(f"\nPrepared {len(items)} materials"
          f"{f' ({len(skipped)} skipped)' if skipped else ''}.")
    return items


# ---------------------------------------------------------------------
# Multi-material fine-tuning
# ---------------------------------------------------------------------
def _load_item(item: Union[str, dict]) -> dict:
    """Resolve a target either as a dict (already prepared) or a .pt path."""
    if isinstance(item, dict):
        return item
    if isinstance(item, str):
        return torch.load(item, map_location="cpu", weights_only=False)
    raise TypeError(
        f"Targets must be dicts (from prepare_finetune_target) or paths to "
        f".pt files written by prepare_finetune_target — got {type(item).__name__}."
    )


def _losses(
    gdata,
    edge_pred_r: torch.Tensor,
    onsite_pred_r: torch.Tensor,
    e_lo: float,
    e_hi: float,
    decay_sigma: float,
    KVECS: torch.Tensor,
    h_mse_weight: float,
    eig_weight: float,
    loss_mode: str,
) -> Tuple[torch.Tensor, float]:
    """Compute total loss + eigenvalue scalar for one material."""
    if loss_mode == "subspace":
        hmse = Subspace_H_MSE_Loss(gdata, edge_pred_r, onsite_pred_r, e_lo, e_hi)
        eig  = Subspace_EigLoss(
            gdata, edge_pred_r, onsite_pred_r,
            KVECS, NeighBrs, e_lo, e_hi, decay_sigma=decay_sigma,
        )
        return (h_mse_weight * hmse + eig_weight * eig), float(eig.item())
    elif loss_mode == "eig_only":
        eig = Eigenvalue_Only_Loss(
            gdata, edge_pred_r, onsite_pred_r, e_lo, e_hi, decay_sigma=decay_sigma,
        )
        return eig, float(eig.item())
    raise ValueError(f"loss_mode must be 'subspace' or 'eig_only', got {loss_mode!r}")


def finetune_heads_multi(
    train_targets: Sequence[Union[str, dict]],
    *,
    start_lr: float,
    end_lr: float,
    num_epochs: int,
    energy_range: Tuple[float, float],
    decay_sigma: float,
    device,
    save_path: str,
    val_targets: Optional[Sequence[Union[str, dict]]] = None,
    val_every: int = 5,
    loss_mode: str = "subspace",
    heads_checkpoint: Optional[str] = None,
    h_mse_weight: float = 0.001,
    eig_weight: float = 1.0,
    kgrid_n: int = 4,
    grad_clip: float = 1.0,
) -> str:
    """Fine-tune the heads on a set of (API-embedding, user-hr) pairs.

    Args
    ----
    train_targets : sequence of dict or str
        Each entry is either a dict from :func:`prepare_finetune_target`
        or a path to a ``.pt`` file written by that function.
    start_lr, end_lr, num_epochs
        Cosine-annealed AdamW. One *epoch* visits every material in
        ``train_targets`` once, accumulating gradients across them,
        then takes a single optimizer step (full-batch GD over
        materials).
    energy_range : (float, float)
        Subspace energy window ``(e_lo, e_hi)``, eV. Eigenvalues outside
        this window are excluded from the loss for *every* material.
    decay_sigma : float
        Gaussian sigma for eigenvalue weighting around the window center.
    device : str | torch.device
        Where to run the forward / backward. Each material's PyG ``Data``
        is moved here per-step and freed immediately after; only the
        heads model lives on ``device`` for the duration of training.
    save_path : str
        Output directory (created if missing). Final checkpoint lands at
        ``save_path/HeadsFT_multi_final.pth``.
    val_targets : sequence of dict or str, optional
        Held-out validation materials. If supplied, the mean validation
        eigenvalue loss is reported every ``val_every`` epochs and the
        best-val checkpoint is kept at
        ``save_path/HeadsFT_multi_best.pth``.
    val_every : int, default 5
        Validation cadence in epochs.
    loss_mode : {"subspace", "eig_only"}, default "subspace"
        Which loss formula to use (mirrors :func:`subspace_projection`).
    heads_checkpoint : str, optional
        Starting HeadsOnly checkpoint. Defaults to the packaged
        ``HeadsOnly_MACE.pth`` bundled with tailwater.
    h_mse_weight, eig_weight : float
        Per-term weights for the ``"subspace"`` loss combination.
    kgrid_n : int, default 4
        Size of the k-mesh used in the eigenvalue loss (n × n × n
        Monkhorst-Pack sampling, same convention as
        :func:`subspace_projection`).
    grad_clip : float, default 1.0
        Max ℓ₂ norm for ``torch.nn.utils.clip_grad_norm_``. Set to
        ``0.0`` or a very large value to disable.

    Returns
    -------
    str
        Path to the final HeadsFT checkpoint (loadable into
        :class:`tailwater.HeadsOnly` for downstream inference).
    """
    if loss_mode not in ("subspace", "eig_only"):
        raise ValueError(f"loss_mode must be 'subspace' or 'eig_only', got {loss_mode!r}")
    if len(train_targets) == 0:
        raise ValueError("train_targets is empty — provide at least one material.")

    e_lo, e_hi = float(energy_range[0]), float(energy_range[1])

    os.makedirs(save_path, exist_ok=True)
    device = torch.device(device) if isinstance(device, str) else device

    # ---- Load all train + val items into CPU memory once ----
    train_items = [_load_item(t) for t in train_targets]
    val_items   = [_load_item(t) for t in (val_targets or [])]

    print(f"[multi-finetune] train = {len(train_items)} materials, "
          f"val = {len(val_items)} materials, "
          f"loss_mode = {loss_mode!r}, "
          f"window = [{e_lo:+.3f}, {e_hi:+.3f}] eV")

    # ---- Heads model ----
    if heads_checkpoint is None:
        heads_checkpoint = _DEFAULT_HEADS_CHECKPOINT
        print(f"[model] using packaged HeadsOnly checkpoint: {heads_checkpoint}")
    model = load_heads_only_checkpoint(heads_checkpoint, map_location=device).to(device)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[model] HeadsOnly loaded — {n_total:,} trainable params, "
          f"irreps_in = {model.irreps_in}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=start_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, num_epochs), eta_min=end_lr,
    )

    KVECS = _build_kgrid(kgrid_n)

    train_hist: List[Tuple[int, float, float]]   = []     # (epoch, total_loss, eig)
    val_hist:   List[Tuple[int, float]]          = []     # (epoch, mean_eig)
    best_val   = float("inf")
    best_path  = os.path.join(save_path, "HeadsFT_multi_best.pth")
    final_path = os.path.join(save_path, "HeadsFT_multi_final.pth")

    start_time = time.perf_counter()
    for epoch in range(num_epochs):
        # -------------------------- train ---------------------------
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_eig  = 0.0
        valid_n    = 0

        for item in train_items:
            gdata = item["gdata"].to(device)
            try:
                edge_pred, onsite_pred = model(gdata)
                edge_pred_r   = edge_pred.reshape  (gdata.edge_index.shape[1], 18, 18, 2)
                onsite_pred_r = onsite_pred.reshape(gdata.node_features.shape[0], 18, 18, 2)
                loss, eig_val = _losses(
                    gdata, edge_pred_r, onsite_pred_r,
                    e_lo, e_hi, decay_sigma, KVECS,
                    h_mse_weight, eig_weight, loss_mode,
                )
                if not torch.isfinite(loss):
                    print(f"  [warn] non-finite loss on {item['name']!r} "
                          f"@ epoch {epoch+1}; skipping this material")
                    continue
                loss.backward()
                total_loss += float(loss.item())
                total_eig  += eig_val
                valid_n    += 1
            except torch.cuda.OutOfMemoryError:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                print(f"  [warn] CUDA OOM on {item['name']!r} @ epoch {epoch+1}; "
                      f"skipping this material")

        if valid_n == 0:
            print(f"Epoch {epoch+1:>3d}/{num_epochs}  ALL non-finite — no step")
            scheduler.step()
            continue

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        scheduler.step()

        mean_loss = total_loss / valid_n
        mean_eig  = total_eig  / valid_n
        train_hist.append((epoch + 1, mean_loss, mean_eig))
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:>3d}/{num_epochs}  LR {lr_now:.3e}  "
              f"mean loss {mean_loss:.6e}  mean eig {mean_eig:.6e}  "
              f"({valid_n}/{len(train_items)} materials)")

        # ----------------------- validation -------------------------
        if val_items and ((epoch + 1) % val_every == 0 or epoch + 1 == num_epochs):
            model.eval()
            v_eig = 0.0
            v_n   = 0
            with torch.no_grad():
                for v in val_items:
                    g = v["gdata"].to(device)
                    edge_pred, onsite_pred = model(g)
                    edge_pred_r   = edge_pred.reshape  (g.edge_index.shape[1], 18, 18, 2)
                    onsite_pred_r = onsite_pred.reshape(g.node_features.shape[0], 18, 18, 2)
                    _, eig_val = _losses(
                        g, edge_pred_r, onsite_pred_r,
                        e_lo, e_hi, decay_sigma, KVECS,
                        h_mse_weight, eig_weight, loss_mode,
                    )
                    v_eig += eig_val
                    v_n   += 1
            mean_v = v_eig / max(1, v_n)
            val_hist.append((epoch + 1, mean_v))
            tag = ""
            if mean_v < best_val:
                best_val = mean_v
                torch.save(
                    {
                        "heads_state_dict":  model.state_dict(),
                        "irreps_in":         str(model.irreps_in),
                        "energy_range":      (e_lo, e_hi),
                        "epoch":             epoch + 1,
                        "val_eig":           best_val,
                    },
                    best_path,
                )
                tag = "  *NEW BEST*"
            print(f"       [val]   mean eig {mean_v:.6e}{tag}")

    elapsed = time.perf_counter() - start_time
    print(f"\n[train] {num_epochs} epochs in {elapsed:.1f} s "
          f"({elapsed / max(1, num_epochs):.2f} s / epoch)")

    # ---- Save the final checkpoint ----
    torch.save(
        {
            "heads_state_dict":  model.state_dict(),
            "irreps_in":         str(model.irreps_in),
            "loss_mode":         loss_mode,
            "energy_range":      (e_lo, e_hi),
            "decay_sigma":       decay_sigma,
            "start_lr":          start_lr,
            "end_lr":            end_lr,
            "num_epochs":        num_epochs,
            "train_history":     train_hist,
            "val_history":       val_hist,
            "n_train_materials": len(train_items),
            "n_val_materials":   len(val_items),
            "best_val_eig":      best_val if val_items else None,
        },
        final_path,
    )
    print(f"[save] final HeadsFT_multi -> {final_path}")
    if val_items:
        print(f"[save] best-val HeadsFT_multi -> {best_path}  (val_eig = {best_val:.6e})")
    return final_path

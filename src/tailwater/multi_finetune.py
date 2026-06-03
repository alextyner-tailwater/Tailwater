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

# Compact-orbital count per atom → canonical Wannier shell combination.
# Every standard Wannier projection produces an unambiguous count
# (s↑↓=2, p↑↓=6, d↑↓=10), so the per-atom orbital count in an hr-file
# uniquely identifies its shell set. Used as the fallback when the
# .win's projection block doesn't agree with the hr-file structure
# (e.g. the user keeps the API-style full-projection .win in the
# directory but the hr was Wannierized with a restricted projection).
_COUNT_TO_SHELLS: Dict[int, List[str]] = {
    2:  ["s"],
    6:  ["p"],
    10: ["d"],
    8:  ["s", "p"],
    12: ["s", "d"],
    16: ["p", "d"],
    18: ["s", "p", "d"],
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


def parse_win_fermi_energy(win_path: str) -> Optional[float]:
    """Read the ``fermi_energy`` keyword from a Wannier90 .win file.

    Wannier90 accepts the keyword in several whitespace / punctuation
    variants — ``fermi_energy = 1.234``, ``fermi_energy 1.234``,
    ``fermi_energy: 1.234``, all case-insensitive. This helper handles
    all of them and returns the numeric value (in eV).

    Returns
    -------
    float or None
        The parsed Fermi energy in eV, or ``None`` if the keyword is
        not present in the file. Lines inside ``begin ... end`` blocks
        and comment lines (starting with ``!`` or ``#``) are ignored.
    """
    import re
    if not os.path.isfile(win_path):
        raise FileNotFoundError(f"win_path does not point to a file: {win_path!r}")

    # Strip comments + skip block bodies — fermi_energy is always a
    # top-level keyword, never inside an explicit begin/end block.
    in_block = False
    with open(win_path, "r") as f:
        for raw in f:
            line = raw.split("!", 1)[0].split("#", 1)[0].strip()
            if not line:
                continue
            low = line.lower()
            if low.startswith("begin"):
                in_block = True; continue
            if low.startswith("end"):
                in_block = False; continue
            if in_block:
                continue
            if low.startswith("fermi_energy"):
                # Strip the keyword and any =/: separator, then grab the
                # first floating-point token.
                rest = line[len("fermi_energy"):].lstrip()
                rest = rest.lstrip(":").lstrip("=").strip()
                m = re.match(r"([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)", rest)
                if m is None:
                    # Template placeholder like `fermi_energy = efermi` —
                    # treat as "absent" so calling code can default to
                    # no shift, rather than failing outright.
                    import warnings as _w
                    _w.warn(
                        f"`fermi_energy` keyword in {win_path!r} is not a "
                        f"numeric value ({line!r}). Treating as absent; "
                        f"no Fermi shift will be applied for this material.",
                        RuntimeWarning,
                    )
                    return None
                return float(m.group(1))
    return None


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


def parse_win_lattice(win_path: str) -> np.ndarray:
    """Parse the ``begin unit_cell_cart / end unit_cell_cart`` block.

    Returns the 3×3 lattice matrix (rows = lattice vectors, Å) from a
    Wannier90 .win file. The optional units line (``Angstrom`` /
    ``Bohr``) is recognised and converted to Å — Wannier90's default
    is Bohr but the convention in Tailwater throughout is Å, so this
    function always returns Å.

    Raises
    ------
    ValueError
        On a missing or malformed ``unit_cell_cart`` block.
    """
    if not os.path.isfile(win_path):
        raise FileNotFoundError(f"win_path does not point to a file: {win_path!r}")
    BOHR_TO_ANG = 0.529177210903
    factor = 1.0
    vectors: List[List[float]] = []
    inside = False
    with open(win_path, "r") as f:
        for raw in f:
            line = raw.split("!", 1)[0].split("#", 1)[0].strip()
            if not line:
                continue
            low = line.lower()
            if low.startswith("begin unit_cell_cart"):
                inside = True; continue
            if low.startswith("end unit_cell_cart"):
                break
            if not inside:
                continue
            if low in ("angstrom", "ang"):
                factor = 1.0; continue
            if low == "bohr":
                factor = BOHR_TO_ANG; continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    vectors.append([float(x) for x in parts[-3:]])
                except ValueError:
                    pass
    if len(vectors) != 3:
        raise ValueError(
            f"Expected 3 lattice vectors in {win_path!r}, got {len(vectors)}."
        )
    return np.array(vectors, dtype=float) * factor


def structure_from_win(win_path: str):
    """Build a :class:`pymatgen.core.structure.Structure` from a .win file.

    Useful when the user wants to hand the structure back to the API
    (e.g. via :func:`tw_api_call`) without keeping a separate
    ``Structure.cif`` per material.

    Args
    ----
    win_path : str
        Path to a Wannier90 .win file with ``begin atoms_cart`` and
        ``begin unit_cell_cart`` blocks.

    Returns
    -------
    pymatgen.Structure
        Constructed from the parsed atoms (Cartesian coordinates) and
        the parsed lattice (Å). Element order is preserved exactly as
        in the .win file.
    """
    try:
        from pymatgen.core.structure import Structure
        from pymatgen.core.lattice   import Lattice
    except ImportError as exc:
        raise ImportError(
            "structure_from_win requires pymatgen: `pip install pymatgen`."
        ) from exc
    atoms_list = parse_win_atoms(win_path)
    lattice    = parse_win_lattice(win_path)
    species    = [sym for sym, _ in atoms_list]
    coords     = [pos for _, pos in atoms_list]
    return Structure(
        Lattice(lattice), species, coords, coords_are_cartesian=True,
    )


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


def infer_active_orbitals_from_hr(
    hr_model: tbmodels.Model,
    *,
    pos_tol: float = 1e-6,
) -> List[List[str]]:
    """Derive per-atom active-orbital labels from the hr-file alone.

    The per-orbital positions in ``hr_model.pos`` group the compact
    orbitals into atom blocks. Within each block, the number of compact
    orbitals uniquely identifies the Wannier shell set (2 = s,
    6 = p, 10 = d, 8 = s+p, 12 = s+d, 16 = p+d, 18 = s+p+d) — the
    standard combinations every Wannier projection produces.

    This is the canonical fallback when the ``.win`` file in a
    customer's directory does not match the actual orbital layout of
    the hr-file (e.g. the user keeps the API-style full-projection
    ``.win`` next to a Wannierization that used a restricted
    projection like ``W: d`` / ``Se: p``). The .win-derived layout
    would imply 18 slots per atom, while the hr-file's per-atom count
    is smaller — :func:`prepare_finetune_target` detects the mismatch
    and falls back to this function.

    Args
    ----
    hr_model : tbmodels.Model
        Loaded user Hamiltonian (e.g. via
        ``tbmodels.Model.from_wannier_files(...)``).
    pos_tol : float, default 1e-6
        Position tolerance for grouping orbitals into atom blocks.

    Returns
    -------
    list of list of str
        Same format as :func:`active_orbitals_from_win` — one
        spatial-orbital list per atom in compact-index order.

    Raises
    ------
    ValueError
        If an atom block has a non-canonical compact count (i.e. not
        in ``{2, 6, 8, 10, 12, 16, 18}``). Surfaces malformed
        hr-files or unusual Wannier setups; the user can pass
        ``active_orbitals`` explicitly to override.
    """
    pos = np.asarray(hr_model.pos)
    n   = int(pos.shape[0])
    if n == 0:
        return []
    # Group consecutive orbitals by position. tbmodels emits orbitals
    # in row-major (atom, shell, suborb) order, so atoms are contiguous.
    blocks: List[List[int]] = [[0]]
    for i in range(1, n):
        if np.linalg.norm(pos[i] - pos[blocks[-1][-1]]) <= pos_tol:
            blocks[-1].append(i)
        else:
            blocks.append([i])

    out: List[List[str]] = []
    for block in blocks:
        count = len(block)
        if count not in _COUNT_TO_SHELLS:
            raise ValueError(
                f"Atom block of size {count} cannot be mapped to a standard "
                f"Wannier shell combination. Supported per-atom compact "
                f"counts are: {sorted(_COUNT_TO_SHELLS)} "
                f"(s↑↓=2, p↑↓=6, d↑↓=10, and their canonical sums). "
                f"If your model uses a non-standard projection, pass "
                f"`active_orbitals` explicitly to `prepare_finetune_target`."
            )
        spatial: List[str] = []
        for sh in _COUNT_TO_SHELLS[count]:
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
    fermi_shift: Optional[float] = None,
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
    fermi_shift : float, optional
        Subtract this value (in eV) from every on-site diagonal entry
        so the resulting target has the Tailwater convention
        (E_F = 0). If left unset (``None``, the default), the value
        is read from the .win file's ``fermi_energy`` keyword
        automatically when ``win_path`` is supplied. Pass an explicit
        float to override (including ``0.0`` to disable shifting
        entirely).
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
        if not os.path.isfile(path):
            raise FileNotFoundError(f"hr file does not exist: {path!r}")
        if path.lower().endswith((".hdf5", ".h5")):
            hr_model = tbmodels.Model.from_hdf5_file(path)
        else:
            # Wannier90 *_hr.dat — tbmodels exposes this via
            # `from_wannier_files`. Pass the .win file alongside (when
            # available) so positions get assigned to nearest atoms;
            # otherwise tbmodels requires a `*_centres.xyz` that
            # customers rarely keep around.
            load_kwargs: Dict[str, object] = {"hr_file": path}
            if win_path is not None and os.path.isfile(win_path):
                load_kwargs["win_file"] = win_path
                load_kwargs["pos_kind"] = "nearest_atom"
            hr_model = tbmodels.Model.from_wannier_files(**load_kwargs)
    else:
        hr_model = hr_path_or_model

    # Resolve the per-atom active orbital layout.
    #   - Explicit `active_orbitals` always wins.
    #   - Otherwise, try the .win projection block first.
    #   - If that disagrees with the hr-file's actual orbital count
    #     (the typical "the user kept the API-style .win in the
    #     directory but Wannierized with a restricted projection"
    #     case), silently fall back to inferring from the hr-file's
    #     own position groupings.
    if active_orbitals is None:
        if win_path is not None:
            if not os.path.isfile(win_path):
                raise FileNotFoundError(f"win_path does not point to a file: {win_path!r}")
            try:
                win_orbitals = active_orbitals_from_win(win_path)
            except Exception as exc:
                print(f"[prepare_finetune_target] .win projection block could "
                      f"not be parsed ({type(exc).__name__}: {exc}); "
                      f"falling back to hr-file topology.")
                win_orbitals = None
            if win_orbitals is not None:
                expected = sum(2 * len(o) for o in win_orbitals)
                if expected == int(hr_model.size):
                    active_orbitals = win_orbitals
                else:
                    tag = (f" [{name}]" if name else "")
                    print(f"[prepare_finetune_target]{tag} .win projection "
                          f"implies {expected} compact orbitals but the hr-file "
                          f"has {int(hr_model.size)}. Falling back to per-atom "
                          f"shell inference from the hr-file's position "
                          f"groupings — typical when the directory's .win is "
                          f"the API-side full-projection file but the hr was "
                          f"Wannierized with a restricted projection.")
        if active_orbitals is None:
            # Either no .win path supplied, the parse failed, or the
            # .win disagreed with the hr — derive from hr topology.
            active_orbitals = infer_active_orbitals_from_hr(hr_model)

    # Resolve fermi_shift: explicit user value > .win `fermi_energy` keyword > 0.
    if fermi_shift is None:
        if win_path is not None:
            ef_win = parse_win_fermi_energy(win_path)
            if ef_win is not None:
                fermi_shift = float(ef_win)
                tag = (f" [{name}]" if name else "")
                print(f"[prepare_finetune_target]{tag} using fermi_energy = "
                      f"{fermi_shift:+.6f} eV from {os.path.basename(win_path)} "
                      f"(subtracted from on-sites to set E_F = 0)")
            else:
                fermi_shift = 0.0
        else:
            fermi_shift = 0.0

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
    # The user's own .win (typically named `wannier90.win`) is the
    # source of truth for the target's projection block. The API
    # writes its canonical .win as `input.win` next to the embedding,
    # so when both are present we want the user's. Glob patterns are
    # tried in order — exact name `wannier90.win` first, then any .win.
    win_patterns:   Sequence[str] = ("wannier90.win", "*.win"),
    hr_patterns:    Sequence[str] = ("*_hr.dat", "*_hr.hdf5", "*_hr.h5"),
    out_dir:        Optional[str] = None,
    fermi_shift:    Optional[float] = None,
    strict:         bool  = False,
    sort_names:     bool  = True,
    generate_embedding: bool = False,
    user:           Optional[str] = None,
    password:       Optional[str] = None,
    embedding_filename: str = "embeddings.pt",
    force_regenerate: bool = False,
    api_url:        Optional[str] = None,
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
    fermi_shift : float, optional
        Forwarded to :func:`prepare_finetune_target` for every material.
        If left unset (``None``, the default), each material's
        ``fermi_energy`` keyword in its own .win file is used —
        every Hamiltonian gets shifted to its own ``E_F = 0`` zero,
        which is the convention training data follows in
        ``CleanedDataset.ipynb``. Pass a single float to apply the
        same shift to every material (overriding the per-.win
        lookup), or ``0.0`` to disable shifting entirely.
    strict : bool, default False
        If True, raise on the first subdirectory that's missing any of
        the three required files. If False (default), skip that
        subdirectory with a warning and continue.
    sort_names : bool, default True
        Process subdirectories in sorted-by-name order for reproducible
        training-set ordering.
    generate_embedding : bool, default False
        If True, the function makes one API call per subdirectory
        (using :func:`tw_api_call` with ``return_embeddings=True``) to
        generate any missing embedding ``.pt`` file. The Structure
        passed to the API is reconstructed from the subdirectory's
        own .win file via :func:`structure_from_win`, so the user only
        needs to drop the (``.win``, hr-file) pair into each
        subdirectory — the embedding is generated, saved as
        ``{subdir}/{embedding_filename}``, and then used as the target
        item like any other.
    user, password : str, optional
        HTTP Basic auth credentials for the API. Required when
        ``generate_embedding=True``; ignored otherwise.
    embedding_filename : str, default ``"embeddings.pt"``
        Filename used when saving a generated embedding inside each
        subdirectory.
    force_regenerate : bool, default False
        When ``generate_embedding=True``, controls whether to re-call
        the API for subdirectories that already have an embedding file
        (matched by ``embed_patterns``). Default ``False`` is
        idempotent: existing embeddings are reused, only missing ones
        are generated. Set to ``True`` to overwrite every embedding.
    api_url : str, optional
        Forwarded to :func:`tw_api_call` when ``generate_embedding=True``.
        Leave unset to hit the production endpoint
        (``https://api.tailwater.io``).

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

    # ------------------------------------------------------------------
    # Optional: generate missing embeddings by calling the API
    # ------------------------------------------------------------------
    if generate_embedding:
        if not user or not password:
            raise ValueError(
                "generate_embedding=True requires `user` and `password` for "
                "API authentication. Either pass both or set generate_embedding=False."
            )
        # Lazy import — keeps the directory walker importable on hosts
        # without pymatgen/requests installed for users who just want to
        # use cached embeddings.
        from .client import tw_api_call

        print(f"[generate_embedding] scanning {len(subdirs)} subdirectories "
              f"for materials that need an embedding from the API ...")
        n_generated = 0
        n_skipped   = 0
        for sd in subdirs:
            name     = os.path.basename(sd)
            existing = _first_glob_match(sd, embed_patterns)
            win      = _first_glob_match(sd, win_patterns)
            if not force_regenerate and existing is not None:
                n_skipped += 1
                print(f"  [skip] {name}: embedding already present "
                      f"({os.path.basename(existing)}) — use force_regenerate=True to override")
                continue
            if win is None:
                msg = (f"{name}: cannot generate embedding without a .win file "
                       f"(in {sd!r})")
                if strict:
                    raise FileNotFoundError(msg)
                print(f"  [skip] {msg}")
                continue
            try:
                structure = structure_from_win(win)
            except Exception as exc:
                if strict:
                    raise
                print(f"  [skip] {name}: structure_from_win failed: "
                      f"{type(exc).__name__}: {exc}")
                continue
            embed_stem = os.path.splitext(embedding_filename)[0]
            api_kwargs = dict(
                structure         = structure,
                user              = user,
                password          = password,
                output_path       = sd,
                filename          = embed_stem,
                return_embeddings = True,
                save_cif          = False,
            )
            if api_url is not None:
                api_kwargs["api_url"] = api_url
            print(f"  [api]  {name}: calling /upload_structure_and_download_embeddings/ ...")
            try:
                response = tw_api_call(**api_kwargs)
            except Exception as exc:
                if strict:
                    raise
                print(f"  [fail] {name}: API call failed — {type(exc).__name__}: {exc}")
                continue
            saved_path = response.get("embeddings")
            print(f"  [done] {name}: saved embedding → "
                  f"{os.path.relpath(saved_path, sd) if saved_path else '?'}")
            n_generated += 1
        print(f"[generate_embedding] generated {n_generated} new embedding(s), "
              f"reused {n_skipped} existing.")

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

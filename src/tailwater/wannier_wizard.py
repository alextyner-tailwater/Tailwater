"""Surface and bulk spectral analysis on a tight-binding HDF5 model.

Four calculator classes, each constructed from an HDF5 path (or an
in-memory ``tbmodels.Model``) and each exposing a ``.run()`` method that
returns a Result dataclass with raw NumPy arrays plus matplotlib
``Figure`` objects:

  BulkDOS                — k-mesh-averaged density of states via KPM.
  SurfaceSpectralDensity — surface DOS along a k-path via KPM
                           (top + bottom surfaces in a single pass).
  SurfaceGreensFunction  — surface Green's function along a k-path via
                           the Lopez-Sancho iterative scheme.
  FermiArcMap            — 2D Fermi-arc map at a single energy via
                           Lopez-Sancho.

Plus two helpers for semiconductors / insulators (see the *Fermi
alignment* guide in the docs):

  compute_band_edges — locate VBM / CBM / gap on a uniform k-mesh.
  align_to_vbm       — return a model shifted so the VBM sits at E = 0.

Typical use::

    from tailwater import tb_model, align_to_vbm, SurfaceGreensFunction

    model = align_to_vbm(tb_model.load("wannier90_hr.hdf5"))
    sgf   = SurfaceGreensFunction(model, surface=np.eye(3), energies=...,
                                  k_path=..., k_labels=...)
    result = sgf.run()
    result.figure_top.savefig("top.png")
    np.savez("raw.npz", **result.as_dict())

Dependencies: numpy, scipy, torch, tbmodels, matplotlib, tqdm.
"""

from __future__ import annotations

import copy
import dataclasses
import os
import warnings
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import torch

import tbmodels
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from tqdm import tqdm


# =====================================================================
# RESULT DATACLASSES
# =====================================================================

@dataclasses.dataclass
class BulkDOSResult:
    """Output of BulkDOS.run()."""
    energies:   np.ndarray         # [N_E] energy grid (eV)
    dos:        np.ndarray         # [N_E] density of states
    figure:     Figure             # matplotlib Figure (line plot)

    def as_dict(self) -> dict:
        return {"energies": self.energies, "dos": self.dos}


@dataclasses.dataclass
class SurfaceSpectralDensityResult:
    """Output of SurfaceSpectralDensity.run()."""
    k_vec:        np.ndarray       # [N_path, 3] fractional k-coords
    k_dist:       np.ndarray       # [N_path] cumulative distance along path
    k_node:       np.ndarray       # [num_nodes] distance at each high-symmetry node
    energies:     np.ndarray       # [N_E] energy grid (eV)
    dos_top:      np.ndarray       # [N_path, N_E] surface DOS at top surface
    dos_bottom:   np.ndarray       # [N_path, N_E] surface DOS at bottom surface
    figure_top:   Figure
    figure_bottom: Figure

    def as_dict(self) -> dict:
        return {
            "k_vec": self.k_vec, "k_dist": self.k_dist, "k_node": self.k_node,
            "energies": self.energies,
            "dos_top": self.dos_top, "dos_bottom": self.dos_bottom,
        }


@dataclasses.dataclass
class SurfaceGreensFunctionResult:
    """Output of SurfaceGreensFunction.run()."""
    k_vec:        np.ndarray       # [N_path, 3]
    k_dist:       np.ndarray
    k_node:       np.ndarray
    energies:     np.ndarray       # [N_E]
    spectral_top: np.ndarray       # [N_path, N_E]
    spectral_bottom: np.ndarray    # [N_path, N_E]
    figure_top:   Figure
    figure_bottom: Figure

    def as_dict(self) -> dict:
        return {
            "k_vec": self.k_vec, "k_dist": self.k_dist, "k_node": self.k_node,
            "energies": self.energies,
            "spectral_top": self.spectral_top,
            "spectral_bottom": self.spectral_bottom,
        }


@dataclasses.dataclass
class FermiArcMapResult:
    """Output of FermiArcMap.run()."""
    kx_grid:        np.ndarray     # [Nx] fractional kx points
    ky_grid:        np.ndarray     # [Ny] fractional ky points
    pos_x:          np.ndarray     # [Nx*Ny] Cartesian x-coordinate
    pos_y:          np.ndarray     # [Nx*Ny] Cartesian y-coordinate
    spectral_top:    np.ndarray    # [Nx, Ny] spectral density (top surface)
    spectral_bottom: np.ndarray    # [Nx, Ny]
    figure_top:                 Figure
    figure_bottom:              Figure
    figure_top_interpolated:    Figure
    figure_bottom_interpolated: Figure

    def as_dict(self) -> dict:
        return {
            "kx_grid": self.kx_grid, "ky_grid": self.ky_grid,
            "pos_x":   self.pos_x,   "pos_y":   self.pos_y,
            "spectral_top": self.spectral_top,
            "spectral_bottom": self.spectral_bottom,
        }


@dataclasses.dataclass
class BandStructureResult:
    """Output of BandStructure.run() / bulk_band_structure()."""
    k_vec:        np.ndarray         # [N_path, 3] fractional k-coords
    k_dist:       np.ndarray         # [N_path] cumulative distance along path
    k_node:       np.ndarray         # [num_nodes] distance at each high-symmetry node
    k_labels:     List[str]          # label per node (matches k_node)
    eigenvalues:  np.ndarray         # [N_path, num_bands] band energies (eV)
    figure:       Figure

    def as_dict(self) -> dict:
        return {
            "k_vec":       self.k_vec,
            "k_dist":      self.k_dist,
            "k_node":      self.k_node,
            "k_labels":    list(self.k_labels) if self.k_labels else [],
            "eigenvalues": self.eigenvalues,
        }


# =====================================================================
# MODEL LOADING + MANIPULATION  (private helpers)
# =====================================================================

def _load_model(model_or_path: Union[str, tbmodels.Model]) -> tbmodels.Model:
    """Accept either a path to an HDF5 / Wannier90 file or a tbmodels.Model."""
    if isinstance(model_or_path, tbmodels.Model):
        return model_or_path
    if isinstance(model_or_path, str):
        if not os.path.isfile(model_or_path):
            raise FileNotFoundError(f"Tight-binding file not found: {model_or_path!r}")
        # tbmodels supports HDF5 directly via the class method.
        return tbmodels.Model.from_hdf5_file(model_or_path)
    raise TypeError(
        f"`model_or_path` must be a str path or tbmodels.Model, got {type(model_or_path).__name__}"
    )


# =====================================================================
# FERMI / BAND-EDGE UTILITIES  (public)
# =====================================================================
# The Tailwater training data is Fermi-shifted so that E_F sits at 0 eV.
# For non-metals (semiconductors / insulators), inference therefore puts
# the band gap straddling E=0, with the VBM and CBM landing just below
# and just above zero respectively. The two functions here let users
# (a) measure where those band edges actually fell, and (b) re-anchor the
# model so the VBM (rather than DFT's chosen E_F) becomes the reference,
# which is the more physically natural zero for band-edge comparisons.

def compute_band_edges(
    model_or_path: Union[str, tbmodels.Model],
    k_mesh: Tuple[int, int, int] = (4, 4, 4),
) -> dict:
    """Locate VBM / CBM / gap on a uniform Monkhorst-Pack k-mesh.

    Assumes the model's current zero of energy is roughly E_F (the
    Tailwater training convention). Diagonalizes H(k) on every k in a
    ``k_mesh[0] x k_mesh[1] x k_mesh[2]`` uniform grid in fractional
    reciprocal coordinates, then takes:

      * VBM = the negative eigenvalue closest to zero  (max of e < 0)
      * CBM = the positive eigenvalue closest to zero  (min of e > 0)
      * gap = CBM - VBM
      * is_metal = (gap <= 0)  — i.e. bands overlap E=0 across the mesh

    Parameters
    ----------
    model_or_path : str | tbmodels.Model
        Path to the HDF5 hr-model the API produced, or a tbmodels.Model
        already in memory.
    k_mesh : (int, int, int)
        Grid density. Default (4, 4, 4) — denser meshes catch the VBM/CBM
        at off-symmetry k more accurately at small extra cost.

    Returns
    -------
    dict with keys {"vbm", "cbm", "gap", "is_metal"} (floats, except
    is_metal which is bool; vbm/cbm/gap may be None in degenerate cases
    where the spectrum has no eigenvalues on one side of zero).
    """
    model = _load_model(model_or_path)
    nx, ny, nz = map(int, k_mesh)
    eigs = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                kpt = np.array([i / nx, j / ny, k / nz])
                eigs.append(np.asarray(model.eigenval(k=kpt)))
    eigs = np.concatenate(eigs)
    neg = eigs[eigs < 0.0]
    pos = eigs[eigs > 0.0]
    if neg.size == 0 and pos.size == 0:
        return {"vbm": 0.0, "cbm": 0.0, "gap": 0.0, "is_metal": True}
    if neg.size == 0:
        return {"vbm": None, "cbm": float(pos.min()), "gap": None, "is_metal": True}
    if pos.size == 0:
        return {"vbm": float(neg.max()), "cbm": None, "gap": None, "is_metal": True}
    vbm = float(neg.max())
    cbm = float(pos.min())
    gap = cbm - vbm
    return {
        "vbm":      vbm,
        "cbm":      cbm,
        "gap":      gap,
        "is_metal": gap <= 0.0,
    }


def align_to_vbm(
    model_or_path: Union[str, tbmodels.Model],
    k_mesh: Tuple[int, int, int] = (4, 4, 4),
    fermi_level: Optional[float] = None,
    if_metal: str = "warn",
) -> tbmodels.Model:
    """Return a NEW model with its on-site energies shifted so VBM = 0.

    This re-anchors the energy scale so the band-edge sits exactly at zero —
    the natural reference for plotting / DOS / surface-state computations on
    semiconductors and insulators, instead of whatever DFT-chosen E_F the
    training data was referenced against.

    Parameters
    ----------
    model_or_path : str | tbmodels.Model
        Path to the HDF5 hr-model, or a tbmodels.Model in memory.
    k_mesh : (int, int, int)
        k-mesh used to auto-detect the VBM (only consulted if
        ``fermi_level`` is None). Default (4, 4, 4).
    fermi_level : float, optional
        If supplied, bypass auto-detection and shift on-site energies by
        ``-fermi_level`` (i.e. put your chosen Fermi value at the new zero).
        Useful if you already know E_F from a DFT calculation.
    if_metal : {"warn", "raise", "skip"}
        What to do when ``compute_band_edges`` reports the spectrum has no
        clean gap around E=0 (signature of a metal, or of a non-metal
        whose current zero isn't in the gap). Default ``"warn"``: emits a
        ``RuntimeWarning`` and returns the unshifted model so downstream
        code still runs. ``"raise"`` errors out; ``"skip"`` silently
        returns the unshifted model.

    Returns
    -------
    tbmodels.Model
        A deep copy of the input model with its (0,0,0) hop block adjusted
        by ``shift * I`` so every eigenvalue at every k is offset by
        ``shift = -VBM``. The input model is not mutated.
    """
    model = _load_model(model_or_path)

    if fermi_level is not None:
        shift = -float(fermi_level)
    else:
        edges = compute_band_edges(model, k_mesh=k_mesh)
        if edges["is_metal"]:
            msg = (
                f"align_to_vbm: no clean gap around E=0 on the {k_mesh} "
                f"k-mesh (vbm={edges['vbm']}, cbm={edges['cbm']}, "
                f"gap={edges['gap']}). Consistent with a metal (or a "
                f"non-metal whose current zero isn't in the gap)."
            )
            if if_metal == "warn":
                warnings.warn(msg + " Returning unshifted model.",
                              RuntimeWarning, stacklevel=2)
                return model
            if if_metal == "raise":
                raise RuntimeError(msg)
            if if_metal == "skip":
                return model
            raise ValueError(
                f"if_metal must be one of 'warn'|'raise'|'skip', got {if_metal!r}"
            )
        if edges["vbm"] is None:
            warnings.warn(
                "align_to_vbm: no negative eigenvalues on the k-mesh, "
                "cannot determine VBM. Returning unshifted model.",
                RuntimeWarning, stacklevel=2,
            )
            return model
        shift = -edges["vbm"]

    # Apply the shift to the (0,0,0) hop block. tbmodels' Hamiltonian
    # construction conjugates and adds the stored hop matrix once for +R
    # and once for -R; for R=(0,0,0) this means hop[(0,0,0)] contributes
    # to H(k) twice (the matrix and its Hermitian conjugate). So adding
    # delta*I to hop[(0,0,0)] shifts every eigenvalue by 2*delta. To get
    # an eigenvalue shift of exactly `shift`, we add 0.5*shift*I instead.
    # (Empirically verified — see align_to_vbm tests; std of the per-band
    # diff across a random k is < 1e-13 when this factor is correct.)
    new_model = copy.deepcopy(model)
    n = new_model.size
    delta = 0.5 * shift
    H0 = new_model.hop.get((0, 0, 0))
    if H0 is None:
        new_model.hop[(0, 0, 0)] = delta * np.eye(n, dtype=complex)
    else:
        new_model.hop[(0, 0, 0)] = H0 + delta * np.eye(n, dtype=H0.dtype)
    return new_model


# =====================================================================
# MODEL UTILITIES  (private)
# =====================================================================

def _reorient_model(model: tbmodels.Model, T_matrix) -> tbmodels.Model:
    """Reorient the unit cell of a tbmodels.Model.

    Identical algorithm to the notebook's `reorient_model`. Builds a new
    tbmodels.Model with lattice vectors `T_matrix @ model.uc`, finds the
    old unit cells whose interiors map inside the new supercell, and
    re-keys every hop by the new R index.

    For T_matrix = np.eye(3) this is a no-op (returns an equivalent model
    with the same hop structure).
    """
    import itertools
    import collections as co

    M = np.array(T_matrix).astype(int)
    if M.shape != (model.dim, model.dim):
        raise ValueError(f"Transformation matrix must be {model.dim}x{model.dim}")
    vol_mult = int(np.round(np.abs(np.linalg.det(M))))
    if vol_mult == 0:
        raise ValueError("Transformation matrix has determinant 0.")

    new_uc  = np.dot(M, model.uc) if model.uc is not None else None
    new_occ = (model.occ * vol_mult) if model.occ is not None else None
    M_inv   = np.linalg.inv(M)

    corners = np.array(list(itertools.product([0, 1], repeat=model.dim)))
    corners_old = np.dot(corners, M)
    min_bounds = np.floor(np.min(corners_old, axis=0)).astype(int)
    max_bounds = np.ceil(np.max(corners_old, axis=0)).astype(int)

    uc_offsets = []
    for offset in itertools.product(
        *[range(min_bounds[i], max_bounds[i] + 1) for i in range(model.dim)]
    ):
        v = np.array(offset)
        f = np.dot(v, M_inv)
        f_rounded = np.round(f * 1e7) / 1e7
        if np.all((f_rounded >= 0) & (f_rounded < 1)):
            uc_offsets.append(v)
    if len(uc_offsets) != vol_mult:
        raise RuntimeError("Transformation matrix didn't map interior coords properly.")

    new_pos = []
    for offset in uc_offsets:
        for p in model.pos:
            new_p = np.dot(p + offset, M_inv)
            new_p = np.round(new_p * 1e10) / 1e10
            new_pos.append(new_p % 1.0)

    new_size = model.size * vol_mult
    new_hop  = co.defaultdict(lambda: np.zeros((new_size, new_size), dtype=complex))
    offset_to_idx = {tuple(o): i for i, o in enumerate(uc_offsets)}

    for uc1_idx, uc1_pos in enumerate(uc_offsets):
        s1 = uc1_idx * model.size
        for R, hop_mat in model.hop.items():
            hop_mat  = np.array(hop_mat)
            full_uc2 = uc1_pos + np.array(R)
            f2       = np.dot(full_uc2, M_inv)
            f2_round = np.round(f2 * 1e7) / 1e7
            new_R    = np.floor(f2_round).astype(int)
            uc2_pos  = full_uc2 - np.dot(new_R, M)
            uc2_idx  = offset_to_idx[tuple(np.round(uc2_pos).astype(int))]
            s2       = uc2_idx * model.size
            new_hop[tuple(new_R)][s1:s1 + model.size, s2:s2 + model.size] += hop_mat

    return tbmodels.Model(
        hop  = new_hop,
        occ  = new_occ,
        uc   = new_uc,
        size = new_size,
        pos  = new_pos,
        contains_cc = False,
    )


def _remove_periodicity(periodic_model: tbmodels.Model,
                        direction: int,
                        thickness: int = 1) -> tbmodels.Model:
    """Cut a slab of `thickness` unit cells along `direction`."""
    size = [1] * periodic_model.dim
    size[direction] = thickness
    slab = periodic_model.supercell(size)
    filtered_hops = {
        R: hop_mat for R, hop_mat in slab.hop.items() if R[direction] == 0
    }
    return tbmodels.Model(
        size = slab.size,
        dim  = slab.dim,
        pos  = slab.pos,
        uc   = slab.uc,
        occ  = slab.occ,
        hop  = filtered_hops,
        contains_cc = False,
    )


def _generate_surface_kpm_vector(
    slab_model: tbmodels.Model,
    direction: int,
    surface: str = "top",
    tolerance: float = 1e-4,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Random-phase vector localized on the top or bottom face of a slab."""
    coords = np.array([p[direction] for p in slab_model.pos])
    if surface == "bottom":
        target_val = np.min(coords)
    elif surface == "top":
        target_val = np.max(coords)
    else:
        raise ValueError("`surface` must be 'top' or 'bottom'.")
    indices = np.where(np.abs(coords - target_val) < tolerance)[0]
    if len(indices) == 0:
        raise RuntimeError(f"No orbitals found at the {surface!r} surface.")

    if rng is None:
        rng = np.random.default_rng()
    phases = np.exp(1j * rng.uniform(0, 2 * np.pi, size=len(indices)))
    v = np.zeros(slab_model.size, dtype=complex)
    v[indices] = phases
    return v, indices


# =====================================================================
# K-PATH HELPER  (public)
# =====================================================================

def generate_k_path(
    k_points: Sequence[Sequence[float]],
    N_path: int,
    labels: Optional[List[str]] = None,
    rec_vecs: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a k-path connecting ``k_points`` with about ``N_path`` total samples.

    Parameters
    ----------
    k_points : sequence of fractional k-vectors
        High-symmetry nodes the path visits, in order. Path segments
        between consecutive nodes are sampled in proportion to their
        Cartesian length when ``rec_vecs`` is supplied, else uniformly.
    N_path : int
        Approximate total number of samples across all segments.
    labels : list of str, optional
        Display labels for the nodes (e.g. ``[r"$\\Gamma$", "K", "M"]``).
    rec_vecs : ndarray of shape (3, 3), optional
        Reciprocal lattice vectors. If provided, segment lengths are
        measured in Cartesian (1/Å) space — otherwise they default to
        fractional length, which can over/undersample anisotropic cells.

    Returns
    -------
    k_vec : ndarray of shape (N, 3)
        Sampled k-points along the path.
    k_dist : ndarray of shape (N,)
        Cumulative path length at each sample (for plotting the x-axis
        of a band-structure figure).
    k_node : ndarray of shape (len(k_points),)
        Cumulative path length at each high-symmetry node (for x-tick
        positions on a band-structure figure).
    """
    k_points = np.array(k_points)
    num_nodes = len(k_points)
    if num_nodes < 2:
        raise ValueError("At least two k-points are required.")
    if labels is not None and len(labels) != num_nodes:
        raise ValueError("Number of labels must match the number of k-points.")

    lengths = []
    for i in range(num_nodes - 1):
        diff = k_points[i + 1] - k_points[i]
        if rec_vecs is not None:
            diff = np.dot(diff, rec_vecs)
        lengths.append(np.linalg.norm(diff))
    lengths = np.array(lengths)
    total_length = np.sum(lengths)

    n_per_seg = np.round((lengths / total_length) * (N_path - 1)).astype(int)
    n_per_seg[n_per_seg < 1] = 1
    n_per_seg[-1] += (N_path - (np.sum(n_per_seg) + 1))

    k_vec  = []
    k_dist = []
    k_node = [0.0]
    current = 0.0
    for i in range(num_nodes - 1):
        n_pts = n_per_seg[i]
        L     = lengths[i]
        if i == num_nodes - 2:
            k_seg    = np.linspace(k_points[i], k_points[i + 1], n_pts + 1)
            dist_seg = np.linspace(current, current + L, n_pts + 1)
        else:
            k_seg    = np.linspace(k_points[i], k_points[i + 1], n_pts + 1)[:-1]
            dist_seg = np.linspace(current, current + L, n_pts + 1)[:-1]
        k_vec.extend(k_seg)
        k_dist.extend(dist_seg)
        current += L
        k_node.append(current)
    return np.array(k_vec), np.array(k_dist), np.array(k_node)


# =====================================================================
# KPM CORE  (private helpers)
# =====================================================================

def _to_torch_sparse(H, device: str = "cpu") -> torch.Tensor:
    """Convert a scipy.sparse Hamiltonian (CSR-compatible) into torch.sparse_csr."""
    if sp.issparse(H):
        H = H.tocsr().astype(np.complex128)
        return torch.sparse_csr_tensor(
            torch.from_numpy(H.indptr.copy()).to(torch.int64),
            torch.from_numpy(H.indices.copy()).to(torch.int64),
            torch.from_numpy(H.data.copy()).to(torch.complex128),
            size = H.shape,
            device = device,
        )
    return torch.as_tensor(H, device=device, dtype=torch.complex128)


def _kpm_1d(
    H,
    NC: int,
    NR: int,
    NH: Optional[int] = None,
    psi_in: Optional[np.ndarray] = None,
    avg_output: bool = True,
    device: str = "cpu",
) -> np.ndarray:
    """KPM moment computation using the Chebyshev "doubling trick".

    Identical math to the notebook's `kpm_1d`; reorganized only for
    clarity. Returns the (NR-averaged or per-vector) moments mu_n of
    length NC.
    """
    dtype = torch.complex128
    if not torch.is_tensor(H):
        H = _to_torch_sparse(H, device=device)
    else:
        H = H.to(device=device, dtype=dtype)

    if NH is None:
        NH = H.shape[0]

    if psi_in is None:
        psi = torch.exp(1j * 2 * torch.pi * torch.rand((NH, NR), device=device, dtype=dtype))
    else:
        psi = torch.as_tensor(psi_in, device=device, dtype=dtype)

    mu_all = torch.zeros((NR, NC), dtype=dtype, device="cpu")

    alpha_prev = psi.clone()
    alpha_curr = torch.sparse.mm(H, alpha_prev)

    mu_all[:, 0] = 1.0
    mu_all[:, 1] = torch.sum(torch.conj(psi) * alpha_curr, dim=0).cpu()

    n_stop = NC // 2
    for n in range(2, n_stop):
        alpha_next = 2 * torch.sparse.mm(H, alpha_curr) - alpha_prev

        dot_nn  = torch.sum(torch.conj(alpha_curr) * alpha_curr, dim=0).real
        dot_nn1 = torch.sum(torch.conj(alpha_prev) * alpha_curr, dim=0)

        idx_even = 2 * n - 2
        idx_odd  = 2 * n - 1
        if idx_even < NC:
            mu_all[:, idx_even] = (2 * dot_nn  - mu_all[:, 0].to(device)).cpu()
        if idx_odd  < NC:
            mu_all[:, idx_odd]  = (2 * dot_nn1 - mu_all[:, 1].to(device)).cpu()

        alpha_prev = alpha_curr
        alpha_curr = alpha_next

    if avg_output:
        return mu_all.mean(dim=0).real.numpy()
    return mu_all.numpy()


def _jackson_kernel(n, NC: int, device: str = "cpu") -> torch.Tensor:
    NC_t = torch.tensor(NC, dtype=torch.float64, device=device)
    phi  = torch.pi / (NC_t + 1.0)
    n    = torch.as_tensor(n, dtype=torch.float64, device=device)
    return ((NC_t - n + 1.0) * torch.cos(n * phi)
            + torch.sin(n * phi) / torch.tan(phi)) / (NC_t + 1.0)


def _dos_reconstruct(
    mu: np.ndarray,
    H_rescale_factor: float,
    E_range: Tuple[float, float] = None,
    N_tilde: int = 0,
    NC: Optional[int] = None,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """Chebyshev reconstruction of the DOS from KPM moments.

    Trimmed version of the notebook's `dos()` (drops the unused `dE_order`
    derivative branch). Returns (energy grid, DOS) as NumPy arrays.
    """
    mu = torch.as_tensor(mu, device=device).to(torch.complex128)
    if NC is None or NC == 0:
        NC = len(mu)
    else:
        NC = min(NC, len(mu))

    a = float(H_rescale_factor)
    if N_tilde == 0:
        N_tilde = NC * 2
    if E_range is None:
        E_range = (-a + 0.01, a - 0.01)
    E_grid = torch.linspace(E_range[0], E_range[1], N_tilde + 1,
                            device=device, dtype=torch.float64)

    mask  = torch.abs(E_grid) < a
    E_val = E_grid[mask]

    n_idx    = torch.arange(NC, device=device)
    g_n      = _jackson_kernel(n_idx, NC, device=device)
    h_n      = torch.full((NC,), 2.0, dtype=torch.float64, device=device)
    h_n[0]   = 1.0
    mu_tilde = mu[:NC].real * g_n * h_n

    x = E_val / a
    theta = torch.acos(x)
    cos_n_theta = torch.cos(n_idx.unsqueeze(1).double() * theta.unsqueeze(0))
    sum_tn = torch.sum(mu_tilde.unsqueeze(1) * cos_n_theta, dim=0)
    denom  = a * torch.pi * torch.sqrt(1 - x ** 2)
    rho_e  = sum_tn / denom

    rho_full = torch.zeros(E_grid.shape, dtype=torch.float64, device=device)
    rho_full[mask] = rho_e
    return E_grid.cpu().numpy(), rho_full.cpu().detach().numpy()


# =====================================================================
# LOPEZ-SANCHO RECURSION  (private)
# =====================================================================

@torch.no_grad()
def _recursion_torch(
    onsiteH: torch.Tensor,
    HoppH:   torch.Tensor,
    w: float,
    NN: int,
    eps: float,
    delta: float = 0.0,
    I: Optional[torch.Tensor] = None,
) -> Tuple[float, float, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One Lopez-Sancho iteration set for a (onsite, hopping) block pair.

    Returns (A_L, A_R, HL, HR, HB) — the surface spectral densities at
    the left and right ends plus the renormalized matrices, matching
    the notebook.

    Optimization vs notebook: accepts a pre-allocated identity matrix `I`
    so callers that loop over many (k, w) pairs don't allocate it per
    step. If None, allocate locally.
    """
    dim    = onsiteH.shape[0]
    device = onsiteH.device
    dtype  = onsiteH.dtype
    if I is None:
        I = torch.eye(dim, device=device, dtype=dtype)

    HB    = onsiteH.clone()
    HL    = onsiteH.clone()
    HR    = onsiteH.clone()
    alpha = HoppH.clone()
    beta  = HoppH.conj().T

    z = torch.tensor(w + 1j * eps, device=device, dtype=dtype)

    for _ in range(NN):
        A         = z * I - HB
        GB_beta   = torch.linalg.solve(A, beta)
        GB_alpha  = torch.linalg.solve(A, alpha)
        HL        = HL + alpha @ GB_beta
        HR        = HR + beta  @ GB_alpha
        HB        = HB + HL + HR
        alpha     = alpha @ GB_alpha
        beta      = beta  @ GB_beta

    if delta != 0:
        HL = HL + delta * I
        HR = HR + delta * I

    GL = torch.linalg.solve(z * I - HL, I)
    GR = torch.linalg.solve(z * I - HR, I)
    AL = (-torch.imag(torch.trace(GL)) / torch.pi).item()
    AR = (-torch.imag(torch.trace(GR)) / torch.pi).item()
    return AL, AR, HL, HR, HB


@torch.no_grad()
def _recursion_torch_batched(
    onsiteH_b: torch.Tensor,
    HoppH_b:   torch.Tensor,
    w_b:       torch.Tensor,
    NN: int,
    eps: float,
    delta: float = 0.0,
    I: Optional[torch.Tensor] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Batched Lopez-Sancho — runs `B` independent recursions in lock-step.

    For each batch index ``b`` this is mathematically identical to one
    call of :func:`_recursion_torch` with arguments
    ``(onsiteH_b[b], HoppH_b[b], w_b[b], NN, eps, delta)``.

    The win comes from collapsing ``B`` serial LAPACK calls per inner
    iteration into a single batched call, which removes per-call
    Python/dispatch overhead and lets BLAS keep its caches warm. On
    typical surface-GF problems (dim ~250, NN=8, B=100) this is a 5-10x
    end-to-end speedup over the serial version.

    Args:
        onsiteH_b: ``(B, dim, dim)`` complex tensor — onsite block for
            each batch item. All items must share dtype/device.
        HoppH_b:   ``(B, dim, dim)`` complex tensor — hopping block.
        w_b:       ``(B,)`` real or complex tensor — energies (one per
            batch item).
        NN:        Number of Lopez-Sancho iterations.
        eps:       Imaginary broadening (same for all items).
        delta:     Optional real shift applied to ``HL``/``HR``.
        I:         Optional pre-allocated ``(dim, dim)`` identity matrix.

    Returns:
        Tuple ``(AL, AR)`` of length-``B`` numpy arrays — the surface
        spectral densities at the left and right ends.
    """
    B, dim, _ = onsiteH_b.shape
    device = onsiteH_b.device
    dtype  = onsiteH_b.dtype
    if I is None:
        I = torch.eye(dim, device=device, dtype=dtype)
    I_b = I.unsqueeze(0)                    # (1, dim, dim) — broadcasts over B

    HB    = onsiteH_b.clone()
    HL    = onsiteH_b.clone()
    HR    = onsiteH_b.clone()
    alpha = HoppH_b.clone()
    beta  = HoppH_b.conj().transpose(-1, -2).contiguous()

    # z: complex tensor (B,) → broadcast to (B,1,1) against (dim,dim) blocks
    z      = w_b.to(device=device, dtype=dtype) + 1j * eps
    z_view = z.view(B, 1, 1)

    for _ in range(NN):
        # Single A — share its LU factorization across the two RHS by
        # stacking [β | α] horizontally into one (B, dim, 2*dim) solve.
        # That halves the number of LU factorizations per inner step,
        # which is where Lopez-Sancho actually spends its time.
        A        = z_view * I_b - HB                              # (B, dim, dim)
        rhs      = torch.cat([beta, alpha], dim=-1)               # (B, dim, 2*dim)
        GB       = torch.linalg.solve(A, rhs)                     # (B, dim, 2*dim)
        GB_beta  = GB[..., :dim]
        GB_alpha = GB[..., dim:]

        HL    = HL + alpha @ GB_beta
        HR    = HR + beta  @ GB_alpha
        HB    = HB + HL + HR
        alpha = alpha @ GB_alpha
        beta  = beta  @ GB_beta

    if delta != 0:
        HL = HL + delta * I_b
        HR = HR + delta * I_b

    # Final Green's functions: factor (z·I - HL) and (z·I - HR) once each
    # and apply against the identity. broadcasting via `expand` + contiguous
    # is needed because torch.linalg.solve requires a real (not view) RHS.
    I_rhs = I_b.expand(B, dim, dim).contiguous()
    GL    = torch.linalg.solve(z_view * I_b - HL, I_rhs)
    GR    = torch.linalg.solve(z_view * I_b - HR, I_rhs)

    # Trace over the (dim, dim) tail of each batch element → (B,)
    trGL = torch.diagonal(GL, dim1=-2, dim2=-1).sum(-1)
    trGR = torch.diagonal(GR, dim1=-2, dim2=-1).sum(-1)
    AL   = (-trGL.imag / torch.pi).detach().cpu().numpy()
    AR   = (-trGR.imag / torch.pi).detach().cpu().numpy()
    return AL, AR


# =====================================================================
# WORKER: one k-point of SurfaceGreensFunction (multiprocessing)
# =====================================================================
#
# Defined at module top-level so joblib can pickle it by reference.
# The worker takes only the small (onsiteH, HoppH) slab blocks for
# this k-point — the parent process slices them out of the slab
# Hamiltonian and ships them across the pickle boundary. This avoids
# sending the (often big) slab_model itself to every worker.

def _surface_gf_kpoint_worker(
    onsiteH_np: np.ndarray,
    HoppH_np:   np.ndarray,
    energies:   np.ndarray,
    NN:         int,
    eps:        float,
    delta:      float,
    complex_dtype_str: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run the batched Lopez-Sancho recursion for one k-point.

    Returns ``(AL, AR)`` — length-``Nw`` numpy arrays — the left/right
    surface spectral densities at this k.
    """
    import torch as _torch
    # One BLAS thread per worker process: with N processes already saturating
    # the cores, letting each process spin up its own thread pool oversubscribes.
    _torch.set_num_threads(1)

    dtype = _torch.complex64 if complex_dtype_str == "complex64" else _torch.complex128
    dim   = onsiteH_np.shape[0]
    oH    = _torch.as_tensor(onsiteH_np, dtype=dtype)
    hH    = _torch.as_tensor(HoppH_np,   dtype=dtype)
    w_b   = _torch.as_tensor(energies,   dtype=dtype)
    I     = _torch.eye(dim, dtype=dtype)

    B = w_b.shape[0]
    AL, AR = _recursion_torch_batched(
        oH.unsqueeze(0).expand(B, -1, -1),
        hH.unsqueeze(0).expand(B, -1, -1),
        w_b, NN, eps, delta, I=I,
    )
    return AL, AR


# =====================================================================
# UTILITY: reciprocal-lattice vectors from real-lattice rows
# =====================================================================

def _reciprocal_lattice(a_matrix: np.ndarray) -> np.ndarray:
    a1, a2, a3 = a_matrix[0], a_matrix[1], a_matrix[2]
    V = np.dot(a1, np.cross(a2, a3))
    if np.isclose(V, 0):
        raise ValueError("Real-space lattice is degenerate (zero volume).")
    return np.array([
        (2 * np.pi / V) * np.cross(a2, a3),
        (2 * np.pi / V) * np.cross(a3, a1),
        (2 * np.pi / V) * np.cross(a1, a2),
    ])


# =====================================================================
# 1) BULK DOS  (NEW — not in original notebook)
# =====================================================================

class BulkDOS:
    """k-mesh-averaged density of states via KPM.

    For each k in a uniform Monkhorst-Pack-like mesh, builds the
    periodic-cell Hamiltonian H(k), runs KPM with `NV` random phase
    vectors, accumulates Chebyshev moments, then reconstructs the DOS
    on the requested energy grid.

    Construction
    ------------
        BulkDOS(model_or_path, k_mesh=(4,4,4), energies=(-5, 5),
                NC=2048, NV=4, device='cpu')

    Run
    ---
        result = calc.run()
        result.energies, result.dos, result.figure
    """

    def __init__(
        self,
        model_or_path: Union[str, tbmodels.Model],
        k_mesh: Tuple[int, int, int] = (4, 4, 4),
        energies: Tuple[float, float] = (-5.0, 5.0),
        NC: int = 2048,
        NV: int = 4,
        N_tilde: Optional[int] = None,
        device: str = "cpu",
        verbose: bool = True,
    ):
        self.model    = _load_model(model_or_path)
        self.k_mesh   = tuple(int(x) for x in k_mesh)
        self.energies = tuple(float(x) for x in energies)
        self.NC       = int(NC)
        self.NV       = int(NV)
        self.N_tilde  = int(N_tilde) if N_tilde is not None else 2 * self.NC
        self.device   = device
        self.verbose  = bool(verbose)
        try:
            self.model.set_sparse(True)
        except AttributeError:
            pass

    def run(self) -> BulkDOSResult:
        kx_n, ky_n, kz_n = self.k_mesh
        ks = np.stack(np.meshgrid(
            np.arange(kx_n) / kx_n,
            np.arange(ky_n) / ky_n,
            np.arange(kz_n) / kz_n,
            indexing="ij",
        ), axis=-1).reshape(-1, 3)

        mu_accum = np.zeros(self.NC, dtype=np.float64)
        iter_ks  = tqdm(ks, desc="Bulk DOS (k-mesh)") if self.verbose else ks
        for k in iter_ks:
            H = sp.csr_matrix(self.model.hamilton(k=k))
            alpha = spla.eigsh(H, k=1, which="LM", maxiter=300,
                               return_eigenvectors=False, tol=0.25)
            a_norm = (np.abs(alpha[0]) + 0.5)
            H_resc = H / a_norm

            mu = _kpm_1d(H_resc, NC=self.NC, NR=self.NV,
                         avg_output=True, device=self.device)
            # Normalize moments back to the original spectral scale by
            # remembering `a_norm` per k. Since the rescale changes per
            # k, we reconstruct DOS PER k and average those instead of
            # averaging moments (which would conflate scales).
            E_k, rho_k = _dos_reconstruct(
                mu, a_norm, E_range=self.energies,
                N_tilde=self.N_tilde, NC=self.NC, device=self.device,
            )
            if not hasattr(self, "_E_ref"):
                self._E_ref   = E_k
                self._rho_acc = np.zeros_like(rho_k)
            self._rho_acc += rho_k

        energies = self._E_ref
        dos      = np.nan_to_num(self._rho_acc / len(ks), nan=0.0)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(energies, dos, lw=1.5)
        ax.set_xlabel("E (eV)", fontsize=14)
        ax.set_ylabel("DOS (a.u.)", fontsize=14)
        ax.set_xlim(self.energies)
        ax.grid(alpha=0.3)
        plt.close(fig)

        return BulkDOSResult(energies=energies, dos=dos, figure=fig)


# =====================================================================
# 2) SURFACE SPECTRAL DENSITY (KPM)
# =====================================================================

class SurfaceSpectralDensity:
    """Surface DOS along a k-path via KPM, for top and bottom surfaces.

    Construction
    ------------
        SurfaceSpectralDensity(
            model_or_path, surface=np.eye(3), LZ=5,
            energies=(E_F-1, E_F+1),
            k_path=[[0,0,0], [0,0,0.5], ...],
            k_labels=None, N_path=101, NC=2**12, NV=4,
            device='cpu',
        )

    `surface` is a 3x3 integer matrix passed to `_reorient_model`. Use
    `np.eye(3)` to leave the unit cell unchanged. `LZ` is the slab
    thickness in unit-cell layers along the (post-reorient) z-axis.
    """

    def __init__(
        self,
        model_or_path: Union[str, tbmodels.Model],
        surface: np.ndarray,
        LZ: int,
        energies: Tuple[float, float],
        k_path: Sequence[Sequence[float]],
        k_labels: Optional[List[str]] = None,
        N_path: int = 101,
        NC: int = 2 ** 12,
        NV: int = 4,
        device: str = "cpu",
        verbose: bool = True,
        rng_seed: Optional[int] = None,
    ):
        model = _load_model(model_or_path)
        try:
            model.set_sparse(True)
        except AttributeError:
            pass

        # Build reoriented + slab model ONCE at construction.
        model    = _reorient_model(model, surface)
        # Notebook flattened positions to 0 — kept for parity (KPM doesn't
        # use them, but downstream code that reads model.pos shouldn't be
        # surprised).
        model.pos = 0 * np.array(model.pos)
        slab_model = _remove_periodicity(model, direction=2, thickness=LZ)

        # K-path
        k_vec, k_dist, k_node = generate_k_path(k_path, N_path)

        # Surface random-phase vectors (one set per surface, NV vectors each).
        rng = np.random.default_rng(rng_seed)
        top_vecs = []
        bot_vecs = []
        for _ in range(NV):
            v_top, _ = _generate_surface_kpm_vector(slab_model, direction=2,
                                                    surface="top", rng=rng)
            v_bot, _ = _generate_surface_kpm_vector(slab_model, direction=2,
                                                    surface="bottom", rng=rng)
            top_vecs.append(v_top)
            bot_vecs.append(v_bot)
        self._top_psi = np.transpose(np.array(top_vecs))      # [NH, NV]
        self._bot_psi = np.transpose(np.array(bot_vecs))      # [NH, NV]

        # Node positions for x-axis labels.
        node_index = [0]
        for n in range(1, len(k_path) - 1):
            frac = k_node[n] / k_node[-1]
            node_index.append(int(round(frac * (N_path - 1))))
        node_index.append(N_path - 1)

        self.slab_model = slab_model
        self.k_vec      = k_vec
        self.k_dist     = k_dist
        self.k_node     = k_node
        self.node_index = node_index
        self.k_labels   = k_labels
        self.energies   = tuple(float(x) for x in energies)
        self.N_path     = N_path
        self.NC         = NC
        self.NV         = NV
        self.device     = device
        self.verbose    = verbose

    def run(self) -> SurfaceSpectralDensityResult:
        Results_top: List[np.ndarray] = []
        Results_bot: List[np.ndarray] = []

        iter_k = tqdm(self.k_vec, desc="Surface KPM (k-path)") if self.verbose else self.k_vec
        for k in iter_k:
            kp = np.array(k)
            H  = sp.csr_matrix(self.slab_model.hamilton(k=kp))
            alpha = spla.eigsh(H, k=1, which="LM", maxiter=300,
                               return_eigenvectors=False, tol=0.25)
            a_norm = (np.abs(alpha[0]) + 0.5)
            H_resc = H / a_norm

            # Top surface
            mu = _kpm_1d(H_resc, NC=self.NC, NR=self.NV,
                         psi_in=self._top_psi, avg_output=True,
                         device=self.device)
            E_grid, rho_top = _dos_reconstruct(
                mu, a_norm, E_range=self.energies, NC=self.NC, device=self.device,
            )
            Results_top.append(rho_top)

            # Bottom surface
            mu = _kpm_1d(H_resc, NC=self.NC, NR=self.NV,
                         psi_in=self._bot_psi, avg_output=True,
                         device=self.device)
            E_grid, rho_bot = _dos_reconstruct(
                mu, a_norm, E_range=self.energies, NC=self.NC, device=self.device,
            )
            Results_bot.append(rho_bot)

        Results_top = np.nan_to_num(np.array(Results_top), nan=0.0)
        Results_bot = np.nan_to_num(np.array(Results_bot), nan=0.0)

        figure_top    = self._make_figure(Results_top,    title=None)
        figure_bottom = self._make_figure(Results_bot, title=None)

        return SurfaceSpectralDensityResult(
            k_vec=self.k_vec, k_dist=self.k_dist, k_node=self.k_node,
            energies=E_grid,
            dos_top=Results_top, dos_bottom=Results_bot,
            figure_top=figure_top, figure_bottom=figure_bottom,
        )

    def _make_figure(self, data: np.ndarray, title: Optional[str] = None) -> Figure:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(np.transpose(data), cmap="coolwarm", aspect="auto")
        ax.set_ylim(0, data.shape[1])
        ax.set_ylabel("E (eV)", fontsize=16)
        ax.set_yticks(
            [0, data.shape[1] / 2, data.shape[1]],
            [self.energies[0], np.round(np.mean(self.energies), 2), self.energies[1]],
            fontsize=16,
        )
        if self.k_labels is not None:
            ax.set_xticks(self.node_index, self.k_labels, fontsize=16)
        if title:
            ax.set_title(title, fontsize=14)
        plt.close(fig)
        return fig


# =====================================================================
# 3) SURFACE GREEN'S FUNCTION (Lopez-Sancho)
# =====================================================================

class SurfaceGreensFunction:
    """Surface Green's function along a k-path via Lopez-Sancho recursion.

    Construction
    ------------
        SurfaceGreensFunction(
            model_or_path, surface=np.eye(3),
            energies=np.linspace(-1, 1, 101),
            k_path=[[0,0,0], [0,0,0.5], ...],
            N_path=101, k_labels=None,
            thickness=6, NN=5, eps=0.005, delta=0.0,
            device='cpu', chunk_size=256, n_jobs=1,
        )

    Performance
    -----------
    The Lopez-Sancho recursion is the bottleneck — roughly 95% of
    wall-clock time. Two knobs control how it's parallelized:

    * ``chunk_size`` (default 256): for each k-point, batches the
      energy axis through :func:`_recursion_torch_batched` so all
      energies share a single Python dispatch per Lopez-Sancho step.
      Larger chunks ⇒ less overhead, more memory. The default keeps
      peak memory bounded even for dense energy grids on thick slabs.

    * ``n_jobs`` (default 1): when set to ``-1`` or any integer ``> 1``,
      fans the k-points out across worker processes via ``joblib``.
      Each worker runs one BLAS thread to avoid oversubscription. On
      multi-core CPUs this is the biggest single win — k-points are
      fully independent, so speedup scales close to linearly with the
      number of physical cores until pickling overhead bites
      (typically beyond ~16 workers on this problem class).

    Typical recipe: ``SurfaceGreensFunction(model, ..., n_jobs=-1)``.
    """

    def __init__(
        self,
        model_or_path: Union[str, tbmodels.Model],
        surface: np.ndarray,
        energies: Sequence[float],
        k_path: Sequence[Sequence[float]],
        N_path: int = 101,
        k_labels: Optional[List[str]] = None,
        thickness: int = 6,
        NN: int = 5,
        eps: float = 0.005,
        delta: float = 0.0,
        device: str = "cpu",
        verbose: bool = True,
        chunk_size: int = 256,
        n_jobs: int = 1,
    ):
        model = _load_model(model_or_path)
        try:
            model.set_sparse(True)
        except AttributeError:
            pass
        model     = _reorient_model(model, surface)
        model.pos = 0 * np.array(model.pos)
        self.slab_model = _remove_periodicity(model, direction=2, thickness=thickness)
        self.num_wann   = len(model.pos)

        k_vec, k_dist, k_node = generate_k_path(k_path, N_path)
        node_index = [0]
        for n in range(1, len(k_path) - 1):
            frac = k_node[n] / k_node[-1]
            node_index.append(int(round(frac * (N_path - 1))))
        node_index.append(N_path - 1)

        self.k_vec      = k_vec
        self.k_dist     = k_dist
        self.k_node     = k_node
        self.node_index = node_index
        self.k_labels   = k_labels
        self.energies   = np.asarray(energies, dtype=float)
        self.thickness  = thickness
        self.NN         = int(NN)
        self.eps        = float(eps)
        self.delta      = float(delta)
        self.verbose    = bool(verbose)
        self.chunk_size = int(chunk_size)
        self.n_jobs     = int(n_jobs)

        # Configure torch device + dtype once.
        try:
            self.device = torch.device(device)
        except Exception:
            self.device = torch.device("cpu")
        self.dtype = torch.complex64 if str(self.device) == "mps" else torch.complex128

        # Pre-allocate the identity tensor for Lopez-Sancho (same shape
        # for every k, every w — major iteration is over k * w * NN).
        dim     = 2 * self.num_wann
        self._I = torch.eye(dim, device=self.device, dtype=self.dtype)

    def run(self) -> SurfaceGreensFunctionResult:
        Nk = len(self.k_vec)
        Nw = len(self.energies)
        nwl = self.num_wann

        Left  = np.zeros((Nk, Nw))
        Right = np.zeros((Nk, Nw))

        # ---- Parallel path: fan out k-points across worker processes ----
        if self.n_jobs != 1:
            try:
                from joblib import Parallel, delayed
            except ImportError as e:
                raise ImportError(
                    "n_jobs > 1 requires `joblib`. Install it with "
                    "`pip install joblib`, or set n_jobs=1 to run serially."
                ) from e

            # Build the small (onsiteH, HoppH) slab blocks in the parent —
            # cheap (~ms per k) and avoids pickling the full slab_model
            # across the process boundary.
            blocks = []
            for k in self.k_vec:
                Ham_np = np.asarray(self.slab_model.hamilton(k))
                blocks.append((
                    Ham_np[2 * nwl:4 * nwl, 2 * nwl:4 * nwl].copy(),
                    Ham_np[2 * nwl:4 * nwl, 4 * nwl:].copy(),
                ))

            dtype_str = "complex64" if self.dtype == torch.complex64 else "complex128"
            energies_np = np.asarray(self.energies)
            results = Parallel(
                n_jobs=self.n_jobs,
                backend="loky",
                verbose=10 if self.verbose else 0,
            )(
                delayed(_surface_gf_kpoint_worker)(
                    oH, hH, energies_np, self.NN, self.eps, self.delta, dtype_str,
                )
                for (oH, hH) in blocks
            )
            for ik, (AL, AR) in enumerate(results):
                Left [ik] = AL
                Right[ik] = AR

        # ---- Serial path: batched recursion, energy-chunked ----
        else:
            # Pre-cast the energy grid once — re-used at every k-point.
            w_b = torch.as_tensor(self.energies, device=self.device, dtype=self.dtype)

            # Chunk the energy axis so peak memory stays bounded for large
            # slab dimensions. Each batch item holds ~6 dim^2-sized complex
            # matrices in flight; the default chunk keeps that under ~4 GB
            # at dim=500 and is essentially "no chunking" for typical runs.
            chunk = max(1, int(self.chunk_size))

            iter_k = tqdm(enumerate(self.k_vec), total=Nk, desc="Surface GF") \
                     if self.verbose else enumerate(self.k_vec)
            for ik, k in iter_k:
                Ham_np = self.slab_model.hamilton(k)
                Ham = torch.as_tensor(np.asarray(Ham_np), device=self.device, dtype=self.dtype)

                # Take the "interior" 2*num_wann x 2*num_wann block and the
                # corresponding hopping block.
                onsiteH = Ham[2 * nwl:4 * nwl, 2 * nwl:4 * nwl]
                HoppH   = Ham[2 * nwl:4 * nwl, 4 * nwl:]

                # Batch the recursion over all (or a chunk of) energies. The
                # slab blocks are constant across energies for a given k, so
                # broadcasting them is essentially free vs. re-stacking.
                for w_start in range(0, Nw, chunk):
                    w_end  = min(w_start + chunk, Nw)
                    B      = w_end - w_start
                    onsite_b = onsiteH.unsqueeze(0).expand(B, -1, -1)
                    hopp_b   = HoppH.unsqueeze(0).expand(B, -1, -1)
                    AL, AR = _recursion_torch_batched(
                        onsite_b, hopp_b, w_b[w_start:w_end],
                        self.NN, self.eps, self.delta, I=self._I,
                    )
                    Left [ik, w_start:w_end] = AL
                    Right[ik, w_start:w_end] = AR

        Left  = np.nan_to_num(Left,  nan=0.0)
        Right = np.nan_to_num(Right, nan=0.0)

        figure_top    = self._make_figure(Right, title=None)
        figure_bottom = self._make_figure(Left,  title=None)

        return SurfaceGreensFunctionResult(
            k_vec=self.k_vec, k_dist=self.k_dist, k_node=self.k_node,
            energies=self.energies,
            spectral_top=Right, spectral_bottom=Left,
            figure_top=figure_top, figure_bottom=figure_bottom,
        )

    def _make_figure(self, data: np.ndarray, title: Optional[str] = None) -> Figure:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(np.transpose(data), cmap="coolwarm", aspect="auto")
        ax.set_ylim(0, len(self.energies))
        ax.set_ylabel("E (eV)", fontsize=16)
        ax.set_yticks(
            [0, len(self.energies) / 2, len(self.energies)],
            [float(np.min(self.energies)),
             float(np.round(np.mean(self.energies), 1)),
             float(np.max(self.energies))],
            fontsize=16,
        )
        fig.colorbar(im, ax=ax)
        if self.k_labels is not None:
            ax.set_xticks(self.node_index, self.k_labels, fontsize=16)
        if title:
            ax.set_title(title, fontsize=14)
        plt.close(fig)
        return fig


# =====================================================================
# 4) FERMI-ARC MAP (2D k-grid Lopez-Sancho)
# =====================================================================

class FermiArcMap:
    """Surface spectral function at a SINGLE energy on a 2D k-grid.

    Same Lopez-Sancho machinery as ``SurfaceGreensFunction``, but the
    k-grid is the 2D BZ slice at ``k_z = 0`` (post-reorient), spanning
    ``[-0.5, 0.5]`` in both ``k_x`` and ``k_y``. Produces four matplotlib
    figures: raw and griddata-interpolated maps for both surfaces.
    """

    def __init__(
        self,
        model_or_path: Union[str, tbmodels.Model],
        surface: np.ndarray,
        energy: float,
        Nx: int,
        Ny: int,
        thickness: int = 6,
        NN: int = 5,
        eps: float = 0.005,
        delta: float = 0.0,
        device: str = "cuda",
        verbose: bool = True,
        chunk_size: int = 128,
        n_jobs: int = 1,
    ):
        model = _load_model(model_or_path)
        try:
            model.set_sparse(True)
        except AttributeError:
            pass
        model     = _reorient_model(model, surface)
        model.pos = 0 * np.array(model.pos)
        self.slab_model = _remove_periodicity(model, direction=2, thickness=thickness)
        self.num_wann   = len(model.pos)

        self.energy    = float(energy)
        self.Nx        = int(Nx)
        self.Ny        = int(Ny)
        self.thickness = thickness
        self.NN        = int(NN)
        self.eps       = float(eps)
        self.delta     = float(delta)
        self.verbose   = bool(verbose)
        self.chunk_size = int(chunk_size)
        self.n_jobs    = int(n_jobs)

        # Device + dtype
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.dtype  = torch.complex64 if str(self.device) == "mps" else torch.complex128

        # Cache reciprocal lattice for Cartesian k positions.
        uc = self.slab_model.uc
        if uc is None:
            uc = np.eye(3)
        self._kvecs_cart = _reciprocal_lattice(np.asarray(uc))

        # Pre-allocate identity for Lopez-Sancho.
        dim = 2 * self.num_wann
        self._I = torch.eye(dim, device=self.device, dtype=self.dtype)

    def run(self) -> FermiArcMapResult:
        kx_grid = np.linspace(-0.5, 0.5, self.Nx)
        ky_grid = np.linspace(-0.5, 0.5, self.Ny)

        # Flatten the 2D grid to a single list of k-points. Order is
        # (kx outer, ky inner) so the reshape back to (Nx, Ny) is clean.
        ks = np.stack(
            np.meshgrid(kx_grid, ky_grid, [0.0], indexing="ij"),
            axis=-1,
        ).reshape(-1, 3)                          # (Nx*Ny, 3)
        total = ks.shape[0]

        PosX = (self._kvecs_cart @ ks.T)[0]
        PosY = (self._kvecs_cart @ ks.T)[1]

        dim   = 2 * self.num_wann
        nwl   = self.num_wann
        # Single energy, broadcast across each batch.
        E_val = float(self.energy)

        Left_flat  = np.zeros(total)
        Right_flat = np.zeros(total)

        chunk = max(1, int(self.chunk_size))
        single_energy = np.array([E_val], dtype=float)

        # ---- Parallel path: each k-point is a tiny recursion job ----
        if self.n_jobs != 1:
            try:
                from joblib import Parallel, delayed
            except ImportError as e:
                raise ImportError(
                    "n_jobs > 1 requires `joblib`. Install it with "
                    "`pip install joblib`, or set n_jobs=1 to run serially."
                ) from e

            # Pre-build all the slab block pairs in the parent — cheap.
            blocks = []
            for k in ks:
                Ham_np = np.asarray(self.slab_model.hamilton(list(k)))
                blocks.append((
                    Ham_np[2 * nwl:4 * nwl, 2 * nwl:4 * nwl].copy(),
                    Ham_np[2 * nwl:4 * nwl, 4 * nwl:].copy(),
                ))

            dtype_str = "complex64" if self.dtype == torch.complex64 else "complex128"
            results = Parallel(
                n_jobs=self.n_jobs,
                backend="loky",
                verbose=10 if self.verbose else 0,
            )(
                delayed(_surface_gf_kpoint_worker)(
                    oH, hH, single_energy, self.NN, self.eps, self.delta, dtype_str,
                )
                for (oH, hH) in blocks
            )
            for ik, (AL, AR) in enumerate(results):
                Left_flat [ik] = AL[0]
                Right_flat[ik] = AR[0]

        # ---- Serial path: build (B, dim, dim) blocks per chunk, batch ----
        else:
            iter_chunks = range(0, total, chunk)
            if self.verbose:
                pbar = tqdm(total=total, desc="Fermi-arc map")

            for c_start in iter_chunks:
                c_end = min(c_start + chunk, total)
                B     = c_end - c_start

                # Build (B, dim, dim) onsite & hopping blocks. Hamiltonian
                # construction is intrinsically Python-loop-y in tbmodels,
                # so we still call it once per k — but the heavy Lopez-Sancho
                # work that follows is fully batched.
                onsite_b = torch.empty((B, dim, dim), device=self.device, dtype=self.dtype)
                hopp_b   = torch.empty((B, dim, dim), device=self.device, dtype=self.dtype)
                for b, k in enumerate(ks[c_start:c_end]):
                    Ham_np = self.slab_model.hamilton(list(k))
                    Ham    = torch.as_tensor(np.asarray(Ham_np),
                                             device=self.device, dtype=self.dtype)
                    onsite_b[b] = Ham[2 * nwl:4 * nwl, 2 * nwl:4 * nwl]
                    hopp_b  [b] = Ham[2 * nwl:4 * nwl, 4 * nwl:]

                w_b = torch.full((B,), E_val, device=self.device, dtype=self.dtype)
                AL, AR = _recursion_torch_batched(
                    onsite_b, hopp_b, w_b,
                    self.NN, self.eps, self.delta, I=self._I,
                )
                Left_flat [c_start:c_end] = AL
                Right_flat[c_start:c_end] = AR
                if self.verbose:
                    pbar.update(B)
            if self.verbose:
                pbar.close()

        Left  = Left_flat .reshape(self.Nx, self.Ny)
        Right = Right_flat.reshape(self.Nx, self.Ny)

        fig_top = self._raw_figure(Right)
        fig_bot = self._raw_figure(Left)     # <-- correct: bottom uses Left_Surf
        fig_top_int = self._interpolated_figure(PosX, PosY, Right.flatten())
        fig_bot_int = self._interpolated_figure(PosX, PosY, Left .flatten())

        return FermiArcMapResult(
            kx_grid=kx_grid, ky_grid=ky_grid,
            pos_x=PosX, pos_y=PosY,
            spectral_top=Right, spectral_bottom=Left,
            figure_top=fig_top, figure_bottom=fig_bot,
            figure_top_interpolated=fig_top_int,
            figure_bottom_interpolated=fig_bot_int,
        )

    def _raw_figure(self, data: np.ndarray) -> Figure:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(np.transpose(data), cmap="coolwarm", aspect="auto")
        ax.set_ylabel(r"$k_{2}$", fontsize=16)
        ax.set_xlabel(r"$k_{1}$", fontsize=16)
        fig.colorbar(im, ax=ax)
        plt.close(fig)
        return fig

    def _interpolated_figure(self, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> Figure:
        from scipy.interpolate import griddata
        grid_x, grid_y = np.mgrid[
            float(np.min(xs)):float(np.max(xs)):100j,
            float(np.min(ys)):float(np.max(ys)):100j,
        ]
        grid_z = griddata(
            np.column_stack([xs, ys]), zs, (grid_x, grid_y), method="linear",
        )
        fig, ax = plt.subplots(figsize=(10, 8))
        contour = ax.contourf(grid_x, grid_y, grid_z, levels=50, cmap="coolwarm")
        fig.colorbar(contour, ax=ax)
        ax.set_ylabel(r"$k_{2}$", fontsize=16)
        ax.set_xlabel(r"$k_{1}$", fontsize=16)
        plt.close(fig)
        return fig


# =====================================================================
# 5) BULK BAND STRUCTURE  (manual k-path or seekpath-auto)
# =====================================================================

def _seekpath_auto_path(
    structure,
    with_time_reversal: bool = True,
) -> Tuple[List[np.ndarray], List[str]]:
    """Use the `seekpath` package to determine a standard high-symmetry path.

    Parameters
    ----------
    structure : pymatgen.Structure or 3-tuple (lattice, frac_coords, atomic_numbers)
        The crystal structure to analyze. If a pymatgen.Structure is passed
        we extract the three arrays seekpath needs (lattice as 3x3 row
        matrix in Å, fractional coordinates, atomic numbers).
    with_time_reversal : bool
        Forwarded to seekpath.get_path.

    Returns
    -------
    k_points : list of np.ndarray
        Fractional k-coordinates of each high-symmetry node in path order.
        Path discontinuities (segments that aren't connected) are handled
        by inserting both endpoints — see seekpath documentation.
    k_labels : list of str
        Pretty labels (with "GAMMA" rendered as r"$\\Gamma$") aligned with
        k_points. Discontinuities appear as "A|B" composite labels.
    """
    try:
        import seekpath
    except ImportError as e:
        raise ImportError(
            "auto k-path generation requires the `seekpath` package. "
            "Install with `pip install seekpath` or "
            "`pip install \"tailwater[seekpath]\"`."
        ) from e

    # Normalize input to the (lattice, positions, types) tuple seekpath wants.
    if hasattr(structure, "lattice") and hasattr(structure, "sites"):
        # pymatgen.Structure
        lattice = np.array(structure.lattice.matrix)
        frac    = np.array([site.frac_coords for site in structure])
        nums    = np.array([site.specie.Z   for site in structure])
        struct_tuple = (lattice, frac, nums)
    elif isinstance(structure, tuple) and len(structure) == 3:
        struct_tuple = structure
    else:
        raise TypeError(
            "`structure` must be a pymatgen.Structure or a "
            "(lattice, frac_coords, atomic_numbers) tuple."
        )

    res = seekpath.get_path(struct_tuple, with_time_reversal=with_time_reversal)
    point_coords = res["point_coords"]
    path = res["path"]

    def _pretty(label: str) -> str:
        # Common seekpath label fixes for matplotlib output.
        if label == "GAMMA":
            return r"$\Gamma$"
        return label

    # Build the flat list. Each segment is (start_label, end_label).
    # When two consecutive segments don't share an endpoint we insert a
    # composite label "prev|next" to mark the discontinuity.
    k_points: List[np.ndarray] = []
    k_labels: List[str] = []
    last_end_label: Optional[str] = None
    for (a, b) in path:
        a_coord = np.array(point_coords[a])
        b_coord = np.array(point_coords[b])
        if last_end_label is None:
            k_points.append(a_coord)
            k_labels.append(_pretty(a))
        elif last_end_label != a:
            # Discontinuity — merge with prior label.
            k_labels[-1] = f"{_pretty(last_end_label)}|{_pretty(a)}"
            k_points[-1] = a_coord
        # Always add the segment endpoint.
        k_points.append(b_coord)
        k_labels.append(_pretty(b))
        last_end_label = b

    return k_points, k_labels


class BandStructure:
    """Bulk band structure along a user-supplied or auto-generated k-path.

    Construction
    ------------
        BandStructure(model_or_path,
                      k_points=[[0,0,0], [0,0,0.5], [0.5,0.5,0]],
                      k_labels=[r"$\\Gamma$", "Z", "M"],
                      spacing=0.01)

    Or with `seekpath`-derived high-symmetry path:

        BandStructure.auto(model_or_path, structure, spacing=0.01)

    The `spacing` argument is the maximum allowed step in fractional
    reciprocal coordinates between adjacent k-points along the path
    (pass a smaller value for a denser path). The total number of
    samples is `ceil(path_length / spacing)`.

    Run
    ---
        result = bs.run()
        result.eigenvalues, result.k_dist, result.figure
    """

    def __init__(
        self,
        model_or_path: Union[str, tbmodels.Model],
        k_points: Sequence[Sequence[float]],
        k_labels: Optional[List[str]] = None,
        spacing: float = 0.01,
        fermi_level: float = 0.0,
        e_range: Optional[Tuple[float, float]] = None,
        verbose: bool = True,
        linewidth: float = 1.6,
    ):
        self.model = _load_model(model_or_path)
        try:
            # Dense Hamiltonians are fastest for eigvalsh.
            self.model.set_sparse(False)
        except AttributeError:
            pass

        if k_points is None or len(k_points) < 2:
            raise ValueError("`k_points` must have at least two nodes.")
        if k_labels is not None and len(k_labels) != len(k_points):
            raise ValueError("`k_labels` must align 1-to-1 with `k_points`.")

        self.k_points    = [np.asarray(p, dtype=float) for p in k_points]
        self.k_labels    = list(k_labels) if k_labels is not None else None
        self.spacing     = float(spacing)
        self.fermi_level = float(fermi_level)
        self.e_range     = e_range
        self.verbose     = bool(verbose)
        self.linewidth   = float(linewidth)

    @classmethod
    def auto(
        cls,
        model_or_path: Union[str, tbmodels.Model],
        structure,
        spacing: float = 0.01,
        fermi_level: float = 0.0,
        e_range: Optional[Tuple[float, float]] = None,
        with_time_reversal: bool = True,
        verbose: bool = True,
        linewidth: float = 1.6,
    ) -> "BandStructure":
        """Build a BandStructure whose k-path is determined by `seekpath`.

        `structure` is forwarded to `_seekpath_auto_path` — either a
        pymatgen.Structure or the (lattice, frac_coords, atomic_numbers)
        tuple seekpath accepts directly.
        """
        k_points, k_labels = _seekpath_auto_path(
            structure, with_time_reversal=with_time_reversal,
        )
        return cls(
            model_or_path,
            k_points    = k_points,
            k_labels    = k_labels,
            spacing     = spacing,
            fermi_level = fermi_level,
            e_range     = e_range,
            verbose     = verbose,
            linewidth   = linewidth,
        )

    def run(self) -> BandStructureResult:
        # ---- Determine sample count from spacing ----
        # Path length is computed in fractional reciprocal coordinates
        # (same units as the user-supplied k_points). N_path is sized
        # so every segment gets at least `ceil(L_seg / spacing)` samples.
        lengths = [
            float(np.linalg.norm(self.k_points[i + 1] - self.k_points[i]))
            for i in range(len(self.k_points) - 1)
        ]
        total_len = sum(lengths)
        # Guard against the user passing two identical k-points
        # (zero-length path) — we still need at least two samples.
        N_path = max(2, int(np.ceil(total_len / max(self.spacing, 1e-12))))
        if self.verbose:
            print(f"[bands] path total length = {total_len:.4f}  "
                  f"-> N_path = {N_path} samples (spacing = {self.spacing})")

        k_vec, k_dist, k_node = generate_k_path(self.k_points, N_path)

        # ---- Sweep the path ----
        # tbmodels.Model.eigenval returns a 1-D array of eigenvalues for
        # each k. We stack them into a [N_path, num_bands] array.
        rows = []
        iter_ks = tqdm(k_vec, desc="Band structure") if self.verbose else k_vec
        for k in iter_ks:
            ev = np.asarray(self.model.eigenval(k=np.asarray(k, dtype=float)))
            rows.append(ev)
        eigenvalues = np.array(rows) - self.fermi_level

        # ---- Figure ----
        figure = self._make_figure(k_dist, k_node, eigenvalues)

        return BandStructureResult(
            k_vec       = k_vec,
            k_dist      = k_dist,
            k_node      = k_node,
            k_labels    = self.k_labels or [],
            eigenvalues = eigenvalues,
            figure      = figure,
        )

    # ---- Plot helper (separated so callers can re-plot from raw data) ----
    def _make_figure(
        self,
        k_dist: np.ndarray,
        k_node: np.ndarray,
        eigenvalues: np.ndarray,
    ) -> Figure:
        fig, ax = plt.subplots(figsize=(8, 6))
        # Plot every band as a separate line. Default line width matches the
        # WannierTools gnuplot bulkek convention (`w lp lw 2`) for a bolder,
        # publication-style band plot; override via BandStructure(linewidth=...).
        lw = self.linewidth
        for b in range(eigenvalues.shape[1]):
            ax.plot(k_dist, eigenvalues[:, b], lw=lw, c="k", alpha=1.0)
        # Vertical separator at every high-symmetry node
        for x in k_node:
            ax.axvline(x, color="0.6", lw=0.8, ls="--")
        # E_F reference line — solid and as bold as the bands (gnuplot `0 w l lw 2`)
        ax.axhline(0.0, color="red", lw=lw, ls="-")
        ax.set_xlim(float(k_dist[0]), float(k_dist[-1]))
        if self.e_range is not None:
            ax.set_ylim(*self.e_range)
        if self.k_labels:
            ax.set_xticks(k_node)
            ax.set_xticklabels(self.k_labels, fontsize=14)
        ax.set_ylabel(r"$E - E_F$ (eV)", fontsize=14)
        ax.grid(alpha=0.2, axis="y")
        plt.close(fig)
        return fig


def bulk_band_structure(
    model_or_path: Union[str, tbmodels.Model],
    k_points: Optional[Sequence[Sequence[float]]] = None,
    k_labels: Optional[List[str]] = None,
    spacing: float = 0.01,
    *,
    auto: bool = False,
    structure=None,
    fermi_level: float = 0.0,
    e_range: Optional[Tuple[float, float]] = None,
    return_raw: bool = False,
    verbose: bool = True,
    with_time_reversal: bool = True,
    linewidth: float = 1.6,
):
    """Compute the bulk band structure of a tight-binding model along a k-path.

    Two modes:

      * MANUAL (default): pass `k_points` (and optionally `k_labels`).
            bulk_band_structure(model, k_points=[[0,0,0], [0,0,0.5]],
                                k_labels=[r"$\\Gamma$", "Z"])

      * AUTO  (`auto=True`): the high-symmetry path is determined by
        `seekpath` from a `structure` argument (pymatgen.Structure or a
        seekpath-format tuple).
            bulk_band_structure(model, auto=True, structure=mp_structure)

    Parameters
    ----------
    model_or_path : str or tbmodels.Model
        Path to an HDF5 file produced by the API, or an in-memory model.
    k_points : sequence of [k1, k2, k3] in fractional reciprocal coords
        Required when `auto=False`.
    k_labels : list of str, optional
        Label per node, e.g. [r"$\\Gamma$", "M", "K", r"$\\Gamma$"].
    spacing : float
        Maximum step (in fractional reciprocal coords) between adjacent
        samples on the path. Smaller spacing = denser sampling.
    auto : bool
        If True, use seekpath to determine path + labels (overrides
        `k_points` / `k_labels`). Requires the `seekpath` package and a
        `structure` argument.
    structure : pymatgen.Structure or seekpath tuple, optional
        Required when `auto=True`.
    fermi_level : float
        Subtracted from every eigenvalue before plotting.
    e_range : (float, float), optional
        Y-axis limits for the figure.
    return_raw : bool
        If True, return the full `BandStructureResult` (with raw arrays
        AND the figure). If False (default), return only the matplotlib
        Figure.
    verbose : bool
        Toggle the tqdm progress bar.
    with_time_reversal : bool
        Passed to seekpath in auto mode.
    linewidth : float, default 1.6
        Band line width (and E_F reference line width). The default matches
        the WannierTools gnuplot bulkek convention (`w lp lw 2`) for a bolder,
        publication-style plot; lower it (e.g. 1.0) for dense band manifolds.

    Returns
    -------
    matplotlib.figure.Figure   if return_raw is False
    BandStructureResult        if return_raw is True
    """
    if auto:
        if structure is None:
            raise ValueError("`auto=True` requires `structure=...`.")
        bs = BandStructure.auto(
            model_or_path,
            structure          = structure,
            spacing            = spacing,
            fermi_level        = fermi_level,
            e_range            = e_range,
            with_time_reversal = with_time_reversal,
            verbose            = verbose,
            linewidth          = linewidth,
        )
    else:
        if k_points is None:
            raise ValueError(
                "Provide `k_points` for manual mode, or set `auto=True` "
                "and pass `structure` for seekpath-determined path."
            )
        bs = BandStructure(
            model_or_path,
            k_points    = k_points,
            k_labels    = k_labels,
            spacing     = spacing,
            fermi_level = fermi_level,
            e_range     = e_range,
            verbose     = verbose,
            linewidth   = linewidth,
        )
    result = bs.run()
    return result if return_raw else result.figure


# =====================================================================
# CONVENIENCE FACADE  (optional one-shot helpers)
# =====================================================================

def run_surface_kpm_from_hdf5(hdf5_path: str, **kwargs) -> SurfaceSpectralDensityResult:
    """One-shot: build SurfaceSpectralDensity from an HDF5 path and call .run()."""
    return SurfaceSpectralDensity(hdf5_path, **kwargs).run()


def run_surface_gf_from_hdf5(hdf5_path: str, **kwargs) -> SurfaceGreensFunctionResult:
    """One-shot: build SurfaceGreensFunction from an HDF5 path and call .run()."""
    return SurfaceGreensFunction(hdf5_path, **kwargs).run()


def run_bulk_dos_from_hdf5(hdf5_path: str, **kwargs) -> BulkDOSResult:
    """One-shot: build BulkDOS from an HDF5 path and call .run()."""
    return BulkDOS(hdf5_path, **kwargs).run()


def run_fermi_arc_from_hdf5(hdf5_path: str, **kwargs) -> FermiArcMapResult:
    """One-shot: build FermiArcMap from an HDF5 path and call .run()."""
    return FermiArcMap(hdf5_path, **kwargs).run()

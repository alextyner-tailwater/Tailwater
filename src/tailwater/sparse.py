"""Sparse Wannier Hamiltonian container — the client-side companion to the
API's optimized (sparse) inference backend.

The API's ``/upload_json_process_and_download_dat/`` endpoint can return the
Hamiltonian as a sparse ``wannier90_hr.npz`` (COO hops + on-site diagonal +
optional geometry) instead of a dense tbmodels HDF5. That format is O(N) in
RAM/egress instead of O(N**2), which is what makes large systems tractable.

:class:`SparseHR` loads that ``.npz`` and converts it into whatever the caller
wants — scipy H(k), a tbmodels.Model, a Wannier90 ``_hr.dat``, or a pybinding /
Kwant model — WITHOUT ever materialising a dense ``[num_wann, num_wann]`` block
for the pybinding / Kwant / ``hr_dict`` paths (those build straight from the COO,
so they scale to large ``num_wann``).

Method names are aligned with the dense (tbmodels) path in :mod:`tailwater`:
``to_pb`` / ``to_kwant`` / ``to_pythtb`` (via :mod:`tailwater.convert`) /
``to_hr_dat`` / ``to_hdf5``, so the same call works whether you are holding a
:class:`SparseHR` or a ``tbmodels.Model``. For a format-agnostic entry point
that auto-detects sparse-``.npz`` vs dense-``.hdf5``/``_hr.dat`` inputs, use the
:mod:`tailwater.convert` dispatchers.

``scipy`` / ``numpy`` / ``tbmodels`` are hard dependencies of the package;
``pybinding`` and ``kwant`` are optional and imported lazily only inside the
``to_pb`` / ``to_kwant`` builders.
"""
import warnings

import numpy as np

__all__ = ["SparseHR"]


class SparseHR:
    """Self-contained sparse Wannier H(R): COO hops + on-site + (optional)
    geometry (primitive lattice ``cell`` and per-orbital ``positions``). Carries
    everything needed to reload the inference output straight into scipy /
    tbmodels / pybinding / Kwant — no crystal graph or GNN required.

    Construct one with :meth:`load` (from an API-produced ``.npz``); the
    per-hop arrays are the *forward-only* half of each ``+/-R`` pair (the H.c. is
    implied), with the R=0 diagonal carried separately in ``on_site``.
    """
    __slots__ = ("num_wann", "on_site", "rows", "cols", "Rs", "vals",
                 "cell", "positions")

    def __init__(self, num_wann, on_site, rows, cols, Rs, vals,
                 cell=None, positions=None):
        self.num_wann = int(num_wann)
        self.on_site = np.asarray(on_site, float)
        self.rows = np.asarray(rows, np.int64)
        self.cols = np.asarray(cols, np.int64)
        self.Rs = np.asarray(Rs, np.int64).reshape(-1, 3)
        self.vals = np.asarray(vals, np.complex128)
        self.cell = None if cell is None else np.asarray(cell, float).reshape(3, 3)
        self.positions = None if positions is None else np.asarray(positions, float)

    def __repr__(self):
        geo = "with geometry" if self.cell is not None else "no geometry"
        return (f"SparseHR(num_wann={self.num_wann}, nnz={self.nnz}, "
                f"num_R={len(np.unique(self.Rs, axis=0))}, {geo})")

    @property
    def nnz(self):
        """Number of stored (forward-only) hops."""
        return self.rows.shape[0]

    # ---- persistence: reload the inference output anywhere ----
    def save(self, path):
        """Write the full sparse model (hops + on-site + geometry) to ``.npz``."""
        d = dict(num_wann=self.num_wann, on_site=self.on_site, rows=self.rows,
                 cols=self.cols, Rs=self.Rs, vals=self.vals)
        if self.cell is not None:
            d["cell"] = self.cell
        if self.positions is not None:
            d["positions"] = self.positions
        np.savez_compressed(
            path if str(path).endswith(".npz") else f"{path}.npz", **d)
        return path

    @classmethod
    def load(cls, path):
        """Reload a SparseHR from an API ``wannier90_hr.npz`` (or one written by
        :meth:`save`) — no graph/model needed."""
        z = np.load(path, allow_pickle=False)
        return cls(int(z["num_wann"]), z["on_site"], z["rows"], z["cols"],
                   z["Rs"], z["vals"],
                   cell=z["cell"] if "cell" in z.files else None,
                   positions=z["positions"] if "positions" in z.files else None)

    # ---- spectra (sparse; large num_wann OK) ----
    def Hk(self, k):
        """Sparse H(k) matching ``tbmodels.hamilton`` (convention 2). Each stored
        hop ``(val, i, j, R)`` contributes ``e^{2*pi*i*k.R}*val`` at ``(i, j)``
        and its conjugate at ``(j, i)``; ``on_site`` fills the diagonal.
        Vectorised -> O(nnz)."""
        import scipy.sparse as sp
        ph = np.exp(2j * np.pi * (self.Rs @ np.asarray(k, float)))
        d = ph * self.vals
        nw = self.num_wann
        r = np.concatenate([self.rows, self.cols, np.arange(nw)])
        c = np.concatenate([self.cols, self.rows, np.arange(nw)])
        data = np.concatenate([d, np.conj(d), self.on_site.astype(complex)])
        return sp.coo_matrix((data, (r, c)), shape=(nw, nw)).tocsr()

    def eigvals_grid(self, kpts, dense=True):
        """Dense eigenvalues (sorted ascending) at each fractional k in ``kpts``.
        Returns an array of shape ``[len(kpts), num_wann]``."""
        out = np.empty((len(kpts), self.num_wann))
        for i, k in enumerate(kpts):
            H = self.Hk(k)
            out[i] = np.linalg.eigvalsh(H.toarray() if dense else H.todense())
        return out

    def eigsh_near_fermi(self, k, e_fermi=0.0, num=40):
        """The ``num`` eigenvalues nearest ``e_fermi`` at fractional k, via
        shift-invert sparse diagonalisation — usable for num_wann far beyond what
        a dense H(k) can hold."""
        import scipy.sparse.linalg as spla
        H = self.Hk(k)
        H = 0.5 * (H + H.getH())
        return np.sort(spla.eigsh(H, k=min(num, self.num_wann - 2),
                                  sigma=e_fermi, which="LM",
                                  return_eigenvectors=False).real)

    # ---- interchange / export ----
    def hr_dict(self):
        """Real-space H(R) as ``{R_tuple: scipy.sparse.csr_matrix}`` — the natural
        sparse in-memory form for large systems (feed to your own solver, KPM,
        pybinding, Kwant, ...). O(nnz)."""
        import scipy.sparse as sp
        nw = self.num_wann
        out = {}
        keys = [tuple(int(x) for x in r) for r in self.Rs]
        by_R = {}
        for idx, R in enumerate(keys):
            by_R.setdefault(R, []).append(idx)
        for R, idxs in by_R.items():
            idxs = np.asarray(idxs)
            M = sp.coo_matrix((self.vals[idxs], (self.rows[idxs], self.cols[idxs])),
                              shape=(nw, nw)).tocsr()
            out[R] = M
            mR = tuple(-x for x in R)                      # implied H.c. at -R
            out[mR] = out.get(mR, sp.csr_matrix((nw, nw))) + M.getH()
        out[(0, 0, 0)] = out.get((0, 0, 0), sp.csr_matrix((nw, nw))) \
            + sp.diags(self.on_site, format="csr")
        return out

    def _lattice(self):
        """Primitive lattice vectors for TB-package construction. Uses the stored
        ``cell``; falls back to identity (bands at fractional k are still correct —
        the cell only sets the k_frac<->k_cart map and real-space geometry)."""
        if self.cell is not None:
            return self.cell
        warnings.warn(
            "SparseHR has no stored lattice; using identity. Fractional-k "
            "eigenvalues are still correct, but cartesian-k / real-space "
            "geometry will be nominal.")
        return np.eye(3)

    def to_tbmodels(self, uc=None):
        """Convert to a ``tbmodels.Model`` (dense per-R; small/medium systems).
        ``uc`` defaults to the stored ``cell``. From the returned model use
        ``.to_hr_file()`` (Wannier90 ``_hr.dat``) or ``.to_hdf5_file()``, or the
        tbmodels solvers directly."""
        import tbmodels
        if uc is None:
            uc = self.cell
        m = tbmodels.Model(on_site=[float(x) for x in self.on_site], dim=3, occ=1,
                           pos=[[0.0, 0.0, 0.0]] * self.num_wann, uc=uc)
        for i, j, R, v in zip(self.rows.tolist(), self.cols.tolist(),
                              self.Rs.tolist(), self.vals.tolist()):
            if R == [0, 0, 0] and i == j:
                continue                                  # on_site already set
            m.add_hop(complex(v), int(i), int(j), tuple(R))
        return m

    def to_pb(self):
        """Load directly into a pybinding ``pb.Lattice`` (one sublattice per
        orbital), built straight from the COO — no dense matrix. Mirrors
        ``tb_model.load(hdf5).to_pb()`` for the dense path."""
        pos = self.positions if self.positions is not None \
            else np.zeros((self.num_wann, 3))
        return _sparse_to_pybinding(self, self._lattice(), pos)

    def to_pybinding(self):
        """Deprecated alias for :meth:`to_pb` (kept for older sparse-API code)."""
        warnings.warn("SparseHR.to_pybinding() is deprecated; use .to_pb().",
                      DeprecationWarning, stacklevel=2)
        return self.to_pb()

    def to_kwant(self):
        """Build a bulk Kwant ``(Builder, lattice)`` (single site carrying all
        ``num_wann`` orbitals as matrix-valued blocks), straight from the COO —
        scales to large num_wann. Mirrors ``tb_model.load(hdf5).to_kwant()``."""
        return _sparse_to_kwant(self, self._lattice())

    def to_hr_dat(self, path, uc=None, max_wann=4000):
        """Write a Wannier90 ``_hr.dat`` (via tbmodels). DENSE format: file size
        ~ ``num_R * num_wann**2``, so it is guarded to small/medium systems. For
        large systems keep it sparse (``hr_dict``/``Hk``/KPM). Pass ``max_wann``
        to override the guard."""
        if self.num_wann > max_wann:
            raise ValueError(
                f"hr.dat is a DENSE format (~num_R * num_wann^2). num_wann="
                f"{self.num_wann} would produce a huge file (~"
                f"{17 * self.num_wann ** 2 * 70 / 1e9:.0f} GB). Keep it sparse "
                f"(hr_dict()/Hk()/KPM), or pass max_wann to override.")
        self.to_tbmodels(uc=uc).to_hr_file(str(path))
        return path

    def to_hdf5(self, path, uc=None, max_wann=4000):
        """Write a tbmodels HDF5 (dense per-R). Same dense-format guard as
        :meth:`to_hr_dat`. This is the format the ``tw_api_call`` auto-conversion
        produces for small systems, so downstream code that expects
        ``wannier90_hr.hdf5`` keeps working unchanged."""
        if self.num_wann > max_wann:
            raise ValueError(
                f"tbmodels HDF5 is a DENSE format (~num_R * num_wann^2). "
                f"num_wann={self.num_wann} would produce a huge file. Keep it "
                f"sparse (hr_dict()/Hk()/KPM), or pass max_wann to override.")
        self.to_tbmodels(uc=uc).to_hdf5_file(str(path))
        return path


# ============================================================
# Sparse -> pybinding / Kwant builders (from the COO; no densify)
# ============================================================
def _sparse_to_pybinding(sparse_hr, prim_vecs, orbital_positions):
    """Build a ``pb.Lattice`` (one sublattice per orbital) from a SparseHR."""
    import pybinding as pb
    nw = sparse_hr.num_wann
    lat = pb.Lattice(a1=list(prim_vecs[0]), a2=list(prim_vecs[1]),
                     a3=list(prim_vecs[2]))
    lat.add_sublattices(*[(f"o{i}", list(map(float, orbital_positions[i])),
                           float(sparse_hr.on_site[i])) for i in range(nw)])
    # SparseHR COO is already forward-only (one of each +/-R pair; R=0 diagonal
    # is on_site) — exactly what pybinding wants, since it auto-adds each H.c.
    hops = [([int(R[0]), int(R[1]), int(R[2])], f"o{int(i)}", f"o{int(j)}",
             complex(v)) for i, j, R, v in zip(
                 sparse_hr.rows, sparse_hr.cols, sparse_hr.Rs, sparse_hr.vals)]
    lat.add_hoppings(*hops)
    return lat


def _sparse_to_kwant(sparse_hr, prim_vecs):
    """Build an (un-finalized) bulk ``kwant.Builder`` with a single site carrying
    all num_wann orbitals; onsite = H(0,0,0), hoppings = H(R) matrices.

    For bulk bands wrap it::

        import kwant.wraparound as wa
        fs = wa.wraparound(syst).finalized()
        H  = fs.hamiltonian_submatrix(params=dict(k_x=.., k_y=.., k_z=..))

    where ``k_i = 2*pi * fractional k_i``. For transport, attach leads to a
    finite build. Returns ``(Builder, lattice)``.
    """
    import kwant
    nw = sparse_hr.num_wann
    pv = [tuple(map(float, v)) for v in prim_vecs]
    lat = kwant.lattice.Monatomic(pv, norbs=nw)     # single site, nw orbitals
    syst = kwant.Builder(kwant.TranslationalSymmetry(*pv))
    d = sparse_hr.hr_dict()
    syst[lat(0, 0, 0)] = d[(0, 0, 0)].toarray()
    seen = set()
    for R, M in d.items():
        if R == (0, 0, 0):
            continue
        if R in seen or tuple(-x for x in R) in seen:
            continue
        seen.add(R)
        syst[kwant.builder.HoppingKind(R, lat)] = M.toarray()
    return syst, lat

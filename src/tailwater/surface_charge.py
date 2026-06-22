"""Surface charge-density heat maps of a (hkl) slab from a Wannier Hamiltonian.

The single public entry point is :func:`surface_charge_density`, which takes
ANY Wannier tight-binding model — a Tailwater prediction or a DFT-generated
Wannier90 Hamiltonian — and renders the real-space surface charge density of
a general ``(hkl)`` slab as two heat maps (top view down the surface normal +
a side cross-section), in the style of ELF surface maps (e.g. Adv. Sci.
10.1002/advs.202307192, Fig 2).

Pipeline (all from the model's real-space ``H(R)`` — no gdata needed):
  1. Re-express ``H(R)`` in an integer supercell whose first two lattice
     vectors lie in the ``(hkl)`` plane (exact, det-preserving remap;
     validated to machine precision).
  2. Stack ``size`` supercells along the surface normal and drop hoppings
     that leave the slab.
  3. Integrate ``|ψ|²`` of occupied states over a 2D surface-BZ mesh to get a
     per-orbital occupation, then render ``ρ(r) = Σ_g n_g·Gaussian(r−r_g)``
     using the Wannier centres as ``r_g``.

Model input forms accepted by :func:`surface_charge_density`:
  * a ``tbmodels.Model`` instance,
  * a path to a tbmodels HDF5 (``.hdf5``/``.h5``) hr-model,
  * a path to a Wannier90 ``*_hr.dat`` (DFT output),
  * the internal dict ``{"Hd", "A", "pos", "norb"}`` from :func:`load_hr`.

Note on threading: the per-k ``eigh`` loop can stall under some OpenBLAS
builds. This module limits BLAS threads to 1 during that loop via
``threadpoolctl`` when available (matching the notebook's
``OMP_NUM_THREADS=1`` workaround); install ``threadpoolctl`` for the safe
path, otherwise set ``OMP_NUM_THREADS=1`` in the environment.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np


# =====================================================
# MODEL LOADING / NORMALISATION
# =====================================================
def _hd_from_tbmodels(m) -> dict:
    """tbmodels.Model -> hermitian, +/-R-complete H(R) dict + lattice/positions.

    tbmodels stores only the +R half; we hermitian-complete (H(-R)=H(R)†).
    """
    A   = np.array(m.uc, float)
    pos = np.array(m.pos, float)
    norb = int(m.size)
    half = {tuple(int(x) for x in R): np.array(M, complex) for R, M in m.hop.items()}
    Hd: dict = {}
    for R, M in half.items():
        Hd[R] = Hd.get(R, np.zeros_like(M)) + M
        if R != (0, 0, 0):
            nR = tuple(-x for x in R)
            Hd[nR] = Hd.get(nR, np.zeros_like(M)) + M.conj().T
    return dict(Hd=Hd, A=A, pos=pos, norb=norb)


def load_hr(path: Union[str, Path]) -> dict:
    """Load a Wannier Hamiltonian from disk into the internal model dict.

    ``.hdf5``/``.h5`` -> ``tbmodels.Model.from_hdf5_file``.
    Anything else (e.g. ``wannier90_hr.dat``) -> ``from_wannier_files``.
    """
    import tbmodels
    path = str(path)
    if path.lower().endswith((".hdf5", ".h5")):
        m = tbmodels.Model.from_hdf5_file(path)
    else:
        m = tbmodels.Model.from_wannier_files(hr_file=path)
    return _hd_from_tbmodels(m)


def _as_model(model) -> dict:
    """Normalise any accepted model form to the internal ``{Hd, A, pos, norb}`` dict."""
    if isinstance(model, dict) and "Hd" in model:
        return model
    if isinstance(model, (str, Path)):
        return load_hr(model)
    # Duck-type a tbmodels.Model (avoid importing tbmodels just to isinstance).
    if hasattr(model, "hop") and hasattr(model, "uc") and hasattr(model, "pos"):
        return _hd_from_tbmodels(model)
    raise TypeError(
        "surface_charge_density: `model` must be a tbmodels.Model, a path to a "
        "tbmodels HDF5 / Wannier90 hr.dat, or the internal dict from load_hr(); "
        f"got {type(model).__name__}."
    )


def recip(A: np.ndarray) -> np.ndarray:
    """Reciprocal lattice (rows = b1,b2,b3) for direct lattice A (rows = a1,a2,a3)."""
    return 2 * np.pi * np.linalg.inv(A).T


def H_of_k(model: dict, kfrac: Sequence[float]) -> np.ndarray:
    """Bloch Hamiltonian H(k), convention I, at reduced coordinates ``kfrac``."""
    A = model["A"]
    kc = np.array(kfrac, float) @ recip(A)
    H = sum(M * np.exp(1j * ((np.array(R) @ A) @ kc)) for R, M in model["Hd"].items())
    return 0.5 * (H + H.conj().T)


# =====================================================
# GENERAL (hkl) SUPERCELL + SLAB CUT
# =====================================================
def miller_supercell_matrix(miller: Sequence[int], A: np.ndarray, maxc: int = 4) -> np.ndarray:
    """Integer 3x3 M (rows = new lattice vectors); rows 0,1 in the (hkl) plane,
    row 2 = shortest out-of-plane stacking vector."""
    h, k, l = (int(x) for x in miller)
    g = np.gcd.reduce([abs(h), abs(k), abs(l)]) or 1
    h, k, l = h // g, k // g, l // g
    rng = range(-maxc, maxc + 1)
    inplane = [np.array(c) for c in itertools.product(rng, rng, rng)
               if c != (0, 0, 0) and h * c[0] + k * c[1] + l * c[2] == 0]
    inplane.sort(key=lambda c: np.linalg.norm(c @ A))
    t1 = inplane[0]
    t2 = next(c for c in inplane[1:] if np.linalg.norm(np.cross(t1 @ A, c @ A)) > 1e-6)
    best = None
    for c in itertools.product(rng, rng, rng):
        d = h * c[0] + k * c[1] + l * c[2]
        if d <= 0:
            continue
        key = (d, float(np.linalg.norm(np.array(c) @ A)))
        if best is None or key < best[0]:
            best = (key, np.array(c))
    M = np.array([t1, t2, best[1]])
    if np.linalg.det(M) < 0:
        M = np.array([t2, t1, best[1]])
    return M


def build_supercell(prim: dict, M: np.ndarray) -> dict:
    """Exact re-expression of H(R) in the integer supercell defined by M (rows)."""
    A, pos, norb, Hd = prim["A"], prim["pos"], prim["norb"], prim["Hd"]
    Minv = np.linalg.inv(M)
    ndet = int(round(abs(np.linalg.det(M))))
    bound = int(np.abs(M).sum()) + 1
    reps = [np.array(c) for c in itertools.product(range(-bound, bound + 1), repeat=3)
            if np.all(c @ Minv > -1e-9) and np.all(c @ Minv < 1 - 1e-9)]
    assert len(reps) == ndet, f"{len(reps)} reps but det={ndet}"
    rep_of = {tuple(int(x) for x in r): i for i, r in enumerate(reps)}

    def split(dcell):
        f = dcell @ Minv
        n = np.floor(f + 1e-9).astype(int)
        rem = tuple(int(x) for x in (dcell - n @ M))
        return rep_of[rem], tuple(int(x) for x in n)

    Nnew = ndet * norb
    Hnew: dict = {}
    for R, blk in Hd.items():
        R = np.array(R)
        for ri, rc in enumerate(reps):
            rj, n = split(rc + R)
            B = Hnew.setdefault(n, np.zeros((Nnew, Nnew), complex))
            B[ri * norb:(ri + 1) * norb, rj * norb:(rj + 1) * norb] += blk
    posnew = np.zeros((Nnew, 3))
    for ri, rc in enumerate(reps):
        for o in range(norb):
            f = (rc + pos[o]) @ Minv
            posnew[ri * norb + o] = f - np.floor(f + 1e-9)
    return dict(Hd=Hnew, A=M @ A, pos=posnew, norb=Nnew, M=M, ndet=ndet)


def cut_slab(sup: dict, n_layers: int) -> dict:
    """Stack n_layers supercells along axis 2 (normal); drop hoppings leaving the slab."""
    Hd, norb = sup["Hd"], sup["norb"]
    Ns, Hs = norb * n_layers, {}
    for R, Mblk in Hd.items():
        Rin = (R[0], R[1], 0)
        for L0 in range(n_layers):
            L1 = L0 + R[2]
            if 0 <= L1 < n_layers:
                B = Hs.setdefault(Rin, np.zeros((Ns, Ns), complex))
                B[L0 * norb:(L0 + 1) * norb, L1 * norb:(L1 + 1) * norb] += Mblk
    return dict(Hs=Hs, Ns=Ns, norb=norb, n_layers=n_layers)


def supercell_self_check(model: dict, miller: Sequence[int], ntrials: int = 20) -> float:
    """Max band-edge mismatch (eV) between bulk and supercell at the same physical k.

    A det-preserving re-basis must reproduce the bulk spectrum exactly; this
    returns the worst band-edge discrepancy over ``ntrials`` random k-points
    (expect ~1e-13 eV). Useful as a sanity gate before trusting a slab.
    """
    M = miller_supercell_matrix(miller, model["A"])
    sup = build_supercell(model, M)
    Bp, Bs = recip(model["A"]), recip(sup["A"])
    rng = np.random.default_rng(7)
    maxd = 0.0
    for _ in range(ntrials):
        kc = rng.random(3) @ Bp
        ep = np.sort(np.linalg.eigvalsh(H_of_k(model, np.linalg.solve(Bp.T, kc))))
        es = np.sort(np.linalg.eigvalsh(H_of_k(sup, np.linalg.solve(Bs.T, kc))))
        maxd = max(maxd, abs(ep.min() - es.min()), abs(ep.max() - es.max()))
    return float(maxd)


# =====================================================
# CHARGE DENSITY + RENDERING
# =====================================================
def _orbital_geometry(sup: dict, n_layers: int):
    """Cartesian position, depth-along-normal, and layer index of every slab orbital."""
    A2 = sup["A"]; a3 = A2[2]; norb_c = sup["norb"]
    nhat = np.cross(A2[0], A2[1]); nhat /= np.linalg.norm(nhat)
    Ns = norb_c * n_layers
    pos = np.zeros((Ns, 3)); layer = np.zeros(Ns, int)
    for L in range(n_layers):
        pos[L * norb_c:(L + 1) * norb_c] = (sup["pos"] @ A2) + L * a3
        layer[L * norb_c:(L + 1) * norb_c] = L
    depth = pos @ nhat; depth -= depth.min()
    return pos, depth, layer, nhat, norb_c


def _occupations(slab: dict, A2: np.ndarray, mu: float, nk: int, energy_window):
    """k-averaged occupied-state probability on each slab orbital over an nk×nk SBZ grid."""
    a1, a2 = A2[0], A2[1]
    G = np.array([[a1 @ a1, a1 @ a2], [a2 @ a1, a2 @ a2]]); Gi = np.linalg.inv(G)
    b1 = 2 * np.pi * (Gi[0, 0] * a1 + Gi[0, 1] * a2)
    b2 = 2 * np.pi * (Gi[1, 0] * a1 + Gi[1, 1] * a2)
    Rs = np.array(list(slab["Hs"].keys()))
    Ms = np.array([slab["Hs"][tuple(R)] for R in Rs]); Tcart = Rs @ A2
    rho = np.zeros(slab["Ns"]); ks = (np.arange(nk) + 0.5) / nk

    def _loop():
        out = np.zeros(slab["Ns"])
        for ki in ks:
            for kj in ks:
                kc = ki * b1 + kj * b2
                H = (Ms * np.exp(1j * (Tcart @ kc))[:, None, None]).sum(0)
                E, V = np.linalg.eigh(0.5 * (H + H.conj().T))
                w = np.abs(V) ** 2
                if energy_window is None:
                    occ = E < mu
                else:
                    occ = (E > energy_window[0]) & (E < energy_window[1])
                out += w[:, occ].sum(1)
        return out

    # Limit BLAS threads during the eigh loop — some OpenBLAS builds hang the
    # nested-parallel per-k diagonalisation otherwise.
    try:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            rho = _loop()
    except ImportError:
        rho = _loop()
    return rho / (nk * nk)


def _heatmap(centers_xy, weights, sigma, nx, ny, tile_vecs, tile, pad=2.0):
    """Render Σ_i w_i · Gaussian2D(r − c_i) over a rectangular grid (with in-plane tiling)."""
    imgs_c, imgs_w = [], []
    reps = range(tile) if tile_vecs is not None else [0]
    for n1 in reps:
        for n2 in (range(tile) if tile_vecs is not None and len(tile_vecs) > 1 else [0]):
            shift = np.zeros(2)
            if tile_vecs is not None:
                shift = n1 * tile_vecs[0] + (n2 * tile_vecs[1] if len(tile_vecs) > 1 else 0)
            imgs_c.append(centers_xy + shift); imgs_w.append(weights)
    C = np.vstack(imgs_c); W = np.concatenate(imgs_w)
    xlo, ylo = C.min(0) - pad; xhi, yhi = C.max(0) + pad
    xs = np.linspace(xlo, xhi, nx); ys = np.linspace(ylo, yhi, ny)
    X, Y = np.meshgrid(xs, ys)
    img = np.zeros_like(X); inv2s2 = 1.0 / (2 * sigma ** 2)
    for (cx, cy), wt in zip(C, W):
        if wt <= 1e-9:
            continue
        img += wt * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) * inv2s2)
    return img, (xlo, xhi, ylo, yhi)


def surface_charge_density(model, miller, size, *, mu=0.0, nk=12, sigma=0.6, tile=3,
                           ngrid=320, surface_thickness=None, energy_window=None,
                           cmap="turbo", show=True, savepath=None, title=None):
    """Render real-space surface charge-density heat maps of a (hkl) slab.

    Parameters
    ----------
    model : tbmodels.Model | str | pathlib.Path | dict
        The Wannier Hamiltonian. A ``tbmodels.Model`` (e.g. from a Tailwater
        prediction loaded via ``tb_model.load``), a path to a tbmodels HDF5
        or a Wannier90 ``*_hr.dat`` (DFT output), or the internal dict from
        :func:`load_hr`.
    miller : (int, int, int)
        Surface Miller index, e.g. ``(0, 0, 1)`` or ``(1, 1, 1)``.
    size : int
        Slab thickness in unit cells (cells stacked along the surface normal).
    mu : float, default 0.0
        Fermi level (eV). Occupied states are those with ``E < mu`` (ignored
        when ``energy_window`` is given). Default 0 matches the Tailwater
        training convention.
    nk : int, default 12
        Surface-BZ Monkhorst-Pack mesh is ``nk × nk``.
    sigma : float, default 0.6
        Gaussian radius (Å) used to render each Wannier centre.
    tile : int, default 3
        Number of in-plane unit-cell repetitions in the top view.
    ngrid : int, default 320
        Pixels along the long axis of each heat map.
    surface_thickness : float, optional
        Depth (Å) from the top surface counted as "the surface" for the top
        view. Defaults to ~0.6 of one layer's thickness.
    energy_window : (float, float), optional
        If given, occupy states with ``emin < E < emax`` instead of ``E < mu``.
        Use a small window around ``E_F`` (e.g. ``(-0.1, 0.1)``) to image
        topological surface states.
    cmap : str, default "turbo"
        Matplotlib colormap.
    show : bool, default True
        Display the figure (``plt.show()``).
    savepath : str | pathlib.Path, optional
        If given, save the figure to this path (PNG by default).
    title : str, optional
        Figure supertitle; defaults to a description built from the inputs.

    Returns
    -------
    dict with keys: ``rho`` (per-orbital occupation), ``pos``, ``depth``,
    ``layer``, ``nhat``, ``top_img``, ``side_img``, ``top_extent``,
    ``side_extent``, ``slab``, ``sup``, and ``fig`` (the matplotlib Figure,
    or None when ``show`` is False and no figure was created).
    """
    import matplotlib.pyplot as plt

    m = _as_model(model)
    sup = build_supercell(m, miller_supercell_matrix(miller, m["A"]))
    slab = cut_slab(sup, size)
    pos, depth, layer, nhat, norb_c = _orbital_geometry(sup, size)
    rho = _occupations(slab, sup["A"], mu, nk, energy_window)

    # orthonormal in-plane frame {e1, e2}; project orbitals to (u, v) and depth
    A2 = sup["A"]
    e1 = A2[0] / np.linalg.norm(A2[0])
    e2 = A2[1] - (A2[1] @ e1) * e1; e2 /= np.linalg.norm(e2)
    U = pos @ e1; Vp = pos @ e2
    t1 = np.array([A2[0] @ e1, A2[0] @ e2]); t2 = np.array([A2[1] @ e1, A2[1] @ e2])

    # outermost layer = the surface; pick orbitals within surface_thickness of the top
    if surface_thickness is None:
        surface_thickness = max(2.0, (depth.max() - depth.min()) / size * 0.6)
    top = depth >= depth.max() - surface_thickness

    top_img, top_ext = _heatmap(np.c_[U[top], Vp[top]], rho[top], sigma,
                                ngrid, ngrid, (t1, t2), tile)
    side_img, side_ext = _heatmap(np.c_[U, depth], rho, sigma,
                                  ngrid, max(120, int(ngrid * 0.7)), (t1,), tile)

    fig = None
    if show or savepath is not None:
        hkl = "".join(str(int(x)) for x in miller)
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        im0 = ax[0].imshow(top_img, origin="lower", extent=top_ext, cmap=cmap, aspect="equal")
        ax[0].set_title(f"surface charge density  ({hkl}) top view")
        ax[0].set_xlabel(r"$x$ ($\AA$)"); ax[0].set_ylabel(r"$y$ ($\AA$)")
        fig.colorbar(im0, ax=ax[0], fraction=0.046, label=r"$\rho$ (arb.)")
        im1 = ax[1].imshow(side_img, origin="lower", extent=side_ext, cmap=cmap, aspect="auto")
        ax[1].set_title(f"({hkl}) cross-section: in-plane vs depth")
        ax[1].set_xlabel(r"in-plane $x$ ($\AA$)"); ax[1].set_ylabel(r"depth $z$ ($\AA$)")
        fig.colorbar(im1, ax=ax[1], fraction=0.046, label=r"$\rho$ (arb.)")
        if title is None:
            title = (f"({hkl}) slab — {size} unit cells, {slab['Ns']} orbitals, "
                     f"E_F={mu} eV"
                     + ("" if energy_window is None else f", window={energy_window} eV"))
        fig.suptitle(title)
        fig.tight_layout()
        if savepath is not None:
            fig.savefig(str(savepath), dpi=130, bbox_inches="tight")
        if show:
            plt.show()

    return dict(rho=rho, pos=pos, depth=depth, layer=layer, nhat=nhat,
                top_img=top_img, side_img=side_img,
                top_extent=top_ext, side_extent=side_ext,
                slab=slab, sup=sup, fig=fig)

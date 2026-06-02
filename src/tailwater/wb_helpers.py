"""WannierBerri helpers — spin matrix elements for SHC and friends.

WannierBerri's `System_R.from_tbmodels(model, spin=True)` does not work
in the general case: tbmodels carries only the Hamiltonian, not the
spin matrix elements ``<n,0 | S^alpha | m,R>`` that Berry-curvature-type
spin calculators (SHC, spin Hall conductivity dipole, spin Berry
curvature, etc.) need.

For the Tailwater 18-orbital basis we *do* know the spin structure
exactly. Every Wannier function in the model is an eigenstate of
S_z with the convention::

    orbital_index = spatial_index * 2 + spin_index

i.e. orbital index ``2k+0`` is the spin-up partner of orbital ``2k+1``
for the same spatial orbital on the same atom. With that knowledge we
can populate ``SS_R`` (the (R=0) block of the spin matrix) analytically
as a sum over (atom, spatial) doublets.

Two entry points:

* :func:`wb_system_with_spin` — takes a ``tbmodels.Model``, returns a
  ``wannierberri.system.System_R`` with both ``berry=True`` and the
  ``SS_R`` matrix populated. Pair this with
  ``wannierberri.calculators.static.SHC`` (or any other spin
  calculator) and you can compute SHC end-to-end.

* :func:`spin_pairs_from_basis_json` — pulls the (up, down)
  subspace-index pairs out of a ``.basis.json`` written by
  ``subspace_projection``. Useful if you want to inspect or hand-edit
  the pairs before building the WB system.

Conventions
-----------
* Spin matrix elements are returned in **units of the Pauli matrix
  itself** (dimensionless), matching WannierBerri's internal
  ``set_spin_pairs`` convention. The eigenvalues of ``SS_R0[..., 2]``
  along its diagonal are ±1, so SHC outputs come out in
  ``(ℏ/e) · S/m`` when read straight off the WannierBerri result.
* All spin entries live at ``R = (0, 0, 0)`` only — spin is local
  in this basis (no inter-cell spin couplings).
"""

import json
import os
from typing import List, Optional, Tuple, Union

import numpy as np
import tbmodels


# Pauli matrix elements indexed as sigma[spin_row, spin_col] -> (x, y, z) triple.
# Equivalent to WannierBerri's internal `pauli_xyz`.
_PAULI_XYZ = {
    (0, 0): (0.0 + 0j,  0.0 + 0j,  1.0 + 0j),     # <up|σ|up>
    (0, 1): (1.0 + 0j, -1.0j,      0.0 + 0j),     # <up|σ|down>
    (1, 0): (1.0 + 0j, +1.0j,      0.0 + 0j),     # <down|σ|up>
    (1, 1): (0.0 + 0j,  0.0 + 0j, -1.0 + 0j),     # <down|σ|down>
}


def spin_pairs_from_basis_json(basis_json_path: str) -> List[Tuple[int, int]]:
    """Read (up_index, down_index) spin doublets from a ``.basis.json``.

    The basis JSON is what :func:`tailwater.subspace_projection` writes
    alongside the projected hr-model. Each entry has an
    ``atom_index``, an ``orbital_index`` (in the per-atom 18-basis),
    a ``subspace_index``, a ``spatial`` label and a ``spin`` label.
    For every ``(atom, spatial)`` combination present with **both**
    "up" and "down" partners, this returns the ordered pair of their
    subspace indices.

    Args
    ----
    basis_json_path : str
        Path to a ``.basis.json`` written by ``subspace_projection``.

    Returns
    -------
    list of (int, int) tuples
        ``[(up_idx, down_idx), ...]`` in canonical (up, down) order.
        Spatial orbitals that survived the subspace mask for only one
        spin (rare; possible if the energy window only contained the
        spin-up half) are silently dropped — they cannot contribute
        coherent intra-doublet spin matrix elements.

    Raises
    ------
    FileNotFoundError, ValueError, KeyError
        On a missing or malformed basis JSON.
    """
    if not os.path.isfile(basis_json_path):
        raise FileNotFoundError(f"basis JSON not found: {basis_json_path!r}")

    with open(basis_json_path) as f:
        doc = json.load(f)

    # Walk atoms, group by (atom_index, spatial) → {spin_label: subspace_idx}.
    by_site = {}
    for atom_block in doc.get("atoms", []):
        a = int(atom_block["atom_index"])
        for orb in atom_block.get("active_orbitals", []):
            key = (a, orb["spatial"])
            by_site.setdefault(key, {})[orb["spin"]] = int(orb["subspace_index"])

    pairs: List[Tuple[int, int]] = []
    for spinmap in by_site.values():
        if "up" in spinmap and "down" in spinmap:
            pairs.append((spinmap["up"], spinmap["down"]))
    return pairs


def _spin_pairs_from_full_basis(num_orb: int) -> List[Tuple[int, int]]:
    """Derive spin pairs for a model that has a full 18-orbital-per-atom basis.

    Assumes the orbital ordering ``orbital_index = atom*18 + spatial*2 + spin``,
    so ``num_orb`` must be divisible by 18. Tailwater hr-models from the
    API are always *compact* (only the populated orbitals per atom are
    kept), so this path is almost never the right one — see
    :func:`spin_pairs_from_model_topology` for the universally-applicable
    fallback that recovers the pairs by walking atomic positions.
    """
    if num_orb % 18 != 0:
        raise ValueError(
            f"Full-basis spin-pair derivation expects num_orb to be a multiple "
            f"of 18 (9 spatial orbitals × 2 spins per atom), got {num_orb}. "
            f"Use `spin_pairs_from_model_topology(model)` instead — it works "
            f"for any compact Wannier basis produced by the Tailwater API."
        )
    num_atoms = num_orb // 18
    pairs: List[Tuple[int, int]] = []
    for a in range(num_atoms):
        for spatial in range(9):
            base = a * 18 + spatial * 2
            pairs.append((base, base + 1))
    return pairs


def spin_pairs_from_model_topology(
    model: tbmodels.Model,
    *,
    pos_tol: float = 1e-6,
    energy_tol: float = 1e-3,
) -> List[Tuple[int, int]]:
    """Infer σ_z eigenstate doublets from the model's geometric structure.

    Works on any Tailwater hr-model — full *or* compact subspace
    projection — by exploiting two structural properties:

    1. Orbitals on the same atom share the same ``model.pos`` row
       (within ``pos_tol``).
    2. Within one atom block, the API's ``build_hr_model_fast`` walks
       the per-atom 18-basis row-major in canonical order
       ``(s↑, s↓, pz↑, pz↓, px↑, px↓, ...)``, and a Kramers pair is
       never broken — for any spatial orbital that survived the
       projection, *both* spin partners survived too. The compact
       compact-orbital indices within one atom are therefore an even
       count, and the doublets are the consecutive pairs
       ``(j, j+1), (j+2, j+3), ...``.

    The Kramers identity is verified by checking that paired orbitals
    have matching ``hop[(0,0,0)]`` diagonal entries (on-site energies).
    Discrepancies above ``energy_tol`` raise — a near-impossible failure
    mode for a TRS-symmetric Wannier model, but worth flagging if it
    happens (e.g. someone passed a TRS-broken hr-file by mistake).

    Args
    ----
    model : tbmodels.Model
        Loaded tight-binding model (e.g. via ``tb_model.load``).
    pos_tol : float, default 1e-6
        Positions closer than this in fractional coordinates are
        considered the same atom.
    energy_tol : float, default 1e-3 eV
        Maximum allowed deviation between paired on-site energies.

    Returns
    -------
    list of (up_idx, down_idx) tuples
        σ_z eigenstate doublets, indexed into the model's compact
        orbital order.
    """
    pos = np.asarray(model.pos)
    n   = int(pos.shape[0])

    # Group consecutive orbitals by atomic position. We rely on the
    # API's row-major-by-atom emission order, so atoms appear in
    # contiguous blocks — no need for full clustering.
    atom_blocks: List[List[int]] = [[0]]
    for i in range(1, n):
        if np.linalg.norm(pos[i] - pos[atom_blocks[-1][-1]]) <= pos_tol:
            atom_blocks[-1].append(i)
        else:
            atom_blocks.append([i])

    # On-site energies live on the diagonal of hop[(0,0,0)]. Use the
    # 2x convention (see tb_model.to_pb docstring) and take the real
    # part; pairing tolerance is generous so float-precision noise from
    # the API's serialisation doesn't trip the check.
    h0     = model.hop.get((0, 0, 0))
    if h0 is not None:
        h0_arr = np.asarray(h0.toarray() if hasattr(h0, "toarray") else h0)
        onsite = 2.0 * np.real(np.diag(h0_arr))
    else:
        onsite = np.zeros(n)

    pairs: List[Tuple[int, int]] = []
    for block in atom_blocks:
        if len(block) % 2 != 0:
            raise ValueError(
                f"Atom block at position {pos[block[0]].tolist()} has an "
                f"odd number of compact orbitals ({len(block)}), which is "
                f"impossible for a Kramers-paired Wannier basis. The model "
                f"may be TRS-broken or the basis may be in an unexpected "
                f"order; pass `pairs` explicitly to override."
            )
        for k in range(0, len(block), 2):
            up_i, dn_i = block[k], block[k + 1]
            if abs(onsite[up_i] - onsite[dn_i]) > energy_tol:
                raise ValueError(
                    f"Kramers check failed at compact indices ({up_i}, {dn_i}): "
                    f"on-site energies {onsite[up_i]:+.6f} / {onsite[dn_i]:+.6f} eV "
                    f"differ by more than `energy_tol={energy_tol}`. The model "
                    f"may not be TRS-symmetric, or the orbital ordering may "
                    f"not be the canonical (s↑, s↓, pz↑, pz↓, ...)."
                )
            pairs.append((up_i, dn_i))
    return pairs


def build_ss_r0(num_orb: int, pairs: List[Tuple[int, int]]) -> np.ndarray:
    """Construct the on-site spin matrix ``SS(R=0)`` from σ_z eigenstate pairs.

    Args
    ----
    num_orb : int
        Number of Wannier functions in the model (= ``model.size``).
    pairs : list of (up_idx, down_idx) tuples
        σ_z eigenstate doublets in the Wannier basis.

    Returns
    -------
    np.ndarray of shape ``(num_orb, num_orb, 3)``, complex
        ``[i, j, alpha]`` element is ``<i, 0 | sigma_alpha | j, 0>``
        with ``alpha`` in ``(x, y, z)``. Entries outside the supplied
        pairs are zero (no spin mixing across distinct spatial slots).
    """
    SS_R0 = np.zeros((num_orb, num_orb, 3), dtype=complex)
    for up_i, dn_i in pairs:
        for (r, c), vec in _PAULI_XYZ.items():
            i = up_i if r == 0 else dn_i
            j = up_i if c == 0 else dn_i
            SS_R0[i, j] = vec
    return SS_R0


def wb_system_with_spin(
    model: Union[tbmodels.Model, str],
    basis_json_path: Optional[str] = None,
    *,
    pairs: Optional[List[Tuple[int, int]]] = None,
    berry: bool = True,
    verbose: bool = True,
):
    """Build a ``wannierberri.System_R`` with ``SS_R`` populated for spin calcs.

    Resolves the σ_z eigenstate doublets one of three ways (in order
    of priority):

    1. ``pairs`` explicitly supplied by the caller — most flexible,
       lowest overhead.
    2. ``basis_json_path`` parsed via :func:`spin_pairs_from_basis_json`
       — the standard path for a subspace-projected hr-model.
    3. **(default)** Inferred from the model's geometry via
       :func:`spin_pairs_from_model_topology` — groups orbitals at
       the same atomic position and pairs them as consecutive Kramers
       doublets. Works on every Tailwater hr-model (full or compact
       subspace projection).

    With the resulting system, every WannierBerri spin calculator
    (e.g. ``wannierberri.calculators.static.SHC``,
    ``wannierberri.calculators.tabulate.Spin``, ...) works exactly as
    documented for an *ab-initio* system that carried its own ``SS_R``.

    Args
    ----
    model : tbmodels.Model or str
        The tight-binding model, or path to an HDF5 the API produced.
    basis_json_path : str, optional
        Path to ``.basis.json`` if the model is subspace-projected.
    pairs : list of (int, int), optional
        Explicit σ_z eigenstate doublets, bypassing the JSON / full-basis
        derivation.
    berry : bool, default True
        Also build position matrix elements (``AA_R``). Required for
        every Berry-curvature calculator — leave True unless you know
        you only need diagonal spin texture.
    verbose : bool, default True
        Print one summary line listing the number of (up, down) pairs
        attached and the resulting num_wann.

    Returns
    -------
    wb.system.System_R
        With ``berry=True`` data (if ``berry=True``), ``SS_R`` populated
        at R=(0,0,0) from the supplied pairs, and ``spinor=True`` set so
        downstream calculators recognise the system as spinful.

    Example
    -------
    .. code-block:: python

        import numpy as np
        import wannierberri as wb
        from tailwater import tb_model, wb_system_with_spin

        model = tb_model.load("wannier90_hr.hdf5")
        sys   = wb_system_with_spin(model, "wannier90.basis.json")

        Efermi = np.linspace(-1.0, 1.0, 51)
        grid   = wb.Grid(sys, NK=(12, 12, 12), NKFFT=(6, 6, 6))
        result = wb.run(
            sys, grid=grid,
            calculators={
                "shc": wb.calculators.static.SHC(Efermi=Efermi),
            },
            parallel=False, symmetrize=False, dump_results=False,
        )
        sigma_shc = result.results["shc"].data        # (Nef, 3, 3, 3) in (ℏ/e)·S/m
    """
    # Lazy import — keeps tailwater importable on hosts without wannierberri.
    try:
        import wannierberri as wb
    except ImportError as exc:
        raise ImportError(
            "wb_system_with_spin requires wannierberri: pip install wannierberri"
        ) from exc

    # Resolve the model
    if isinstance(model, str):
        model = tbmodels.Model.from_hdf5_file(model)

    # Resolve the σ_z eigenstate pairs
    if pairs is None:
        if basis_json_path is not None:
            pairs = spin_pairs_from_basis_json(basis_json_path)
        else:
            pairs = spin_pairs_from_model_topology(model)

    if not pairs:
        raise ValueError(
            "No σ_z eigenstate pairs could be resolved. For a "
            "subspace-projected model where the energy window only kept "
            "one spin partner per spatial slot, this is expected — the "
            "spin Hall conductivity is not well-defined in such a "
            "restricted basis. Re-run the projection with a wider window."
        )

    # Build the base wb System from the tbmodels Hamiltonian.
    sys_wb = wb.system.System_R.from_tbmodels(model, berry=berry)

    # Build and install the spin matrix at R=(0,0,0).
    SS_R0 = build_ss_r0(int(sys_wb.num_wann), pairs)
    sys_wb.set_R_mat("SS", SS_R0, R=[0, 0, 0], reset=True)
    sys_wb.spinor = True

    if verbose:
        print(
            f"[tailwater] wb_system_with_spin: attached SS_R for "
            f"{len(pairs)} σ_z eigenstate pair(s) "
            f"out of num_wann={sys_wb.num_wann}."
        )

    return sys_wb

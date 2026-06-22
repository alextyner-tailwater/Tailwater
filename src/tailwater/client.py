#!/usr/bin/env python
# coding: utf-8
"""Client-side helpers for hitting the Tailwater inference API.

Two entry points:

  tw_api_call(...)     — accepts an in-memory pymatgen Structure
                         (no disk round-trip), uses `requests` for the
                         POST (proper status-code + error handling),
                         and supports five inference modes:

                           * (default):
                             receive a tbmodels HDF5 hr-model.
                           * return_embeddings = True:
                             receive a .pt file containing the
                             pre-head backbone embeddings, suitable
                             for energetic-subspace fine-tuning via
                             finetune_heads.py / finetune_subspace.py.
                           * return_input = True:
                             receive a .pt file containing the raw
                             GNN input graph — no model inference runs.
                             Useful for debugging the structure-to-graph
                             pipeline or for fully-offline inference.
                           * return_graph_output = True:
                             receive a .pt file containing the model's
                             dense head outputs (edge_pred,
                             onsite_pred) — full forward runs but
                             tbmodels assembly is skipped. Use this to
                             debug the tbmodels-construction step
                             locally without re-running the model.
                           * project = True:
                             receive a SINGLE zip containing all of
                             {wannier90_hr.hdf5, embeddings.pt,
                             graph_output.pt} — the exact bundle
                             finetune_heads.subspace_projection needs.
                             Costs one credit; saves two follow-up API
                             round trips. The zip is auto-extracted
                             into `output_path` and the function returns
                             a dict mapping artifact -> filesystem path.

                           * symmetrize = True:
                             receive a zip containing a Kramers-degeneracy-
                             enforced tbmodels HDF5, the raw HDF5, and a
                             per-k helper script. The server detects spatial
                             inversion (P) / C2 around z (C₂ᶻ); if present
                             it applies a minimum-perturbation spectral fix
                             on an adaptive k-mesh (Δk ≈ 0.1 Å⁻¹ by default)
                             — bands stay as close to the raw prediction as
                             possible while doublets become Kramers-paired.
                             If neither symmetry is present the raw model
                             is returned unchanged (a note explains why
                             generic-k splittings must not be averaged
                             out for non-PT crystals). For exact Kramers
                             at arbitrary k, call the bundled
                             `kramers_helper.per_k_kramers_fix(raw, k)`
                             on the raw HDF5. Costs one credit.

                         If multiple flags are True the most expensive
                         request wins:
                           project > symmetrize > return_input
                                   > return_embeddings
                                   > return_graph_output > default HDF5.

  tb_model.load(...)   — local HDF5 loader, not an API call. Reads a
                         tight-binding model produced by the API and
                         returns the standard tbmodels.Model with an
                         instance-bound `.to_pb()` method that converts
                         it to a pybinding.Lattice for visualization /
                         transport workflows.

Both API entry points use HTTP Basic auth — credentials are checked against
the server-side users.db, and each accepted call decrements the caller's
credit balance by 1 (enforced by the `require_credit` dependency on the
server). On 401 (bad credentials) or 402 (out of credits) we surface a
clean Python exception so calling code can react.
"""

import json
import os
import types
from typing import Optional, Union

import numpy as np
import requests
from pymatgen.core.structure import Structure

# tbmodels is needed by the `tb_model.load(...)` helper at the bottom of
# this file. Pybinding is imported lazily inside `_to_pb_method` so this
# module still imports cleanly on hosts that only need the HTTP client
# parts (tw_api_call).
import tbmodels


# Default API location: the production Tailwater inference API.
# (The `api_url=` argument and the TW_API_URL env var can override this in
# the rare case the Tailwater team points you at a different endpoint —
# the production URL is what every normal user should hit.)
DEFAULT_API_URL = os.environ.get("TW_API_URL", "https://api.tailwater.io")


# =====================================================
# ENTRY POINT  (in-memory Structure, requests-based)
# =====================================================

# Endpoint routing for the API modes. Names come from API/RunAPI.py.
_ENDPOINT_FULL_HDF5       = "/upload_json_process_and_download_dat/"
_ENDPOINT_EMBEDDINGS_PT   = "/upload_structure_and_download_embeddings/"
_ENDPOINT_INPUT_PT        = "/upload_structure_and_download_input/"
_ENDPOINT_GRAPH_OUTPUT_PT = "/upload_structure_and_download_graph_output/"
_ENDPOINT_PROJECT_ZIP     = "/upload_structure_and_download_project/"
_ENDPOINT_SYMMETRIZED_ZIP = "/upload_structure_and_download_symmetrized/"


def tw_api_call(
    structure: Structure,
    user: str,
    password: str,
    output_path: str,
    filename: str,
    return_embeddings: bool = False,
    return_input: bool = False,
    return_graph_output: bool = False,
    project: bool = False,
    symmetrize: bool = True,
    api_url: str = DEFAULT_API_URL,
    timeout: float = 600.0,
    save_cif: bool = True,
    keep_zip: bool = False,
    dev: bool = False,
    model: Optional[str] = None,
):
    """Submit a pymatgen Structure to the API and save the response.

    Five output modes are available, each tapping into a different stage
    of the inference pipeline. They are mutually exclusive; if multiple
    flags are True, `project` wins, then return_input >
    return_embeddings > return_graph_output > default full HDF5.

      * (default) full inference         -> tbmodels HDF5 hr-model (.hdf5)
      * return_embeddings = True         -> pre-head backbone embeddings (.pt)
      * return_input      = True         -> raw GNN input graph (.pt),
                                            no model inference runs at all
      * return_graph_output = True       -> dense model output
                                            (edge_pred, onsite_pred) plus
                                            graph metadata, before tbmodels
                                            assembly. Use this to debug the
                                            tbmodels build step locally
                                            without re-running the model.
      * project           = True         -> BUNDLE mode: a single zip
                                            containing all of {full HDF5,
                                            embeddings.pt, graph_output.pt}
                                            — every artifact
                                            finetune_heads.subspace_projection
                                            needs, in one API call (one credit).
                                            The zip is extracted into
                                            `output_path` and a dict mapping
                                            artifact name to path is returned
                                            instead of a single string.
      * symmetrize        = True         -> SYMMETRIZATION mode: a single
                                            zip containing the symmetrized
                                            HDF5 (post-WannSymm), the raw
                                            (pre-symm) HDF5, the wannsymm.in
                                            actually used, and the wannsymm
                                            stdout/stderr log. One credit
                                            per call. Use this when you
                                            want the predicted Hamiltonian
                                            to obey the crystal's point /
                                            space group symmetries
                                            exactly, as a post-processing
                                            cleanup on top of inference.

    Parameters
    ----------
    structure : pymatgen.core.structure.Structure
        The structure to model. Serialized in memory via
        ``Structure.as_dict()`` -> JSON; no on-disk intermediate file
        is created.
    user, password : str
        HTTP Basic auth credentials. Must match a row in the server's
        users.db. Each successful call decrements the user's credit
        balance by 1 on the server side.
    output_path : str
        Local directory where the response will be saved (created if
        missing).
    filename : str
        Filename stem (without extension). The extension is chosen by
        the mode: ".hdf5" for the full hr-model, ".pt" for either the
        embeddings or the input-graph .pt files.
    return_embeddings : bool, default False
        Hit the embeddings endpoint instead of full inference. The .pt
        file is a dict with keys ``data`` (PyG Data object with .f_out
        and .edge_feat populated), ``LM`` (lattice), ``atoms``, and
        ``irreps_in``.
    return_input : bool, default False
        Hit the raw-input endpoint — no model inference runs. The .pt
        file is a dict with keys ``data`` (PyG Data object with the
        structural fields only: node_features, edge_index, edge_vectors,
        inv_data, atom_number), ``LM``, and ``atoms``. Use this to
        inspect the parsed graph (e.g. before feeding it through your
        own model + heads), or for offline experimentation that doesn't
        need a full server-side inference call.
    return_graph_output : bool, default False
        Run the full model but DON'T assemble tbmodels. The .pt file
        is a dict with keys ``sparse_edge_list`` (list of [18,18]
        complex CSR matrices, one per edge), ``sparse_onsite_list``
        (one [18,18] CSR per atom), plus ``data`` / ``LM`` / ``atoms``.
        Use this to debug the tbmodels assembly step (phase convention,
        sublattice positions, duplicate-hop handling) by feeding the
        sparse matrices into your own / a modified hr_export.build_hr_model
        locally — no model re-runs needed per attempt.
    project : bool, default False
        Bundle mode for the subspace-projection workflow. Server runs
        the full pipeline ONCE and returns a single zip containing
        wannier90_hr.hdf5 + embeddings.pt + graph_output.pt. The zip is
        extracted into `output_path` and the function returns a dict
        instead of a single path:
            {"hdf5": "...", "embeddings": "...", "graph_output": "..."}
        Costs one credit per call regardless of how many artifacts.
        Wins over the other `return_*` flags if multiple are True.
    symmetrize : bool, default True
        Kramers-degeneracy enforcement. When True (the default) the
        server applies the minimum-perturbation spectral fix to the
        prediction if the crystal has spatial inversion (P) or C2 around
        z (C₂ᶻ); if not, the raw model is returned unchanged with a note
        explaining why generic-k splittings (Rashba / Weyl-style) must
        not be averaged out. Either way you get a single
        ``wannier90_hr.hdf5`` under the same key, so callers can ignore
        the symmetry detail and just load ``r["hdf5"]``. The bundle is:
            {"hdf5":            "...",  # the (possibly Kramers-fixed) model
             "win":             "...",  # canonical .win
             "symmetrize_note": "..."}  # symmetry findings + diagnostics
        Set ``symmetrize=False`` to get the raw prediction (no fix, no
        symmetry check). Loses to ``project`` and the ``return_*`` flags
        if any of those is also True. For exact Kramers at arbitrary k
        (band paths, BZ integration on non-mesh k), hit the PT endpoint
        directly — it bundles the raw HDF5 + a per-k helper script.
    keep_zip : bool, default False
        When `project=True`, controls whether the downloaded .zip is
        retained after extraction. Default False (delete the zip;
        keep only the three unpacked artifacts).
    api_url : str
        Base URL of the API. Defaults to ``https://api.tailwater.io``
        (the production deployment). Almost no one should need to set
        this — only pass it if the Tailwater team specifically pointed
        you at a different endpoint.
    timeout : float
        Request timeout in seconds. Backbone inference on a 50-atom
        material is typically <60 s on CPU; the default 600 s is
        generous for batched / cold-start cases.
    save_cif : bool, default True
        If True, also write the structure to ``{output_path}/Structure.cif``.
        Set False to skip.
    dev : bool, default False
        Opt into the server's canonical-cell position-wrap fix (sent as
        ``?dev=true``). Corrects band structures for inputs whose atoms sit
        on/over the unit-cell boundary (e.g. fractional coords numerically
        ~1.0). Default False reproduces the current production behavior, and
        the flag is harmlessly ignored by servers that predate the patch.
    model : str, optional (default None)
        Model checkpoint version. When None (the default) the SDK does
        NOT forward ?model= to the server, so the server's own default
        applies — i.e. whichever checkpoint the operator most recently
        promoted to default with DEFAULT_MODEL in RunAPI.py. Pass a
        specific version string to force a particular checkpoint:
          * "V0.0" → evMace_Epoch_51.pth (the original GWANN release).
          * "V0.1" → Mace_FT2_Gaps_Epoch_7.pth (FT2-Gaps fine-tune;
                      the current production default since 2026-06-15).
        Unknown versions return 400 with the list of valid choices.
        Older deployments without the registry silently ignore the flag
        (FastAPI tolerates unknown query params), so forwarding is
        backward-safe.

    Returns
    -------
    dict
        Always a dict. Keys depend on the mode:
          default      -> {"hdf5":         "...", "win": "..."}
          return_input -> {"input":        "...", "win": "..."}
          return_embeddings   -> {"embeddings":   "...", "win": "..."}
          return_graph_output -> {"graph_output": "...", "win": "..."}
          project      -> {"hdf5": "...", "embeddings": "...",
                           "graph_output": "...", "win": "..."}
          symmetrize=True
            (default)  -> {"hdf5": "...", "win": "...", "symmetrize_note": "..."}
        The ``"win"`` key always points at the canonical wannier90.win
        file the server actually ran inference on — useful for
        reproducing the exact graph the server built from your structure
        (positions, lattice, projections) in any downstream tool.

    Raises
    ------
    PermissionError
        On HTTP 401 — bad username/password.
    RuntimeError
        On HTTP 402 — out of credits.
        On any other non-2xx response — surfaces the server's detail
        message for debugging.
    """
    os.makedirs(output_path, exist_ok=True)

    # ---- Serialize the structure in memory ----
    # Stream the JSON straight into the multipart upload. No temp file
    # on disk means no race conditions across concurrent invocations
    # and no cleanup needed.
    payload_bytes = json.dumps(structure.as_dict()).encode("utf-8")

    # Optional CIF dump on the client side.
    if save_cif:
        try:
            structure.to(filename=os.path.join(output_path, "Structure.cif"))
        except Exception as cif_err:
            # Non-fatal: the response file is still what the user asked
            # for. Surface the warning but keep going.
            print(f"[tw_api_call] Warning: failed to write Structure.cif: {cif_err}")

    # ---- Route to the right endpoint ----
    # Priority order:
    #   project > return_input > return_embeddings > return_graph_output
    #          > symmetrize > raw-HDF5 (symmetrize=False)
    # `project` wins because it's the most expensive/comprehensive
    # subspace-projection bundle. The `return_*` flags select alternate
    # output *types* (graph, embeddings, raw model tensors) and beat
    # `symmetrize`, which only chooses between the Kramers-fixed and the
    # raw HDF5 of the same primary artifact. With symmetrize=True as the
    # new default, demoting it below the return_* flags is required —
    # otherwise every call to e.g. return_embeddings would silently get
    # symmetrized HDF5 instead of the requested embeddings.
    # The server returns a ZIP for every endpoint — the zip bundles the
    # primary artifact alongside the canonical `input.win` file that was
    # actually parsed and run through inference. The client extracts the
    # zip on receipt and returns a dict of paths (the `.win` key is always
    # present).
    if project:
        endpoint        = _ENDPOINT_PROJECT_ZIP
        primary_arcname = None       # multiple primary artifacts in this bundle
    elif return_input:
        endpoint        = _ENDPOINT_INPUT_PT
        primary_arcname = "gnn_input.pt"
    elif return_embeddings:
        endpoint        = _ENDPOINT_EMBEDDINGS_PT
        primary_arcname = "embeddings.pt"
    elif return_graph_output:
        endpoint        = _ENDPOINT_GRAPH_OUTPUT_PT
        primary_arcname = "graph_output.pt"
    elif symmetrize:
        # Default branch — Kramers-fixed HDF5 named with the standard
        # primary arcname so callers see one "wannier90_hr.hdf5" file
        # regardless of whether the server applied the spectral fix.
        endpoint        = _ENDPOINT_SYMMETRIZED_ZIP
        primary_arcname = "wannier90_hr.hdf5"
    else:
        endpoint        = _ENDPOINT_FULL_HDF5
        primary_arcname = "wannier90_hr.hdf5"
    out_file_path = os.path.join(output_path, filename + ".zip")

    # ---- POST with streaming so large HDF5 / .pt files don't OOM ----
    # `dev=True` opts into the server's canonical-cell position-wrap fix for
    # band structures (sent as a ?dev=true query param, leaving the multipart
    # body untouched). Omitted when False so default requests are byte-identical
    # to the pre-flag client, and harmlessly ignored by servers without the
    # dev-flag patch deployed.
    files  = {"file": ("structure.json", payload_bytes, "application/json")}
    params = {}
    if dev:
        params["dev"] = "true"
    if model is not None:
        # Always forward `model` when set — even when the caller asked for
        # V0.0 — because the server's default can change (e.g. when the
        # operator promotes a new release to default, an unforwarded
        # V0.0 request would silently return the new default instead).
        # Old server builds without the registry tolerate unknown query
        # params (FastAPI ignores them by default), so forwarding is
        # backward-safe.
        params["model"] = model
    params = params or None
    response = requests.post(
        api_url.rstrip("/") + endpoint,
        files   = files,
        params  = params,
        auth    = (user, password),
        timeout = timeout,
        stream  = True,
    )

    # ---- Error handling: surface server-side credit / auth state cleanly ----
    if response.status_code == 401:
        raise PermissionError(
            "API returned 401: incorrect username or password. "
            "Check credentials with the API provider."
        )
    if response.status_code == 402:
        # Match the server's `require_credit` 402 response.
        raise RuntimeError(
            "API returned 402: account is out of credits. "
            "Ask the API provider to top up your balance."
        )
    if not response.ok:
        # Surface the server's detail blob (truncated) so the caller
        # has something to grep for.
        body_preview = response.text[:500] if response.text else "<empty>"
        raise RuntimeError(
            f"API returned {response.status_code}: {body_preview}"
        )

    # ---- Stream response body to disk in 8 KiB chunks ----
    with open(out_file_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    # ---- Extract the bundle zip ----
    # Every endpoint now returns a zip containing the primary artifact(s)
    # PLUS the canonical `input.win` file (the deterministic record of
    # exactly what the server ran inference on). Unpack everything into
    # `output_path` and return a dict mapping artifact name -> file path.
    import zipfile

    # Map server-side arcnames -> friendly dict keys the caller sees.
    _ARCNAME_TO_KEY = {
        "wannier90_hr.hdf5":         "hdf5",
        "embeddings.pt":             "embeddings",
        "graph_output.pt":           "graph_output",
        "gnn_input.pt":              "input",
        "input.win":                 "win",
        "symmetrize.note":           "symmetrize_note",
        # The PT endpoint (power-user variant) still bundles these:
        "kramers_helper.py":         "kramers_helper",
        # Kept for backward-compatibility with cached zips from older
        # server builds; new server bundles no longer include any of
        # these arcnames.
        "wannier90_symmed_hr.hdf5":  "symmed_hdf5",
        "wannsymm.in":               "wannsymm_in",
        "wannsymm.out":              "wannsymm_log",
    }

    extracted_paths = {}
    with zipfile.ZipFile(out_file_path, "r") as zf:
        zf.extractall(output_path)
        for member in zf.namelist():
            key = _ARCNAME_TO_KEY.get(member, member)
            extracted_paths[key] = os.path.join(output_path, member)

    if not keep_zip:
        try:
            os.remove(out_file_path)
        except OSError:
            pass

    return extracted_paths


# =====================================================
# CREDIT-BALANCE HELPER  (optional)
# =====================================================
# Some clients will want to know how many credits remain before they
# blow through a big batch. Not currently exposed as an API endpoint;
# expose this client-side helper if/when the server gets a GET /credits/
# route.
def remaining_credits(user: str, password: str,
                      api_url: str = DEFAULT_API_URL) -> Optional[int]:
    """Return the caller's current credit balance via GET /credits/.

    This is a free, read-only check — the server's /credits/ route
    authenticates but does NOT consume a credit. Returns None if the
    server doesn't expose the endpoint (e.g. an older deployment that
    predates it, which answers 404) or on any non-auth error; raises
    PermissionError on 401 (bad credentials).
    """
    try:
        response = requests.get(api_url.rstrip("/") + "/credits/",
                                auth=(user, password),
                                timeout=30)
    except Exception:
        return None
    if response.status_code == 404:
        return None
    if response.status_code == 401:
        raise PermissionError("API returned 401: incorrect username or password.")
    if not response.ok:
        return None
    try:
        return int(response.json().get("credits"))
    except Exception:
        return None


# =====================================================
# HDF5 LOADER WITH PYBINDING-CONVERSION HELPER
# =====================================================
# `tb_model.load(path)` reads an HDF5 tight-binding model produced by the
# API (the .hdf5 file shipped by /upload_json_process_and_download_dat/
# or extracted from a /upload_structure_and_download_project/ bundle)
# and returns the underlying tbmodels.Model with one extra method
# attached: `.to_pb()` converts the loaded model into a pybinding
# Lattice object for visualization / transport workflows.
#
# Why bind the method to the instance instead of subclassing:
#   - The returned object still passes `isinstance(model, tbmodels.Model)`
#     so any downstream code that type-checks the model behaves
#     unchanged.
#   - tbmodels.Model has no __slots__ restriction, so binding an
#     instance attribute via types.MethodType is safe.
#   - Subclassing + __class__ reassignment also works, but is more
#     fragile if tbmodels later switches to slots or uses a custom
#     metaclass.

def _to_pb_method(self, lattice_vectors=None, hop_threshold: float = 1e-12):
    """Convert this tbmodels.Model into a pybinding.Lattice.

    Bound as ``model.to_pb`` on instances returned by
    :func:`tb_model.load`. After conversion, eigenvalues of
    ``model.hamilton(k_frac)`` and the pybinding model match to
    float32 precision (~1e-6 eV) at every k.

    To plug pybinding into a band-structure calculation, pair the
    returned lattice with the companion helper :func:`k_cart_from_frac`
    — pybinding expects ``set_wave_vector(k_cart)`` in rad/length:

    .. code-block:: python

        from tailwater import tb_model, k_cart_from_frac
        import pybinding as pb

        model = tb_model.load("wannier90_hr.hdf5")
        lat   = model.to_pb()
        pmod  = pb.Model(lat, pb.translational_symmetry())

        for k_frac in k_path:
            pmod.set_wave_vector(k_cart_from_frac(k_frac, model.uc))
            bands.append(np.linalg.eigvalsh(pmod.hamiltonian.todense()))

    Args
    ----
    lattice_vectors : array-like (3, 3), optional
        Real-space lattice vectors as rows. If None, uses ``self.uc``
        (the unit cell the tbmodels.Model carried when it was loaded);
        if that's also None, falls back to ``np.eye(3)``.
    hop_threshold : float, default 1e-12
        Skip hops with ``|val| <= hop_threshold``. Keep this low —
        we're only filtering exact-zero entries from sparse hop
        storage; the band-relevant threshold should have been applied
        upstream when the HDF5 was first written.

    Returns
    -------
    pb.Lattice
        with the same sublattices, on-site energies, lattice vectors,
        and hops as ``self``, producing the same H(k) eigenvalues at
        every k.

    Conventions
    -----------
    **On-site doubling.** tbmodels' Hamiltonian construction sums
    ``stored[R] * exp(i k . R)`` over R, then adds its Hermitian
    conjugate to symmetrise. That second step supplies the missing
    minus-R half for nonzero R, but at R=0 it doubles the stored block
    on top of itself. tbmodels therefore stores half the user-supplied
    on-site value at R=0, and the round-trip Hamiltonian matches the
    physical Hamiltonian. Pybinding has no such double-up step, so
    we feed it twice the stored R=0 block, restoring the physical
    contribution.

    **Position basis.** tbmodels stores ``self.pos`` in fractional
    coordinates. Pybinding expects positions in Cartesian. We convert
    ``pos_cart = pos_frac @ LM`` so the resulting lattice's
    Brillouin-zone and real-space geometry routines are physically
    meaningful. Eigenvalues are invariant under the per-orbital phase
    change induced by this choice — only the eigenvectors get rephased.

    **Hop duplicates.** For nonzero R, both ``(R, i, j)`` and
    ``(R, j, i)`` entries of the stored hop matrix are added explicitly
    to pybinding; the H.c. of each pybinding add automatically supplies
    the matching minus-R contribution, so the full Hamiltonian is
    reconstructed. For R = 0, the auto-implied H.c. of a given
    ``add_one_hopping`` call lands at the transposed indices —
    pybinding rejects the explicit second add as a duplicate. We catch
    that rejection silently.
    """
    # Lazy import — keeps tailwater importable on hosts without pybinding.
    import pybinding as pb

    # Resolve the real-space lattice vectors.
    if lattice_vectors is not None:
        LM = np.asarray(lattice_vectors, dtype=float)
    elif getattr(self, "uc", None) is not None:
        LM = np.asarray(self.uc, dtype=float)
    else:
        LM = np.eye(3)

    lat = pb.Lattice(a1=LM[0], a2=LM[1], a3=LM[2])

    # ---- Sublattices: position + on-site energy per orbital ----
    # Positions: convert fractional → Cartesian for pybinding.
    pos_frac = np.asarray(self.pos)             # [num_orb, 3], fractional
    pos_cart = pos_frac @ LM                     # rows-of-frac · rows-of-LM
    num_orb  = int(pos_frac.shape[0])

    # On-site: read the diagonal of hop[(0,0,0)] and double it (see
    # "On-site doubling" in the docstring above).
    hop_zero = self.hop.get((0, 0, 0))
    if hop_zero is None:
        h0 = np.zeros((num_orb, num_orb), dtype=complex)
    else:
        h0 = np.asarray(hop_zero.toarray() if hasattr(hop_zero, "toarray") else hop_zero)
    h0_phys = 2.0 * h0                                  # ← the fix.

    for i in range(num_orb):
        lat.add_one_sublattice(
            str(i),
            pos_cart[i].tolist(),
            onsite_energy=float(np.real(h0_phys[i, i])),
        )

    # ---- Hoppings ----
    # R = (0,0,0): off-diagonal entries of the doubled (0,0,0) block.
    rows, cols = np.nonzero(np.abs(h0_phys) > hop_threshold)
    R_zero = np.array([0, 0, 0], dtype=int)
    for i, j in zip(rows, cols):
        i, j = int(i), int(j)
        if i == j:
            continue                                    # diagonal handled above
        try:
            lat.add_one_hopping(R_zero, str(i), str(j), complex(h0_phys[i, j]))
        except Exception:
            # Pybinding rejects the second of {(0,0,0),i,j} / {(0,0,0),j,i}
            # as the auto-H.c. of the first. Swallow silently.
            continue

    # R ≠ (0,0,0): pass each stored hop through unchanged.
    for R, hop_mat in self.hop.items():
        if tuple(int(x) for x in R) == (0, 0, 0):
            continue
        hop_arr = np.asarray(hop_mat.toarray() if hasattr(hop_mat, "toarray") else hop_mat)
        R_arr   = np.asarray(R, dtype=int)
        rs, cs  = np.nonzero(np.abs(hop_arr) > hop_threshold)
        for i, j in zip(rs, cs):
            try:
                lat.add_one_hopping(R_arr, str(int(i)), str(int(j)), complex(hop_arr[i, j]))
            except Exception:
                continue

    return lat


def _to_pythtb_method(self, lattice_vectors=None, hop_threshold: float = 1e-12):
    """Convert this tbmodels.Model into a pythtb.tb_model.

    Bound as ``model.to_pythtb`` on instances returned by
    :func:`tb_model.load`. After conversion, eigenvalues of
    ``model.hamilton(k_frac)`` and ``py_model.solve_one(k_frac)``
    match to float64 precision (~1e-12 eV) at every k.

    The PythTB conversion is simpler than the pybinding one in two
    ways:

    * PythTB takes ``orb`` (orbital positions) in **fractional**
      coordinates, the same convention as ``tbmodels.Model.pos``, so
      no Cartesian conversion is needed.
    * PythTB's ``solve_one(k)`` accepts **fractional** k directly, so
      no analogue of :func:`k_cart_from_frac` is needed:

    .. code-block:: python

        from tailwater import tb_model

        model    = tb_model.load("wannier90_hr.hdf5")
        py_model = model.to_pythtb()

        # Sample H(k) at any fractional k:
        eig = py_model.solve_one([0.0, 0.0, 0.0])    # Γ
        eig = py_model.solve_one([0.5, 0.0, 0.0])    # M

        # ...or use PythTB's built-in band-path helpers:
        k_path, k_dist, k_node = py_model.k_path(
            [[0,0,0], [0.5,0,0], [0.333,0.333,0], [0,0,0]],
            nk=101, report=False,
        )
        bands = py_model.solve_all(k_path)            # (num_wann, nk)

    Args
    ----
    lattice_vectors : array-like (3, 3), optional
        Real-space lattice vectors as rows. If None, uses ``self.uc``;
        if that's also None, falls back to ``np.eye(3)``.
    hop_threshold : float, default 1e-12
        Skip hops with ``|val| <= hop_threshold``. Keep this low —
        only filters exact-zero entries from sparse hop storage.

    Returns
    -------
    pythtb.tb_model
        A 3D periodic tight-binding model with the same Hamiltonian
        as ``self``.  For slabs / wires, call PythTB's
        ``.cut_piece(num, fin_dir)`` on the returned model.

    Conventions
    -----------
    The same on-site doubling story as :func:`_to_pb_method` applies
    here — PythTB counts each ``(0,0,0)`` entry once (like pybinding
    does), while tbmodels effectively counts it twice via its
    ``H += H.c.`` symmetrisation. To match, we multiply
    ``hop[(0,0,0)]`` by 2 before feeding PythTB.

    PythTB defaults to ``allow_conjugate_pair=False``: only one of
    each H.c. pair may be added explicitly. At R=(0,0,0) we therefore
    only add the upper-triangle off-diagonals; PythTB fills in the
    lower triangle via its automatic conjugate. For R != 0 we add
    every nonzero entry of ``hop[+R]``; PythTB implies the
    corresponding (-R, j, i) contribution from each.
    """
    # Lazy import — keeps tailwater importable without pythtb.
    try:
        import pythtb
    except ImportError as exc:
        raise ImportError(
            "model.to_pythtb() requires the `pythtb` package, which "
            "isn't installed: pip install pythtb"
        ) from exc

    # Resolve the real-space lattice vectors.
    if lattice_vectors is not None:
        LM = np.asarray(lattice_vectors, dtype=float)
    elif getattr(self, "uc", None) is not None:
        LM = np.asarray(self.uc, dtype=float)
    else:
        LM = np.eye(3)

    # PythTB uses fractional orbital positions, matching tbmodels.
    pos_frac = np.asarray(self.pos)
    num_orb  = int(pos_frac.shape[0])
    dim      = int(getattr(self, "dim", 3))

    # Build an empty pythtb model. dim_k = dim_r = self.dim (assume
    # fully periodic; users can cut_piece() for slabs afterwards).
    py_model = pythtb.tb_model(
        dim_k=dim,
        dim_r=dim,
        lat=LM.tolist(),
        orb=pos_frac.tolist(),
    )

    # ---- On-site doubling fix (same logic as to_pb) ----
    hop_zero = self.hop.get((0, 0, 0))
    if hop_zero is None:
        h0 = np.zeros((num_orb, num_orb), dtype=complex)
    else:
        h0 = np.asarray(hop_zero.toarray() if hasattr(hop_zero, "toarray") else hop_zero)
    h0_phys = 2.0 * h0

    onsite = np.real(np.diag(h0_phys)).tolist()
    py_model.set_onsite(onsite)

    # R = (0,0,0) off-diagonals: only upper triangle (PythTB auto-fills
    # the lower triangle via its H.c. convention).
    rows, cols = np.nonzero(np.abs(h0_phys) > hop_threshold)
    R_zero = [0] * dim
    for i, j in zip(rows, cols):
        i, j = int(i), int(j)
        if i >= j:
            continue
        try:
            py_model.set_hop(complex(h0_phys[i, j]), i, j, R_zero)
        except Exception:
            continue

    # R != (0,0,0): every nonzero entry of each stored block. PythTB
    # auto-implies (-R, j, i, conj(val)) for each (R, i, j, val) call,
    # which is exactly tbmodels' missing -R half.
    for R, hop_mat in self.hop.items():
        if tuple(int(x) for x in R) == (0,) * dim:
            continue
        hop_arr = np.asarray(hop_mat.toarray() if hasattr(hop_mat, "toarray") else hop_mat)
        R_list  = [int(x) for x in R]
        rs, cs  = np.nonzero(np.abs(hop_arr) > hop_threshold)
        for i, j in zip(rs, cs):
            try:
                py_model.set_hop(complex(hop_arr[i, j]), int(i), int(j), R_list)
            except Exception:
                # Auto-H.c. duplicate-pair rejection — swallow.
                continue

    return py_model


def _to_kwant_method(self, lattice_vectors=None, hop_threshold: float = 1e-12):
    """Convert this tbmodels.Model into a kwant.Builder.

    Bound as ``model.to_kwant`` on instances returned by
    :func:`tb_model.load`. Returns an *unfinalised* ``kwant.Builder``
    with a 3D :class:`kwant.TranslationalSymmetry` so callers can
    either:

    * Finalise immediately for bulk H(k) sampling:

      .. code-block:: python

          import numpy as np, kwant
          from tailwater import tb_model

          model = tb_model.load("wannier90_hr.hdf5")
          syst  = kwant.wraparound.wraparound(model.to_kwant()).finalized()

          # Kwant's wraparound takes 2π·k_frac (per-cell Bloch phase),
          # NOT Cartesian rad/length.
          k_frac = [0.5, 0.0, 0.0]
          phase  = 2 * np.pi * np.asarray(k_frac)
          H      = syst.hamiltonian_submatrix(
              params=dict(k_x=phase[0], k_y=phase[1], k_z=phase[2]),
          )
          eigs   = np.sort(np.linalg.eigvalsh(H))

    * Attach leads / build a finite scattering region on top of the
      bulk Builder for transport calculations:

      .. code-block:: python

          bulk = model.to_kwant()
          # Cut a finite slab, add leads, attach to bulk, etc.
          # See the Kwant tutorial: https://kwant-project.org/doc/

    The returned eigenvalues match
    ``np.linalg.eigvalsh(model.hamilton(k_frac))`` to ~float64
    precision (~1e-12 eV) at every k.

    Args
    ----
    lattice_vectors : array-like (3, 3), optional
        Real-space lattice vectors as rows. If None, uses ``self.uc``;
        if that's also None, falls back to ``np.eye(3)``.
    hop_threshold : float, default 1e-12
        Skip hops with ``|val| <= hop_threshold``. Keep this low —
        only filters exact-zero entries from sparse hop storage.

    Returns
    -------
    kwant.Builder
        A 3D-periodic Builder with one site per Wannier orbital.
        Sublattices are accessible in the same order as
        ``model.pos`` via the Builder's ``lattice.sublattices`` (the
        lattice object is the first argument of the
        TranslationalSymmetry stored on the Builder).

    Conventions
    -----------
    The same on-site doubling story as :func:`_to_pb_method` and
    :func:`_to_pythtb_method` applies — Kwant counts each ``(0,0,0)``
    entry once, while tbmodels effectively counts it twice via its
    ``H += H.c.`` symmetrisation. To match, we multiply
    ``hop[(0,0,0)]`` by 2 before feeding Kwant.

    Positions are Cartesian (Kwant's convention; converted from
    fractional via ``pos_cart = pos_frac @ LM``).

    .. warning::

       The k-parameters that ``kwant.wraparound`` exposes
       (``k_x``, ``k_y``, ``k_z``) are **not** Cartesian rad/length
       like pybinding's. They are the **per-cell Bloch phase**
       (i.e. ``2π · k_frac``), independent of the physical cell size.
       To sample H(k) at the same fractional k tbmodels uses, pass
       ``k_x = 2π · k_frac[0]``, etc. — see the worked example above.
       This is the most common source of "the Kwant bands don't match
       the tbmodels bands" reports.
    """
    # Lazy import — kwant is an optional, heavy dependency.
    try:
        import kwant
    except ImportError as exc:
        raise ImportError(
            "model.to_kwant() requires the `kwant` package, which "
            "isn't installed. Kwant is best installed via conda: "
            "`conda install -c conda-forge kwant`."
        ) from exc

    # Resolve the real-space lattice vectors.
    if lattice_vectors is not None:
        LM = np.asarray(lattice_vectors, dtype=float)
    elif getattr(self, "uc", None) is not None:
        LM = np.asarray(self.uc, dtype=float)
    else:
        LM = np.eye(3)

    # Cartesian orbital positions (Kwant's convention).
    pos_frac = np.asarray(self.pos)
    pos_cart = pos_frac @ LM
    num_orb  = int(pos_frac.shape[0])

    # Build the Kwant lattice + 3D translational symmetry.
    lat = kwant.lattice.general(
        prim_vecs=LM.tolist(),
        basis=pos_cart.tolist(),
        norbs=1,
    )
    subs = lat.sublattices
    sym  = kwant.TranslationalSymmetry(
        lat.vec((1, 0, 0)),
        lat.vec((0, 1, 0)),
        lat.vec((0, 0, 1)),
    )
    builder = kwant.Builder(sym)

    # ---- On-site doubling fix (same logic as to_pb / to_pythtb) ----
    hop_zero = self.hop.get((0, 0, 0))
    if hop_zero is None:
        h0 = np.zeros((num_orb, num_orb), dtype=complex)
    else:
        h0 = np.asarray(hop_zero.toarray() if hasattr(hop_zero, "toarray") else hop_zero)
    h0_phys = 2.0 * h0

    # On-site energies (one site per Wannier orbital, in cell (0,0,0)).
    for i in range(num_orb):
        builder[subs[i](0, 0, 0)] = float(np.real(h0_phys[i, i]))

    # R = (0,0,0) off-diagonal hops — upper-triangle only. Kwant fills
    # in the lower triangle automatically when constructing H(k).
    rows, cols = np.nonzero(np.abs(h0_phys) > hop_threshold)
    for i, j in zip(rows, cols):
        i, j = int(i), int(j)
        if i >= j:
            continue
        try:
            builder[subs[i](0, 0, 0), subs[j](0, 0, 0)] = complex(h0_phys[i, j])
        except Exception:
            continue

    # R != (0,0,0): every nonzero entry of each stored block.
    for R, hop_mat in self.hop.items():
        R_tup = tuple(int(x) for x in R)
        if R_tup == (0, 0, 0):
            continue
        hop_arr = np.asarray(hop_mat.toarray() if hasattr(hop_mat, "toarray") else hop_mat)
        Rx, Ry, Rz = R_tup
        rs, cs = np.nonzero(np.abs(hop_arr) > hop_threshold)
        for i, j in zip(rs, cs):
            try:
                builder[subs[int(i)](0, 0, 0),
                        subs[int(j)](Rx, Ry, Rz)] = complex(hop_arr[i, j])
            except Exception:
                # Kwant raises if the (a,b) pair is already set or
                # canonicalises to an existing one — swallow.
                continue

    return builder


def k_cart_from_frac(k_frac, lattice_vectors) -> np.ndarray:
    """Convert a fractional k-point to Cartesian (rad/length) for pybinding.

    Pybinding's ``set_wave_vector(k)`` expects ``k`` in rad/length —
    i.e. in the basis of the Cartesian reciprocal-lattice vectors
    ``b_i``, not the fractional ``k_i`` Wannier90 and tbmodels use by
    default. The conversion is::

        k_cart = 2π · inv(LM) @ k_frac

    where ``LM`` has the real-space lattice vectors as rows.

    Args
    ----
    k_frac : array-like, shape (3,) or (N, 3)
        Fractional k (or batch of k-points), in the same units
        ``tbmodels.Model.hamilton(k)`` expects.
    lattice_vectors : array-like, shape (3, 3)
        Real-space lattice vectors as rows (e.g. ``model.uc``).

    Returns
    -------
    np.ndarray of shape ``(3,)`` or ``(N, 3)``
        Cartesian k in rad/length, ready for ``pb.Model.set_wave_vector``.

    Example
    -------
    .. code-block:: python

        import numpy as np, pybinding as pb
        from tailwater import tb_model, k_cart_from_frac

        model = tb_model.load("wannier90_hr.hdf5")
        lat   = model.to_pb()
        pmod  = pb.Model(lat, pb.translational_symmetry())

        # Sample H(k) at Gamma → M (Bi2Se3) on a fractional path:
        k_path_frac = np.array([[0,0,0], [0.5, 0, 0]])
        bands = []
        for kf in k_path_frac:
            pmod.set_wave_vector(k_cart_from_frac(kf, model.uc))
            bands.append(np.sort(np.linalg.eigvalsh(pmod.hamiltonian.todense())))
    """
    LM = np.asarray(lattice_vectors, dtype=float)
    kf = np.asarray(k_frac, dtype=float)
    B  = 2 * np.pi * np.linalg.inv(LM)
    if kf.ndim == 1:
        return B @ kf
    return (B @ kf.T).T


class tb_model:
    """Loader namespace for the API's tight-binding HDF5 outputs.

    Usage
    -----
        from Tailwater import tb_model

        model = tb_model.load("wannier90_hr.hdf5")

        # All standard tbmodels.Model methods/attributes still work:
        bands = np.stack([model.eigenval(k) for k in k_path])
        hops  = model.hop
        size  = model.size

        # Plus three converters to other tight-binding libraries:
        pb_lat    = model.to_pb()        # pybinding.Lattice
        py_model  = model.to_pythtb()    # pythtb.tb_model
        kwant_b   = model.to_kwant()     # kwant.Builder (3D periodic)

        # All three accept an optional lattice-vector override:
        pb_lat    = model.to_pb    (lattice_vectors=np.diag([3.5, 3.5, 12.0]))
        py_model  = model.to_pythtb(lattice_vectors=np.diag([3.5, 3.5, 12.0]))
        kwant_b   = model.to_kwant (lattice_vectors=np.diag([3.5, 3.5, 12.0]))

    The returned object still passes ``isinstance(model, tbmodels.Model)``
    — we attach the converters as bound instance methods rather than
    swapping the class. Loading multiple HDF5 files in the same Python
    session is safe: each instance carries its own bindings.
    """

    @staticmethod
    def load(path_to_hdf5: str):
        """Load a tight-binding model from an HDF5 file with the ``to_pb()`` / ``to_pythtb()`` / ``to_kwant()`` converters attached.

        Parameters
        ----------
        path_to_hdf5 : str
            Path to an HDF5 file produced by the API
            (``/upload_json_process_and_download_dat/`` or extracted
            from the project bundle).

        Returns
        -------
        tbmodels.Model
            The loaded model, with instance-bound ``to_pb()``,
            ``to_pythtb()``, and ``to_kwant()`` methods for conversion
            to pybinding, PythTB, and Kwant respectively. All standard
            ``tbmodels.Model`` functionality is preserved.
        """
        if not os.path.isfile(path_to_hdf5):
            raise FileNotFoundError(f"HDF5 not found: {path_to_hdf5!r}")
        model = tbmodels.Model.from_hdf5_file(path_to_hdf5)
        # Bind the converters as instance methods — `self` is the model
        # whenever the user calls model.to_pb() / .to_pythtb() / .to_kwant().
        model.to_pb     = types.MethodType(_to_pb_method,     model)
        model.to_pythtb = types.MethodType(_to_pythtb_method, model)
        model.to_kwant  = types.MethodType(_to_kwant_method,  model)
        return model

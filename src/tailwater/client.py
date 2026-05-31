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
                             receive a zip containing the symmetrized
                             tbmodels HDF5 (produced by running
                             WannSymm on the model's prediction),
                             alongside the raw pre-symmetrization HDF5
                             and the wannsymm input/log for provenance.
                             Costs one credit; the server runs full
                             inference plus a WannSymm pass.

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
    symmetrize: bool = False,
    api_url: str = DEFAULT_API_URL,
    timeout: float = 600.0,
    save_cif: bool = True,
    keep_zip: bool = False,
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
    symmetrize : bool, default False
        Symmetrization mode. Server runs full inference, writes the
        predicted Hamiltonian as `<seedname>_hr.dat`, generates a POSCAR
        and a wannsymm.in (with projections lifted from the canonical
        .win), invokes WannSymm, and bundles the symmetrized HDF5 +
        raw HDF5 + wannsymm input/log into a single zip:
            {"symmed_hdf5": "...", "hdf5": "...", "win": "...",
             "wannsymm_in": "...", "wannsymm_log": "..."}
        Costs one credit per call. Use this when downstream
        post-processing (DOS, band structure, surface states) needs the
        Hamiltonian to obey the crystal symmetries exactly. Loses to
        `project` if both are True (project is more comprehensive).
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
          symmetrize   -> {"symmed_hdf5": "...", "hdf5": "...",
                           "win": "...", "wannsymm_in": "...",
                           "wannsymm_log": "..."}
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
    # Priority: project > symmetrize > return_input > return_embeddings
    #           > return_graph_output > full HDF5.
    # `project` wins because it's the most expensive/comprehensive
    # subspace-projection bundle. `symmetrize` sits next: it also
    # produces a multi-artifact zip (raw + symmetrized HDF5 + the
    # wannsymm input/log), but it's a different workflow (post-process,
    # not fine-tune) so it never overlaps with `project` semantically.
    # Anyone setting either flag is opting into the corresponding bundle,
    # so we shouldn't silently downgrade.
    # The server now returns a ZIP for every endpoint — the zip bundles
    # the primary artifact alongside the canonical `input.win` file that
    # was actually parsed and run through inference. The client extracts
    # the zip on receipt and returns a dict of paths (the `.win` key is
    # always present).
    if project:
        endpoint        = _ENDPOINT_PROJECT_ZIP
        primary_arcname = None       # multiple primary artifacts in this bundle
    elif symmetrize:
        endpoint        = _ENDPOINT_SYMMETRIZED_ZIP
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
    else:
        endpoint        = _ENDPOINT_FULL_HDF5
        primary_arcname = "wannier90_hr.hdf5"
    out_file_path = os.path.join(output_path, filename + ".zip")

    # ---- POST with streaming so large HDF5 / .pt files don't OOM ----
    files = {"file": ("structure.json", payload_bytes, "application/json")}
    response = requests.post(
        api_url.rstrip("/") + endpoint,
        files   = files,
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
        "wannier90_symmed_hr.hdf5":  "symmed_hdf5",
        "embeddings.pt":             "embeddings",
        "graph_output.pt":           "graph_output",
        "gnn_input.pt":              "input",
        "input.win":                 "win",
        "wannsymm.in":                "wannsymm_in",
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

    Bound as `model.to_pb` on instances returned by `tb_model.load(...)`.

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
        upstream by build_hr_model when the HDF5 was first written.

    Returns
    -------
    pb.Lattice
        with the same sublattices, on-site energies, lattice vectors,
        and hops as ``self``.

    Notes on duplicates
    -------------------
    Some tbmodels representations store both (R, i, j) and the
    auto-generated Hermitian conjugate (-R, j, i). pybinding's
    ``add_one_hopping`` rejects duplicates (it implies the H.c.
    automatically), so we wrap each insertion in try/except to silently
    skip whichever half tbmodels supplies second. The resulting pb
    Lattice contains the unique set of hops + the implied conjugates.
    """
    # Lazy import — keeps Tailwater importable on hosts without pybinding.
    import pybinding as pb

    # Real-space lattice. tbmodels stored uc=None for our API-produced
    # HDF5 files (we deliberately don't pass `uc` when building the
    # tbmodels.Model — see API/hr_export.py module docstring). In that
    # case fall back to the identity unit cell, matching the original
    # notebook's `LM = np.diag([1, 1, 1])` convention.
    if lattice_vectors is not None:
        LM = np.asarray(lattice_vectors, dtype=float)
    elif getattr(self, "uc", None) is not None:
        LM = np.asarray(self.uc, dtype=float)
    else:
        LM = np.eye(3)

    lat = pb.Lattice(a1=LM[0], a2=LM[1], a3=LM[2])

    # ---- Sublattices: position + on-site energy per orbital ----
    # `tbmodels.Model` doesn't always expose an `.on_site` attribute on
    # the loaded instance — the original constructor accepts an
    # `on_site=` kwarg but internally folds it into the diagonal of the
    # R=(0,0,0) hop matrix. To stay version-agnostic we read the
    # on-site energies straight off that diagonal. If for some reason
    # `hop[(0,0,0)]` is missing (e.g. a model with no on-site term)
    # we fall back to zeros.
    positions = np.asarray(self.pos)             # [num_orb, 3]
    num_orb   = int(positions.shape[0])

    on_site_attr = getattr(self, "on_site", None)
    if on_site_attr is not None:
        # tbmodels version that DOES expose .on_site directly.
        onsite = np.real(np.asarray(on_site_attr))
    else:
        # Fall back to the diagonal of hop[(0,0,0)].
        hop_zero = self.hop.get((0, 0, 0))
        if hop_zero is None:
            onsite = np.zeros(num_orb, dtype=float)
        else:
            if hasattr(hop_zero, "toarray"):
                hop_zero_arr = np.asarray(hop_zero.toarray())
            else:
                hop_zero_arr = np.asarray(hop_zero)
            onsite = np.real(np.diag(hop_zero_arr))

    for i in range(num_orb):
        lat.add_one_sublattice(
            str(i),
            positions[i].tolist(),
            onsite_energy=float(np.real(onsite[i])),
        )

    # ---- Hoppings: (R, orbital_i, orbital_j, complex value) ----
    # self.hop is a dict mapping R-tuple -> (num_orb, num_orb) hop
    # matrix. Some tbmodels versions store the matrix as scipy.sparse;
    # handle both dense and sparse with `.toarray()` fallback.
    for R, hop_mat in self.hop.items():
        R_arr = np.asarray(R, dtype=int)
        if hasattr(hop_mat, "toarray"):
            hop_arr = np.asarray(hop_mat.toarray())
        else:
            hop_arr = np.asarray(hop_mat)

        mag = np.abs(hop_arr)
        rows, cols = np.nonzero(mag > hop_threshold)

        R_tuple = tuple(int(x) for x in R)
        for i, j in zip(rows, cols):
            i, j = int(i), int(j)
            # On-site diagonal at R=(0,0,0) is already represented in
            # add_one_sublattice; don't re-add it here.
            if R_tuple == (0, 0, 0) and i == j:
                continue
            try:
                lat.add_one_hopping(
                    R_arr,
                    str(i), str(j),
                    complex(hop_arr[i, j]),
                )
            except Exception:
                # Duplicate H.c. half — pybinding implies the conjugate
                # from the first add, so the second raises. Silently
                # skip and continue.
                continue

    return lat


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

        # Plus a .to_pb() helper that converts to pybinding.Lattice:
        pb_lat = model.to_pb()
        # Optional override of the lattice vectors used by pb:
        pb_lat = model.to_pb(lattice_vectors=np.diag([3.5, 3.5, 12.0]))

    The returned object still passes ``isinstance(model, tbmodels.Model)``
    — we attach ``to_pb`` as a bound instance method rather than
    swapping the class. Loading multiple HDF5 files in the same Python
    session is safe: each instance carries its own ``to_pb`` binding.
    """

    @staticmethod
    def load(path_to_hdf5: str):
        """Load a tight-binding model from an HDF5 file and attach ``.to_pb()``.

        Parameters
        ----------
        path_to_hdf5 : str
            Path to an HDF5 file produced by the API
            (``/upload_json_process_and_download_dat/`` or extracted
            from the project bundle).

        Returns
        -------
        tbmodels.Model
            The loaded model, with an instance-bound ``to_pb()`` method
            for pybinding conversion. All standard ``tbmodels.Model``
            functionality is preserved.
        """
        if not os.path.isfile(path_to_hdf5):
            raise FileNotFoundError(f"HDF5 not found: {path_to_hdf5!r}")
        model = tbmodels.Model.from_hdf5_file(path_to_hdf5)
        # Bind to_pb as an instance method — `self` is the model
        # whenever the user calls model.to_pb().
        model.to_pb = types.MethodType(_to_pb_method, model)
        return model

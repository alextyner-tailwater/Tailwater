"""Customer-side fine-tuning on pre-computed backbone embeddings.

This module exposes a single function — `subspace_projection` — that
performs the end-to-end fine-tune + subspace export workflow on a
SINGLE material described by two .pt files from the API:

  * `embed_path`        — the API's
                          /upload_structure_and_download_embeddings/ output
                          (PyG Data with f_out / edge_feat + structural
                          metadata).
  * `graph_output_path` — the API's
                          /upload_structure_and_download_graph_output/ output
                          (dense edge_pred / onsite_pred = full predicted
                          Hamiltonian + structural metadata).

The graph-output predictions are attached to the embedding's gdata as
`edge_targets` so the subspace H MSE / eigenvalue losses have a target
to fit against. This is the self-distillation downfolding setup: the
trained model predicts a full Hamiltonian, and the heads are refined so
the SUBSPACE-restricted Hamiltonian reproduces the in-window eigenvalues
of that full Hamiltonian as accurately as possible.

Use it from another script as:

    from finetune_heads import subspace_projection
    subspace_projection(
        start_lr          = 5e-5,
        end_lr            = 5e-7,
        num_epochs        = 20,
        energy_range      = (-2.0, 2.0),
        decay_sigma       = 1.0,
        device            = "cpu",
        save_path         = "./customer_finetune_out",
        embed_path        = "./customer_package/embeddings.pt",
        graph_output_path = "./customer_package/graph_output.pt",
        loss_mode         = "subspace",     # default
    )

The supplier's backbone is NEVER imported, loaded, or required at any
point. Fine-tuning runs the heads on pre-computed embeddings against
the supplier-provided target Hamiltonians (or, if the customer has
their own Hamiltonians keyed by material id, they can substitute those
— see the "Bring-Your-Own-Targets" note at the bottom of this file).

Three loss modes
----------------
"full"     : standard H MSE loss across all orbitals. Requires target
             Hamiltonians on each Data object (gdata.edge_targets).
"subspace" : energy-windowed subspace fine-tuning (see
             finetune_subspace.py). Requires target Hamiltonians and
             trains the heads to fit eigenvalues that lie in
             [E_LO, E_HI] of the full target spectrum.
"eig_only" : eigenvalue-only fine-tuning. The customer provides
             target eigenvalues at custom k-points; no target
             Hamiltonian is needed. Each material's Data object must
             carry attributes gdata.kpts, gdata.target_eigs,
             gdata.target_eigs_mask (use
             subspace_utils.make_eigenvalue_only_data to attach them).
             The number of valid target eigenvalues at each k defines
             the effective subspace size at that k.

Per-epoch the function prints the mean per-material eigenvalue loss
(for "subspace" / "eig_only" modes). For "full" mode it prints the
total H MSE since there is no separate eigenvalue term.

Outputs (written to `save_path`)
--------------------------------
HeadsFT_final.pth                 fine-tuned heads checkpoint + metadata
{embedding_stem}_pred.hdf5        per-material projected tbmodels.Model
                                  (subspace-restricted to orbitals whose
                                  PREDICTED onsite diagonal lies inside
                                  the energy window).
{embedding_stem}.basis.json       per-material basis-info JSON describing
                                  the projected subspace (orbital labels,
                                  per-atom positions, energy window, etc.)
"""

import os
import gc
import time
from typing import Tuple

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from .heads_only_model import load_heads_only_checkpoint
from .subspace_utils import (
    NeighBrs,
    Subspace_H_MSE_Loss,
    Subspace_EigLoss,
    Eigenvalue_Only_Loss,
    build_subspace_active_mask,
    write_subspace_basis_file,
    make_eigenvalue_only_data,        # re-exported for customer convenience
)
from .hr_export import build_hr_model_fast, write_hr_output


# The MACE-compatible HeadsOnly checkpoint bundled with the package. Used as
# the default starting checkpoint for `subspace_projection` so customers don't
# have to source a HeadsOnly.pth themselves. Built from the production
# WanE3MACE checkpoint via `API/make_heads_only.py`; the older `HeadsOnly.pth`
# at the repo root is from the retired WanE3Lite backbone and IS NOT
# compatible with embeddings the API now returns.
_DEFAULT_HEADS_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "HeadsOnly_MACE.pth",
)


# =====================================================
# INTERNAL HELPERS
# =====================================================

def _load_payload(path: str):
    """Normalize a .pt embedding file into (list_of_Data, raw_payload).

    Two on-disk formats are supported:
      * dict with keys {"data", "LM", "atoms", "irreps_in"} — the
        per-material format the API endpoint
        /upload_structure_and_download_embeddings/ produces.
      * list of PyG Data objects — the multi-material format
        extract_embeddings.py writes per shard.

    The raw payload is returned alongside the dataset so callers that
    need the per-material LM / atoms metadata (e.g. for the final TB
    model build) can pull it from the dict version.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "data" in payload:
        return [payload["data"]], payload
    if isinstance(payload, list):
        return payload, None
    return [payload], None


def _full_h_mse(gdata, edge_pred, onsite_pred):
    """Plain H MSE across all orbitals — same form as train_wider.py."""
    is_self_loop = (gdata.edge_vectors.norm(dim=-1) == 0)
    weights = torch.sign(torch.abs(gdata.edge_vectors.norm(dim=1))).view(-1, 1, 1, 1)
    target = gdata.edge_targets[:, 0, :, :, :]
    loss_edge   = ((weights * (edge_pred - target)) ** 2).sum()
    loss_onsite = ((onsite_pred - target[is_self_loop]) ** 2).sum()
    return (loss_onsite + 10.0 * loss_edge) / gdata.node_features.shape[0]


def _build_kgrid(kgrid_n: int) -> torch.Tensor:
    """N^3 cubic k-grid in fractional coordinates, used by the subspace
    eigenvalue loss."""
    _kvals = torch.arange(kgrid_n, dtype=torch.float32) / kgrid_n
    kx, ky, kz = torch.meshgrid(_kvals, _kvals, _kvals, indexing="ij")
    return torch.stack([kx, ky, kz], dim=-1).reshape(-1, 3)


# =====================================================
# PUBLIC API
# =====================================================

def subspace_projection(
    start_lr: float,
    end_lr: float,
    num_epochs: int,
    energy_range: Tuple[float, float],
    decay_sigma: float,
    device,
    save_path: str,
    embed_path: str,
    graph_output_path: str,
    loss_mode: str = "subspace",
    *,
    heads_checkpoint: str = None,
    kgrid_n: int = 4,
    h_mse_weight: float = 0.001,
    eig_weight: float = 1.0,
) -> str:
    """Fine-tune the heads to project a single material into an energy subspace.

    Single-material workflow. The customer hits the API twice for the
    same structure:

      * /upload_structure_and_download_embeddings/ -> `embed_path`     (.pt)
            backbone features f_out / edge_feat + structural metadata
      * /upload_structure_and_download_graph_output/ -> `graph_output_path` (.pt)
            full predicted Hamiltonian (edge_pred / onsite_pred) +
            structural metadata

    The graph-output predictions are attached to the embedding's PyG
    Data object as `edge_targets` — i.e. the model's pre-fine-tune
    output is the "ground truth" the subspace fine-tune fits against.
    This is the self-distillation downfolding setup: the trained model
    predicts a full Hamiltonian, and the heads are refined so the
    SUBSPACE-restricted Hamiltonian reproduces the in-window eigenvalues
    of that full Hamiltonian as accurately as possible.

    The LR is cosine-annealed from `start_lr` down to `end_lr` over
    `num_epochs` epochs. After every epoch the mean eigenvalue loss for
    the (single) material is printed.

    Outputs written to `save_path`
    ------------------------------
      HeadsFT_final.pth              fine-tuned heads weights + metadata
      {stem}_pred.hdf5               projected tbmodels.Model
                                     (SUBSPACE-restricted to orbitals
                                     whose predicted onsite-diagonal
                                     lies inside the energy window)
      {stem}.basis.json              basis-info file mapping subspace
                                     indices to (atom, spatial, spin)
                                     labels.

    where `{stem}` is the basename of `embed_path` without extension.

    Parameters
    ----------
    start_lr           : initial LR for AdamW.
    end_lr             : minimum LR at the end of cosine decay.
    num_epochs         : number of training epochs.
    energy_range       : (e_lo, e_hi) tuple, eV, relative to E_F = 0.
    decay_sigma        : Gaussian sigma for eigenvalue weights around
                         the window center.
    device             : torch.device or str.
    save_path          : output directory (created if missing).
    embed_path         : path to ONE .pt embedding file (API-format
                         dict with `data`, `LM`, `atoms`, `irreps_in`).
    graph_output_path  : path to ONE .pt graph-output file (API-format
                         dict with `edge_pred`, `onsite_pred`, `data`,
                         `LM`, `atoms`). This is the FULL model output
                         that the projection refines toward.
    loss_mode          : "subspace" (default) | "full" | "eig_only".
    heads_checkpoint   : starting HeadsOnly checkpoint (path to a .pth).
                         Defaults to the MACE-compatible `HeadsOnly_MACE.pth`
                         that ships inside the installed package, so you don't
                         need to supply one for the standard workflow. Pass an
                         explicit path only if you're starting from a
                         custom-fine-tuned heads checkpoint.

    Returns
    -------
    str   path to the saved HeadsFT_final.pth checkpoint.
    """
    e_lo, e_hi = float(energy_range[0]), float(energy_range[1])

    os.makedirs(save_path, exist_ok=True)
    device = torch.device(device) if isinstance(device, str) else device

    # =====================================================
    # LOAD EMBEDDING + GRAPH-OUTPUT PAYLOADS
    # =====================================================
    if not os.path.isfile(embed_path):
        raise FileNotFoundError(f"embed_path does not point to a file: {embed_path!r}")
    if not os.path.isfile(graph_output_path):
        raise FileNotFoundError(
            f"graph_output_path does not point to a file: {graph_output_path!r}"
        )

    embed_pkg = torch.load(embed_path,        map_location="cpu", weights_only=False)
    gout_pkg  = torch.load(graph_output_path, map_location="cpu", weights_only=False)

    # The embedding payload is the API's dict-format. We need its `data`
    # (PyG Data with f_out / edge_feat) for the forward pass; LM and
    # atoms are needed for the final TB-model + basis export.
    if not (isinstance(embed_pkg, dict) and "data" in embed_pkg):
        raise ValueError(
            f"embed_path must point to a dict-format .pt file with a "
            f"`data` key (the format produced by the API's "
            f"/upload_structure_and_download_embeddings/ endpoint). "
            f"Got top-level type {type(embed_pkg).__name__}."
        )
    gdata = embed_pkg["data"]
    LM    = embed_pkg.get("LM",    gout_pkg.get("LM"))
    atoms = embed_pkg.get("atoms", gout_pkg.get("atoms"))
    if LM is None or atoms is None:
        raise ValueError(
            "Could not resolve `LM` and `atoms` from either payload; "
            "make sure at least one of the two .pt files is API-format "
            "(contains those keys)."
        )

    if not (isinstance(gout_pkg, dict)
            and "edge_pred" in gout_pkg and "onsite_pred" in gout_pkg):
        raise ValueError(
            "graph_output_path must point to a dict-format .pt file with "
            "`edge_pred` and `onsite_pred` keys (the format produced by "
            "the API's /upload_structure_and_download_graph_output/ "
            "endpoint)."
        )

    # =====================================================
    # ATTACH FULL-MODEL OUTPUT AS edge_targets ON gdata
    # =====================================================
    # Subspace_H_MSE_Loss and Subspace_EigLoss read gdata.edge_targets
    # to compute both the H MSE term and the FULL target H(k) that the
    # subspace prediction is compared against. We populate edge_targets
    # from the API's full graph output, with onsite_pred substituted
    # into self-loop slots (same convention as build_hr_model).
    edge_pred_full   = gout_pkg["edge_pred"  ].reshape(gdata.edge_index.shape[1],     18, 18, 2)
    onsite_pred_full = gout_pkg["onsite_pred"].reshape(gdata.node_features.shape[0], 18, 18, 2)

    is_self_loop = (gdata.edge_vectors.norm(dim=-1) == 0)
    weights      = torch.sign(torch.abs(gdata.edge_vectors.norm(dim=1))).view(-1, 1, 1, 1)
    targets_inline = (weights * edge_pred_full).clone()
    targets_inline[is_self_loop] = onsite_pred_full

    # Subspace losses expect edge_targets shape [num_edges, 1, 18, 18, 2]
    # (the leading 1 is a legacy K-axis they index with [:, 0, ...]).
    gdata.edge_targets = targets_inline.unsqueeze(1)

    print(f"[data] {os.path.basename(embed_path)}  "
          f"atoms={gdata.node_features.shape[0]}, "
          f"edges={gdata.edge_index.shape[1]}")
    print(f"[data] graph-output attached as edge_targets "
          f"(self-distillation downfolding)")

    # =====================================================
    # LOAD HEADS + SETUP OPTIMIZER
    # =====================================================
    if heads_checkpoint is None:
        heads_checkpoint = _DEFAULT_HEADS_CHECKPOINT
        print(f"[model] using packaged HeadsOnly checkpoint: {heads_checkpoint}")
    model = load_heads_only_checkpoint(heads_checkpoint,
                                       map_location=device).to(device)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[model] HeadsOnly loaded — {n_total:,} trainable params, "
          f"irreps_in={model.irreps_in}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=start_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max  = max(1, num_epochs),
        eta_min = end_lr,
    )

    KVECS = _build_kgrid(kgrid_n)

    # Move the (single) gdata to device once; reuse across all epochs.
    gdata = gdata.to(device)

    # =====================================================
    # TRAINING LOOP  (single material, one step per epoch)
    # =====================================================
    train_loss_history = []
    eig_loss_history   = []
    start_time = time.perf_counter()

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        try:
            edge_pred, onsite_pred = model(gdata)
            edge_pred   = edge_pred.reshape(gdata.edge_index.shape[1], 18, 18, 2)
            onsite_pred = onsite_pred.reshape(gdata.node_features.shape[0], 18, 18, 2)

            if loss_mode == "subspace":
                hmse = Subspace_H_MSE_Loss(
                    gdata, edge_pred, onsite_pred, e_lo, e_hi,
                )
                eig = Subspace_EigLoss(
                    gdata, edge_pred, onsite_pred,
                    KVECS, NeighBrs, e_lo, e_hi,
                    decay_sigma=decay_sigma,
                )
                loss      = h_mse_weight * hmse + eig_weight * eig
                eig_value = float(eig.item())
            elif loss_mode == "eig_only":
                eig = Eigenvalue_Only_Loss(
                    gdata, edge_pred, onsite_pred,
                    e_lo, e_hi,
                    decay_sigma=decay_sigma,
                )
                loss      = eig
                eig_value = float(eig.item())
            elif loss_mode == "full":
                loss      = _full_h_mse(gdata, edge_pred, onsite_pred)
                eig_value = float("nan")
            else:
                raise ValueError(
                    f"Unknown loss_mode={loss_mode!r}; "
                    f"expected 'subspace', 'eig_only', or 'full'."
                )

            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                loss_value = float(loss.item())
            else:
                # NaN/Inf — don't step, but advance the scheduler so the
                # cosine schedule still completes in the prescribed
                # num_epochs.
                print(f"  [warn] non-finite loss at epoch {epoch + 1}; skipping step")
                loss_value = float("nan")

        except torch.cuda.OutOfMemoryError:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            loss_value = float("nan")
            eig_value  = float("nan")

        scheduler.step()
        train_loss_history.append(loss_value)
        eig_loss_history.append(eig_value)

        lr_now = optimizer.param_groups[0]["lr"]
        if loss_mode in ("subspace", "eig_only"):
            print(f"Epoch {epoch + 1:>3d}/{num_epochs}  "
                  f"LR {lr_now:.3e}  "
                  f"Eig loss {eig_value:.6e}")
        else:
            # "full" mode has no separate eigenvalue loss; report the
            # H MSE in its place so the print line stays informative.
            print(f"Epoch {epoch + 1:>3d}/{num_epochs}  "
                  f"LR {lr_now:.3e}  "
                  f"H MSE {loss_value:.6e}  "
                  f"(loss_mode='full' has no eigenvalue term)")

    # =====================================================
    # SAVE FINE-TUNED HEADS CHECKPOINT
    # =====================================================
    heads_ft_path = os.path.join(save_path, "HeadsFT_final.pth")
    torch.save(
        {
            "heads_state_dict":   model.state_dict(),
            "irreps_in":          str(model.irreps_in),
            "loss_mode":          loss_mode,
            "energy_range":       (e_lo, e_hi),
            "decay_sigma":        decay_sigma,
            "start_lr":           start_lr,
            "end_lr":             end_lr,
            "num_epochs":         num_epochs,
            "train_loss_history": train_loss_history,
            "eig_loss_history":   eig_loss_history,
        },
        heads_ft_path,
    )
    print(f"\n[save] HeadsFT checkpoint -> {heads_ft_path}")

    # =====================================================
    # BUILD PROJECTED SUBSPACE TB MODEL + BASIS JSON
    # =====================================================
    model.eval()
    with torch.no_grad():
        edge_pred, onsite_pred = model(gdata)
    edge_pred_r   = edge_pred.reshape(gdata.edge_index.shape[1], 18, 18, 2)
    onsite_pred_r = onsite_pred.reshape(gdata.node_features.shape[0], 18, 18, 2)

    # Subspace mask: orbitals whose predicted onsite diagonal lies in
    # [e_lo, e_hi]. Combined with the structural orbital-active flags
    # already in node_features[:, 109:127].
    active = build_subspace_active_mask(
        gdata.node_features, onsite_pred_r, e_lo, e_hi,
    )

    # Stamp the subspace mask into a clone of gdata.node_features so
    # build_hr_model_fast restricts the TB model to subspace orbitals.
    gdata_sub = gdata.clone()
    gdata_sub.node_features = gdata.node_features.clone()
    gdata_sub.node_features[:, 109:127] = active.float()

    hr_model = build_hr_model_fast(
        edge_pred   = edge_pred_r,
        onsite_pred = onsite_pred_r,
        gdata       = gdata_sub,
        LM          = LM,
        atoms       = atoms,
    )

    stem       = os.path.splitext(os.path.basename(embed_path))[0]
    hr_path    = os.path.join(save_path, f"{stem}_pred.hdf5")
    basis_path = os.path.join(save_path, f"{stem}.basis.json")

    write_hr_output(hr_model, hr_path, fmt="hdf5")

    # Predicted onsite diagonal per (atom, orbital) — feeds the
    # `onsite_energy_eV` field in the basis JSON.
    pred_diag = onsite_pred_r[:, :, :, 0].detach().cpu().numpy()
    diag_grid = np.stack(
        [np.diag(pred_diag[i]) for i in range(pred_diag.shape[0])],
        axis=0,
    )

    write_subspace_basis_file(
        out_path        = basis_path,
        active_mask     = active,
        atoms           = atoms,
        LM              = np.asarray(LM),
        energy_window   = (e_lo, e_hi),
        onsite_energies = diag_grid,
        extra_metadata  = {
            "loss_mode":          loss_mode,
            "decay_sigma":        decay_sigma,
            "heads_checkpoint":   os.path.basename(heads_ft_path),
            "source_embedding":   os.path.basename(embed_path),
            "source_graph_output": os.path.basename(graph_output_path),
        },
    )

    print(f"[save] projected hr-model -> {hr_path}")
    print(f"[save] basis info         -> {basis_path}")

    elapsed = time.perf_counter() - start_time
    print(f"\nTotal fine-tune + export time: {elapsed:.1f}s")
    return heads_ft_path


# =====================================================
# STAND-ALONE SCRIPT ENTRY POINT
# =====================================================
# Running `python finetune_heads.py` directly executes the function
# with the same defaults the legacy script used. Customers who want a
# library-only API can ignore this and call `subspace_projection`
# directly from their own code.

if __name__ == "__main__":
    HEADS_CHECKPOINT  = "customer_package/HeadsOnly.pth"
    EMBED_PATH        = "customer_package/embeddings.pt"
    GRAPH_OUTPUT_PATH = "customer_package/graph_output.pt"
    SAVE_PATH         = "customer_finetune_out"

    E_LO, E_HI  = -2.0, +2.0
    DECAY_SIGMA = (E_HI - E_LO) / 4.0

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    subspace_projection(
        start_lr          = 5e-5,
        end_lr            = 5e-7,
        num_epochs        = 20,
        energy_range      = (E_LO, E_HI),
        decay_sigma       = DECAY_SIGMA,
        device            = DEVICE,
        save_path         = SAVE_PATH,
        embed_path        = EMBED_PATH,
        graph_output_path = GRAPH_OUTPUT_PATH,
        loss_mode         = "subspace",
        heads_checkpoint  = HEADS_CHECKPOINT,
    )


# -------- Bring-Your-Own-Targets note --------
#
# If a customer has their own DFT/Wannier targets (e.g. they re-ran a
# Wannier90 step with different disentanglement windows), they can swap
# in their targets at training time without re-running the backbone:
#
#   pkg = torch.load("./embeddings.pt", weights_only=False)
#   gdata = pkg["data"]
#   gdata.edge_targets = lookup_my_targets(gdata.idn)   # [E, 1, 18, 18, 2]
#   torch.save(pkg, "./embeddings.pt")
#
# As long as the (edge_index, inv_data) topology matches the original
# extraction, subspace_projection() will fine-tune against the
# customer's targets using the same supplier-provided embeddings.
#
# -------- Eigenvalue-only mode --------
#
# If the customer only has DFT band-structure data (eigenvalues at a
# set of k-points) and NO Hamiltonian targets, attach them once and
# then call subspace_projection with loss_mode="eig_only":
#
#   from subspace_utils import make_eigenvalue_only_data
#   pkg = torch.load("./embeddings.pt", weights_only=False)
#   make_eigenvalue_only_data(
#       pkg["data"], my_dft_kpts, my_dft_eigs_per_k,
#       e_lo=-2.0, e_hi=2.0,
#   )
#   torch.save(pkg, "./embeddings.pt")
#
# Each material's subspace size at each k is then implicitly the number
# of provided in-window eigenvalues at that k — the loss picks exactly
# that many predicted bands (the ones closest to the window center)
# and matches them to the targets. No orbital mask needed.

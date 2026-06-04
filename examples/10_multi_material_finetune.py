"""Multi-material heads-only fine-tune against user Wannier targets.

Refines the API heads on a *set* of materials whose ground-truth
Wannier Hamiltonians the user computed themselves (the complement of
single-material `subspace_projection`, which self-distills against
the API's own full-model output).

Layout convention
-----------------

Organise the user's data on disk as one subdirectory per material::

    datasets/
    ├── train/
    │   ├── Bi2Se3/
    │   │   ├── wannier90.win        # the user's Wannier90 input
    │   │   └── wannier90_hr.dat     # the user's Wannier hr-file (or `_hr.hdf5`)
    │   ├── Bi2Te3/
    │   │   └── ...
    │   └── ...
    └── val/
        └── ...

Set `GENERATE_EMBEDDING = True` below and the function calls the
API once per subdirectory to populate the `embeddings.pt`
automatically — the Structure handed to the API is reconstructed
from each subdirectory's own .win file.  Existing embeddings are
reused, so re-runs don't burn extra credits.

If you already have your embeddings in place (e.g. you saved them
manually from a previous API run), set
`GENERATE_EMBEDDING = False` and the function just discovers the
existing files.

The .win projection block (e.g. `Bi: s, p, d` / `Se: s, p`) plus the
`atoms_cart` block together fully determine the per-atom orbital
layout; the .win's `fermi_energy` keyword is auto-subtracted from
on-site energies so every target sits at E_F = 0.
"""

import os

# macOS-conda quirk: PyTorch's bundled libomp and the libomp used by
# numpy / MKL / matplotlib can clash and SIGSEGV when both are alive
# in the same process — the symptom is a silent crash *during* the
# band-structure plot at the end of this script. The OMP error message
# itself points at the right escape hatch:
#     `KMP_DUPLICATE_LIB_OK=TRUE`
# Setting it via `os.environ` BEFORE any torch import propagates into
# libomp's init. We also use `threadpoolctl` further down to pin BLAS
# to a single thread for the duration of the band plot — together
# these two guards keep the in-process pipeline clean on macOS without
# affecting Linux runs (where the conflict doesn't exist).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Use a non-interactive matplotlib backend so the band figure renders
# cleanly even on headless servers / in batch contexts.
import matplotlib                                    # noqa: E402
matplotlib.use("Agg")                                # noqa: E402

import numpy as np                                   # noqa: E402
import torch                                         # noqa: E402

import tailwater                                     # noqa: E402
from tailwater import (                              # noqa: E402
    prepare_finetune_targets_from_directory,
    finetune_heads_multi,
    # for the post-training inference + band-structure demo at the bottom:
    load_heads_only_checkpoint,
    build_hr_model_fast,
    write_hr_output,
    tb_model,
    bulk_band_structure,
    align_to_vbm,
    parse_win_fermi_energy,
)
import tbmodels                                      # noqa: E402
import matplotlib.pyplot as plt                      # noqa: E402


# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------
TRAIN_DIR = "datasets/train"            # one subdirectory per training material
VAL_DIR   = "datasets/val"              # one subdirectory per validation material
CACHE_DIR = "finetune_out/cache"        # per-material prepared targets get saved here
SAVE_DIR  = "finetune_out"              # final checkpoint lands here

ENERGY_RANGE = (-2.0, 2.0)              # eV — eigenvalues outside this window are masked
DEVICE       = "cpu"                     # use "cuda" if available

GENERATE_EMBEDDING = True               # call the API to generate missing embeddings
API_USER           = "your-username"    # only used if GENERATE_EMBEDDING
API_PASSWORD       = "your-password"    # only used if GENERATE_EMBEDDING


class _NullCtx:
    """No-op context manager used as a stand-in when threadpoolctl isn't installed."""
    def __enter__(self): return self
    def __exit__(self, *exc): return False


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1)  Auto-discover (embedding, hr, .win) triples in the train + val
    #     directories.  The function walks each subdirectory, finds the
    #     three files by glob pattern, parses the .win for the per-atom
    #     orbital layout, and prepares the PyG `Data`-with-targets
    #     object.  Each prepared item is also cached as a .pt under
    #     `CACHE_DIR` so re-running this script is cheap.
    # ------------------------------------------------------------------
    common = dict(
        out_dir            = CACHE_DIR,
        strict             = False,        # skip subdirectories missing any of the 3 files
        generate_embedding = GENERATE_EMBEDDING,
        user               = API_USER     if GENERATE_EMBEDDING else None,
        password           = API_PASSWORD if GENERATE_EMBEDDING else None,
    )

    train_items = prepare_finetune_targets_from_directory(TRAIN_DIR, **common)
    val_items   = prepare_finetune_targets_from_directory(VAL_DIR,   **common)

    # ------------------------------------------------------------------
    # 2)  Multi-material fine-tune
    # ------------------------------------------------------------------
    final_ckpt = finetune_heads_multi(
        train_targets   = train_items,
        val_targets     = val_items or None,
        start_lr        = 5e-5,
        end_lr          = 5e-7,
        num_epochs      = 50,
        energy_range    = ENERGY_RANGE,
        decay_sigma     = 1.0,
        device          = DEVICE,
        save_path       = SAVE_DIR,
        val_every       = 5,
        loss_mode       = "subspace",
        # heads_checkpoint=None  -> use the packaged HeadsOnly_MACE.pth
        kgrid_n         = 4,
    )

    print(f"\nDone. Final checkpoint: {final_ckpt}")

    # ------------------------------------------------------------------
    # 3)  Use the fine-tuned heads to build a model + plot bands for a
    #     held-out validation material.
    # ------------------------------------------------------------------
    #     This is the inference workflow you'd run any time you want a
    #     Hamiltonian out of the fine-tuned model: load the heads, hand
    #     them the API embedding for the target structure, assemble the
    #     resulting predictions into a tbmodels.Model via
    #     build_hr_model_fast, and post-process as usual.
    #
    #     The embedding can come from one of two places:
    #       (a) The .pt that prepare_finetune_targets_from_directory
    #           already cached inside the validation subdirectory
    #           (`{val_subdir}/embeddings.pt`) — this is the cheap path,
    #           reuses the existing API call.
    #       (b) A fresh API call with `tw_api_call(..., return_embeddings=True)`
    #           for a structure you don't have an embedding for yet.
    #     The example uses path (a) since the directory walker already
    #     produced an embedding for every validation material.
    # ------------------------------------------------------------------
    if not val_items:
        print("No validation materials configured — skipping band plot.")
        return
    val_item   = val_items[0]
    val_subdir = os.path.join(VAL_DIR, val_item["name"])
    print(f"\nBuilding three-way band comparison for '{val_item['name']}':")
    print(f"  - target     : the user's own hr-file (Fermi-shifted to E_F = 0)")
    print(f"  - pre-tune   : packaged HeadsOnly_MACE.pth, no fine-tune applied")
    print(f"  - post-tune  : HeadsFT_multi_best.pth from this run")

    # ------------------------------------------------------------------
    # 3a)  Load the API embedding once — both head checkpoints get it.
    # ------------------------------------------------------------------
    emb_pkg = torch.load(val_item["embed_path"], map_location=DEVICE,
                         weights_only=False)
    gdata   = emb_pkg["data"].to(DEVICE)
    LM      = np.asarray(emb_pkg["LM"], dtype=float)
    atoms   = emb_pkg["atoms"]

    # ------------------------------------------------------------------
    # 3b)  Build the three tbmodels.Model objects to compare.
    # ------------------------------------------------------------------
    # (1) Target: the user's own hr-file.  Apply the .win's
    #     `fermi_energy` to put E_F = 0, matching what the multi-finetune
    #     loss saw during training.
    win_path = os.path.join(val_subdir, "wannier90.win")
    if not os.path.isfile(win_path):
        # fall back to whatever .win is in the directory (e.g. input.win)
        win_path = [os.path.join(val_subdir, n) for n in os.listdir(val_subdir)
                    if n.lower().endswith(".win")][0]
    hr_path = [os.path.join(val_subdir, n) for n in os.listdir(val_subdir)
               if n.lower().endswith(("_hr.dat", "_hr.hdf5", "_hr.h5"))
               and "finetuned" not in n][0]
    if hr_path.lower().endswith((".hdf5", ".h5")):
        target_model = tbmodels.Model.from_hdf5_file(hr_path)
    else:
        target_model = tbmodels.Model.from_wannier_files(
            hr_file = hr_path, win_file = win_path, pos_kind = "nearest_atom",
        )
    ef = parse_win_fermi_energy(win_path)
    if ef is not None:
        target_model = align_to_vbm(target_model, fermi_level=ef)
        print(f"  target Fermi shift: -{ef:+.4f} eV (from {os.path.basename(win_path)})")

    # (2) Pre-finetune: the packaged HeadsOnly_MACE.pth (no training).
    default_ckpt = os.path.join(
        os.path.dirname(tailwater.__file__), "HeadsOnly_MACE.pth",
    )
    heads_pre  = load_heads_only_checkpoint(default_ckpt, map_location=DEVICE).to(DEVICE).eval()
    with torch.no_grad():
        edge_pre, onsite_pre = heads_pre(gdata)
    pretune_model = build_hr_model_fast(edge_pre, onsite_pre, gdata, LM, atoms)

    # (3) Post-finetune: the best-val (or final) HeadsFT_multi checkpoint.
    best_ckpt = os.path.join(SAVE_DIR, "HeadsFT_multi_best.pth")
    use_ckpt  = best_ckpt if os.path.isfile(best_ckpt) else final_ckpt
    heads_post = load_heads_only_checkpoint(use_ckpt, map_location=DEVICE).to(DEVICE).eval()
    with torch.no_grad():
        edge_post, onsite_post = heads_post(gdata)
    posttune_model = build_hr_model_fast(edge_post, onsite_post, gdata, LM, atoms)

    # Save the post-finetune model for downstream use.
    pred_hr_path = os.path.join(val_subdir, f"{val_item['name']}_finetuned_hr.hdf5")
    write_hr_output(posttune_model, pred_hr_path, fmt="hdf5")
    print(f"  wrote post-finetune hr-model → {pred_hr_path}  "
          f"({posttune_model.size} orbitals)")

    # ------------------------------------------------------------------
    # 3c)  Compute bands for all three models on the same k-path and
    #      overlay them on one figure for direct visual comparison.
    # ------------------------------------------------------------------
    # Generic Γ → M → K → Γ path; adjust for your crystal class.
    band_path   = [[0.0, 0.0, 0.0],
                   [0.5, 0.0, 0.0],
                   [0.333, 0.333, 0.0],
                   [0.0, 0.0, 0.0]]
    band_labels = [r"$\Gamma$", "M", "K", r"$\Gamma$"]

    # `threadpoolctl.threadpool_limits(1)` is the macOS-specific guard
    # against the OpenMP runtime clash discussed at the top of this
    # file. No-op on Linux.
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        threadpool_limits = None
        print("  [warn] `threadpoolctl` is not installed; on macOS the band "
              "plot below may segfault. Install with `pip install "
              "threadpoolctl`.")

    plot_ctx = threadpool_limits(limits=1) if threadpool_limits else _NullCtx()
    with plot_ctx:
        # Get raw band data (k_dist + eigenvalues) for each model.
        models_styled = [
            (target_model,    "target",      "k",       "-",   1.6, 0.9),
            (pretune_model,   "pre-tune",    "tab:blue","--",  1.0, 0.7),
            (posttune_model,  "post-tune",   "tab:red", "-",   1.1, 0.85),
        ]
        results = []
        for m, _lbl, _c, _ls, _lw, _al in models_styled:
            results.append(bulk_band_structure(
                m, k_points=band_path, k_labels=band_labels,
                e_range=ENERGY_RANGE, spacing=0.02, verbose=False,
                return_raw=True,
            ))

        fig, ax = plt.subplots(figsize=(8.5, 6))
        for (_, label, color, ls, lw, alpha), res in zip(models_styled, results):
            kd   = res.k_dist
            eigs = res.eigenvalues                  # (N_path, num_bands)
            # First band carries the label; subsequent bands re-use color/style.
            ax.plot(kd, eigs[:, 0], color=color, ls=ls, lw=lw, alpha=alpha, label=label)
            for b in range(1, eigs.shape[1]):
                ax.plot(kd, eigs[:, b], color=color, ls=ls, lw=lw, alpha=alpha)
        ax.set_xticks(results[0].k_node)
        ax.set_xticklabels(results[0].k_labels)
        for x in results[0].k_node:
            ax.axvline(x, ls=":", color="0.7", lw=0.5)
        ax.axhline(0.0, ls=":", color="r",   lw=0.5)
        ax.set_xlim(results[0].k_dist[0], results[0].k_dist[-1])
        ax.set_ylim(*ENERGY_RANGE)
        ax.set_ylabel(r"$E - E_F$ (eV)")
        ax.set_title(f"Band-structure comparison — {val_item['name']}")
        ax.legend(frameon=False, loc="best")
        fig.tight_layout()

        bands_png = os.path.join(val_subdir, f"{val_item['name']}_bands_comparison.png")
        fig.savefig(bands_png, dpi=180)
    print(f"  wrote band-structure comparison → {bands_png}")


# ----------------------------------------------------------------------
# Alternative inference path: call the API afresh for a brand-new
# structure (one you have not yet generated an embedding for). Most of
# the time you don't need this — the directory walker already produced
# embeddings for every validation material — but it's the recipe to
# follow when the structure is brand new.
# ----------------------------------------------------------------------
def inference_from_fresh_api_call_example():
    """Reference snippet — not invoked by main()."""
    from tailwater import tw_api_call, structure_from_win
    structure = structure_from_win("path/to/new_material.win")
    response  = tw_api_call(
        structure, user="...", password="...",
        output_path="./new_material",
        filename="embeddings",
        return_embeddings=True,
    )
    emb_pkg = torch.load(response["embeddings"],
                         map_location="cpu", weights_only=False)
    heads = load_heads_only_checkpoint(
        "finetune_out/HeadsFT_multi_best.pth", map_location="cpu",
    ).eval()
    with torch.no_grad():
        edge_pred, onsite_pred = heads(emb_pkg["data"])
    model = build_hr_model_fast(
        edge_pred, onsite_pred, emb_pkg["data"],
        np.asarray(emb_pkg["LM"]), emb_pkg["atoms"],
    )
    return model


if __name__ == "__main__":
    main()

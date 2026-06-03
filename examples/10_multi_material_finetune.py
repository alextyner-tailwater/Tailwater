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
from tailwater import (
    prepare_finetune_targets_from_directory,
    finetune_heads_multi,
)


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
    print(
        "\nNext: pass this checkpoint to any inference workflow that\n"
        "loads a HeadsOnly model.  For instance, in subspace_projection:\n"
        f'    subspace_projection(..., heads_checkpoint="{final_ckpt}")\n'
        "or directly via tailwater.load_heads_only_checkpoint(...)."
    )


if __name__ == "__main__":
    main()

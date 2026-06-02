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
    │   │   ├── embeddings.pt        # from `/upload_structure_and_download_embeddings/`
    │   │   ├── wannier90.win        # the user's Wannier90 input
    │   │   └── wannier90_hr.dat     # the user's Wannier hr-file (or `_hr.hdf5`)
    │   ├── Bi2Te3/
    │   │   ├── embeddings.pt
    │   │   ├── wannier90.win
    │   │   └── wannier90_hr.dat
    │   └── ...
    └── val/
        ├── Sb2Te3/
        │   └── ...
        └── ...

Then a single call discovers every triple, parses the .win to derive
per-atom orbital layouts, and prepares the per-material training
items.  The subdirectory name becomes the material name in training
logs and the cached `.pt` filename.

The .win projection block (e.g. `Bi: s, p, d` / `Se: s, p`) plus the
`atoms_cart` block together fully determine the per-atom orbital
layout — nothing else needed.
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
    train_items = prepare_finetune_targets_from_directory(
        TRAIN_DIR,
        out_dir = CACHE_DIR,
        strict  = False,        # skip subdirectories missing any of the 3 files
    )

    val_items = prepare_finetune_targets_from_directory(
        VAL_DIR,
        out_dir = CACHE_DIR,
        strict  = False,
    )

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

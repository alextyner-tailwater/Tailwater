"""Multi-material heads-only fine-tune against user Wannier targets.

Refines the API heads on a *set* of materials whose ground-truth
Wannier Hamiltonians the user computed themselves (the complement of
single-material `subspace_projection`, which self-distills against
the API's own full-model output).

Workflow per material
---------------------
  1. Call the API once to get the backbone embedding `.pt`
     (the file produced by `/upload_structure_and_download_embeddings/`).
  2. Have the matching Wannier hr-file (`_hr.dat` or `_hr.hdf5`) AND
     the `.win` file you fed Wannier90.  The .win's projection block
     (e.g. `Bi: s, p, d` / `Se: s, p`) plus the `atoms_cart` block
     fully determine the per-atom orbital layout — nothing else needed.
  3. Call `prepare_finetune_target(embed_path, hr_path, win_path)`.
     The per-atom active mask is parsed straight out of the .win
     (the same convention the API uses server-side), so you never
     need to type orbital lists by hand.

Then call `finetune_heads_multi` on the list of prepared items.

Example below uses one material as both training and validation just
to keep the script self-contained — replace `TRAIN_INPUTS` /
`VAL_INPUTS` with your real list.
"""

import os
from tailwater import (
    prepare_finetune_target,
    finetune_heads_multi,
)


# ----------------------------------------------------------------------
# Per-material inputs.  In a real run you'd have N tuples here.
# ----------------------------------------------------------------------
# Just three paths per material — no orbital lists.  The .win projection
# + atoms_cart blocks fully determine the active mask.
TRAIN_INPUTS = [
    # (embedding .pt, hr-file, .win file, friendly name)
    ("outputs/Bi2Se3_embeddings.pt", "wannier_data/Bi2Se3_hr.dat",  "wannier_data/Bi2Se3.win",  "Bi2Se3"),
    # ("outputs/Bi2Te3_embeddings.pt", "wannier_data/Bi2Te3_hr.dat",  "wannier_data/Bi2Te3.win",  "Bi2Te3"),
    # ("outputs/Sb2Te3_embeddings.pt", "wannier_data/Sb2Te3_hr.hdf5", "wannier_data/Sb2Te3.win",  "Sb2Te3"),
    # ... add as many as you have ...
]

VAL_INPUTS = [
    # ("outputs/SnTe_embeddings.pt",   "wannier_data/SnTe_hr.dat",    "wannier_data/SnTe.win",    "SnTe"),
]

SAVE_DIR     = "./finetune_out"
ENERGY_RANGE = (-2.0, 2.0)             # eV — eigenvalues outside this window are masked
DEVICE       = "cpu"                    # use "cuda" if available


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1)  Build per-material training targets.  The .win parsing is
    #     automatic — no manual orbital lists.
    # ------------------------------------------------------------------
    print(f"Building {len(TRAIN_INPUTS)} training targets ...")
    train_items = []
    for embed_path, hr_path, win_path, name in TRAIN_INPUTS:
        item = prepare_finetune_target(
            embed_path       = embed_path,
            hr_path_or_model = hr_path,
            win_path         = win_path,
            name             = name,
            out_path         = os.path.join(SAVE_DIR, f"{name}_target.pt"),
        )
        train_items.append(item)
        print(f"  {name}: {item['gdata'].edge_targets.shape[0]} edges, "
              f"{int(item['gdata'].node_features[:, 109:127].sum())} active orbitals")

    val_items = []
    for embed_path, hr_path, win_path, name in VAL_INPUTS:
        val_items.append(prepare_finetune_target(
            embed_path       = embed_path,
            hr_path_or_model = hr_path,
            win_path         = win_path,
            name             = name,
            out_path         = os.path.join(SAVE_DIR, f"{name}_target.pt"),
        ))

    # ------------------------------------------------------------------
    # 2)  Run the multi-material fine-tune
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

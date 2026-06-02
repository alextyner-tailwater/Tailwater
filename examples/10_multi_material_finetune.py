"""Multi-material heads-only fine-tune against user Wannier targets.

This example walks through the full workflow when you want to refine
the API heads on a *set* of materials whose ground-truth Wannier
Hamiltonians you computed yourself (as opposed to single-material
self-distillation, which is `subspace_projection`).

Workflow
--------

For each material in the dataset:

  1. Call the API once to get the backbone embedding `.pt`
     (the file produced by /upload_structure_and_download_embeddings/).
  2. Have a Wannier Hamiltonian (`_hr.dat` or `_hr.hdf5`) and know
     which spatial orbitals were projected per atom — e.g. atoms
     1..2 ("Se") have just ``[s, pz, px, py]`` and atoms 3..8 ("Bi")
     have the full ``[s, pz, px, py, dz2, dxz, dyz, dx2-y2, dxy]``.
  3. Call `prepare_finetune_target` to build the per-material
     PyG `Data`-with-targets object (cached as a .pt if you'd like
     to skip the build step on subsequent runs).

Then call `finetune_heads_multi` on the list of prepared targets to
run the actual fine-tune.

The example below uses one Bi2Se3 material as both training and
validation just to keep the script self-contained — replace
`TRAIN_ITEMS` / `VAL_ITEMS` with your real list.

Requires:
    pip install tailwater
"""

import os
from tailwater import (
    prepare_finetune_target,
    finetune_heads_multi,
)


# ----------------------------------------------------------------------
# Per-material inputs.  In a real run you'd have N tuples here.
# ----------------------------------------------------------------------
# active_orbitals[a] = the list of SPATIAL orbital labels projected on
#                       atom `a` in the user's hr-model.  Each spatial
#                       slot covers both spin partners automatically.
#                       Valid labels: s, pz, px, py, dz2, dxz, dyz,
#                                     dx2-y2, dxy.
BI2SE3_ACTIVE = (
    [["s", "pz", "px", "py"]] * 2                                       # 2 Se atoms (s+p)
    + [["s", "pz", "px", "py", "dz2", "dxz", "dyz", "dx2-y2", "dxy"]] * 6  # 6 Bi atoms (s+p+d)
)

TRAIN_INPUTS = [
    # (embedding .pt path, hr-file path, per-atom active_orbitals, friendly name)
    ("outputs/Bi2Se3_embeddings.pt", "wannier_data/Bi2Se3_hr.hdf5", BI2SE3_ACTIVE, "Bi2Se3"),
    # ("outputs/Bi2Te3_embeddings.pt", "wannier_data/Bi2Te3_hr.hdf5", BI2TE3_ACTIVE, "Bi2Te3"),
    # ... add as many as you have ...
]

VAL_INPUTS = [
    # ("outputs/Sb2Te3_embeddings.pt", "wannier_data/Sb2Te3_hr.hdf5", SB2TE3_ACTIVE, "Sb2Te3"),
]

SAVE_DIR     = "./finetune_out"
ENERGY_RANGE = (-2.0, 2.0)             # eV — eigenvalues outside this window are masked
DEVICE       = "cpu"                    # use "cuda" if available


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1)  Build per-material training targets
    # ------------------------------------------------------------------
    print(f"Building {len(TRAIN_INPUTS)} training targets ...")
    train_items = []
    for embed_path, hr_path, active_orbitals, name in TRAIN_INPUTS:
        item = prepare_finetune_target(
            embed_path      = embed_path,
            hr_path_or_model= hr_path,
            active_orbitals = active_orbitals,
            name            = name,
            out_path        = os.path.join(SAVE_DIR, f"{name}_target.pt"),
        )
        train_items.append(item)
        print(f"  {name}: {item['gdata'].edge_targets.shape[0]} edges, "
              f"{int(item['gdata'].node_features[:, 109:127].sum())} active orbitals")

    val_items = []
    for embed_path, hr_path, active_orbitals, name in VAL_INPUTS:
        val_items.append(prepare_finetune_target(
            embed_path      = embed_path,
            hr_path_or_model= hr_path,
            active_orbitals = active_orbitals,
            name            = name,
            out_path        = os.path.join(SAVE_DIR, f"{name}_target.pt"),
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

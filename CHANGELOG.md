# Changelog

All notable changes to the `tailwater` package. This project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.0]

### Changed
- **`symmetrize` now defaults to `False` on `tw_api_call`** (was `True` in
  0.5.0–0.6.0). A plain call returns the raw prediction again; pass
  `symmetrize=True` to route to the Kramers-fixed endpoint. This is a
  client-side routing default only — the server endpoints are unchanged.

## [0.6.0]

### Added
- **`surface_charge_density`** — real-space surface charge-density heat maps of
  a general `(hkl)` slab, built directly from a Wannier Hamiltonian's `H(R)`.
  Accepts any model form: a `tbmodels.Model`, a tbmodels HDF5 path, a DFT
  Wannier90 `*_hr.dat`, or the dict from the new `load_hr`. Renders a top view
  (down the surface normal) + a side cross-section; `energy_window=` restricts
  to near-`E_F` states to image topological surface states. Also exports
  `load_hr` and `supercell_self_check` (verifies the general-`(hkl)` integer
  supercell remap reproduces the bulk spectrum to machine precision).
- **`examples/11_surface_charge_density.py`** + docs page
  (`docs/api/surface_charge.rst`).

## [0.5.1]

### Added
- **`symmetrize_targets` on `Subspace_EigLoss` / `subspace_projection`** —
  pair-averages the target eigenvalues so the fit enforces Kramers degeneracy,
  gated on detected `P` / `C2z` symmetry (off for non-PT crystals).

## [0.5.0]

### Changed
- **`symmetrize=True` is now the default on `tw_api_call`** — the standard call
  returns a Kramers-fixed `wannier90_hr.hdf5` when the crystal has `P` / `C2z`,
  else the raw model with an explanatory note. The `return_*` flags take
  precedence over `symmetrize`.

### Added
- **`model` parameter on `tw_api_call`** (`"V0.0"` / `"V0.1"`, default = server
  default) to select the inference checkpoint.

## [0.4.18]

### Added
- **`dev` flag on `tw_api_call`** — pass `dev=True` to opt into the server's
  canonical-cell position-wrap fix for band structures (sent as a `?dev=true`
  query param). Corrects bands for structures whose atoms sit on/over the
  unit-cell boundary (e.g. CIF fractional coords numerically ~1.0), which
  previously made the finite neighbor-shell sample the wrong periodic images.
  Default `False` reproduces prior behavior; older servers ignore the flag.

## [0.4.17]

### Added
- **Three-way band-structure comparison in
  `examples/10_multi_material_finetune.py`** — replaces the
  fine-tuned-only plot from 0.4.16. After training, the example now
  computes and overlays bands for three models on a single figure:

  1. **target** (black, solid) — the user's own ``_hr.dat`` /
     ``_hr.hdf5`` loaded via ``tbmodels.Model.from_wannier_files``,
     then Fermi-shifted to ``E_F = 0`` via ``align_to_vbm`` using the
     value parsed by ``parse_win_fermi_energy``. This is the same
     reference the multi-finetune loss saw during training.
  2. **pre-tune** (blue, dashed) — the packaged ``HeadsOnly_MACE.pth``
     run on the same validation embedding *without* any fine-tuning.
     The "before" picture.
  3. **post-tune** (red, solid) — the ``HeadsFT_multi_best.pth``
     produced by ``finetune_heads_multi`` run on the same embedding.
     The "after" picture.

  All three are computed on a generic Γ → M → K → Γ path inside one
  ``threadpoolctl.threadpool_limits(1)`` context, then plotted on a
  single matplotlib axis with distinct line styles. Watching the
  pre-tune blue dashes shift visibly toward the black target as
  training progresses is the cleanest single-figure way to validate
  that the multi-material fine-tune is actually doing something
  useful on the held-out material.

  Saved as ``{val_subdir}/{name}_bands_comparison.png`` alongside the
  predicted hr-model from the post-finetune run.


## [0.4.16]

### Added
- **End-to-end "use the fine-tuned heads" section in
  `examples/10_multi_material_finetune.py`** — after training, the
  script now:

  1. Reloads the best-val (or final) `HeadsFT_multi_*.pth` checkpoint
     via `load_heads_only_checkpoint`.
  2. Runs the heads on the first validation material's API embedding.
  3. Assembles the resulting `(edge_pred, onsite_pred)` into a
     `tbmodels.Model` via `build_hr_model_fast` and writes it as
     `{val_subdir}/{name}_finetuned_hr.hdf5`.
  4. Plots the bulk band structure on a generic Γ → M → K → Γ path
     and writes it as `{val_subdir}/{name}_bands_finetuned.png`.

  Also exposes the inference recipe for a brand-new structure (with
  a fresh `tw_api_call(..., return_embeddings=True)`) as a reference
  function at the bottom of the script.

### Fixed
- **macOS conda libomp clash** that segfaulted the example mid
  band-plot (PyTorch's bundled libomp and the libomp used by
  numpy / matplotlib racing in the same process). The example now
  sets `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"` *before* any
  torch import and wraps `bulk_band_structure` in
  `threadpoolctl.threadpool_limits(1)` to pin BLAS to a single
  thread for the duration. Together these two guards keep the
  in-process pipeline clean; the example exits 0 cleanly on a
  vanilla macOS conda environment without any shell env-var
  preamble. Linux users are unaffected — the guards no-op on
  systems without the conflict.


## [0.4.15]

### Fixed
- **`prepare_finetune_targets_from_directory` now prefers
  `wannier90.win` over `input.win`** when both are present in the
  same subdirectory.

  Background: the API's embedding endpoint writes its canonical
  Wannier90 input as `input.win` alongside the embedding `.pt`, while
  the user's own Wannier90 input — the one used to generate the
  fine-tune hr-file — is typically named `wannier90.win`. The 0.4.14
  walker globbed `*.win` and picked alphabetically, so it grabbed
  `input.win` (the API's full-projection .win) instead of the user's
  restricted-projection `wannier90.win`. Two failure modes followed:

  1. `tbmodels.Model.from_wannier_files` was called with the wrong
     .win, leading to all-zero Wannier-centre positions when the
     API's 8-atom .win didn't agree with the user's 6-atom hr.
  2. The hr-topology fallback added in 0.4.14 then saw all 44
     compact orbitals at one position and emitted
     `Atom block of size 44 cannot be mapped to a standard Wannier
     shell combination`, even though everything *could* have worked
     if the right .win had been picked up front.

  The fix flips the default `win_patterns` to
  ``("wannier90.win", "*.win")`` — exact `wannier90.win` first, any
  `.win` as fallback. Customers with a custom-named target .win can
  still override via `win_patterns=("my_target.win", "*.win")`.

  Verified on three layouts:

  - both `wannier90.win` + `input.win` → picks `wannier90.win` ✓
  - only `input.win`                    → picks `input.win` ✓
  - only `wannier90.win`                → picks `wannier90.win` ✓


## [0.4.14]

### Fixed
- **`prepare_finetune_target` now auto-recovers when the
  directory's `.win` projection block doesn't match the hr-file's
  actual orbital count** — the typical "fine-tune on materials with
  restricted projections" case. Previously, customers with hr-files
  Wannierized using just a subset of the canonical 18-orbital basis
  (e.g. `W: d` and `Se: p` → 10 + 6 = 16 compact orbs per WSe₂
  formula unit) hit a `ValueError: Orbital-map size (108) does not
  match hr_model.size (44)` because the API-style `.win` next to
  the hr still listed full `s, p, d` projections for the embedding
  step.

  The loader now falls back to the new
  `infer_active_orbitals_from_hr` helper whenever the .win-implied
  orbital count disagrees with `hr_model.size`. The fallback walks
  the hr-file's per-orbital `model.pos` groupings, counts compact
  orbitals per atom, and maps to the canonical Wannier shell
  combination (2 = s, 6 = p, 10 = d, 8 = s+p, 12 = s+d, 16 = p+d,
  18 = s+p+d). For physically reasonable projections the mapping is
  unambiguous, so the inferred per-atom shell list matches what
  `CleanedDatasets.ipynb` derives from a matching .win.

  Both paths produce the same `gdata.edge_targets` for materials
  where the .win and hr-file agree, so the change is backward
  compatible. A one-line log message reports which path was taken
  per material:

      [prepare_finetune_target] [WSe2-mock] .win projection implies 124
        compact orbitals but the hr-file has 44. Falling back to per-atom
        shell inference from the hr-file's position groupings — typical
        when the directory's .win is the API-side full-projection file
        but the hr was Wannierized with a restricted projection.

### Added
- **`tailwater.infer_active_orbitals_from_hr(hr_model, *, pos_tol=1e-6)`**
  — public helper that returns the canonical per-atom spatial-orbital
  list derived from the hr-file's own structure. Useful when the
  user wants the orbital layout but does not have (or trust) a
  matching .win projection block. Raises a clear `ValueError` if any
  per-atom compact count doesn't match a standard Wannier shell
  combination, so non-standard projections surface early instead of
  silently miscoding the targets.


## [0.4.13]

### Fixed
- **`prepare_finetune_target` could not load Wannier90 `_hr.dat`
  files** — the loader called `tbmodels.Model.from_hr_file(...)`,
  which is not part of the tbmodels public API. The directory walker
  surfaced this as a per-material `[skip]` line that read
  `AttributeError: type object 'Model' has no attribute 'from_hr_file'`
  on every subdirectory that had a `.dat` instead of an `.hdf5`,
  even when the rest of the inputs were correct.

  The correct API is `tbmodels.Model.from_wannier_files(hr_file=...)`
  — the loader now calls that. When the matching `.win` is also
  available in the same directory, it's passed through as
  `win_file=win_path, pos_kind='nearest_atom'` so orbital positions
  get assigned from the .win's `atoms_cart` block rather than
  requiring an extra `*_centres.xyz`.

  Verified end-to-end with the user's exact failing layout — a
  subdirectory named `s_-0.04_-0.25` containing `embeddings.pt`,
  `wannier90.win`, and `wannier90_hr.dat`:

      [ok]   s_-0.04_-0.25: 1088 edges, 124 active orbitals
             (embed=embeddings.pt, win=wannier90.win, hr=wannier90_hr.dat)
      Prepared 1 materials.


## [0.4.12]

### Added
- **`prepare_finetune_targets_from_directory(..., generate_embedding=True,
  user=..., password=...)`** — the directory walker can now call the
  API to generate any missing embedding file itself, so the user only
  needs to drop the (`.win`, `_hr.dat`) pair into each subdirectory:

      datasets/train/
      ├── Bi2Se3/
      │   ├── wannier90.win        # required
      │   └── wannier90_hr.dat     # required
      ├── Bi2Te3/
      │   └── ...
      └── ...

  Per subdirectory, the function reconstructs the Structure from the
  .win itself (via the new :func:`structure_from_win`) and calls
  :func:`tw_api_call` with ``return_embeddings=True`` to populate
  `{subdir}/embeddings.pt`. Subdirectories that already have an
  embedding are skipped — re-runs don't burn extra credits unless
  `force_regenerate=True` is passed.

  New related public helpers:

  - `parse_win_lattice(win_path) -> np.ndarray` — the 3×3 lattice
    matrix from the ``unit_cell_cart`` block (Bohr → Å conversion
    handled).
  - `structure_from_win(win_path) -> pymatgen.Structure` — combines
    `parse_win_atoms` + `parse_win_lattice` into a Structure ready to
    hand back to `tw_api_call`.

  Lazy import: the API-call code path only fires when
  `generate_embedding=True`, so users who only need the discovery
  loop don't need API credentials in scope.

### Fixed
- **`parse_win_fermi_energy` now warns + returns `None`** instead of
  raising when it finds a templated `fermi_energy = efermi`
  placeholder. The previous strict behavior made
  `prepare_finetune_targets_from_directory` skip whole materials
  whose .win still carried the placeholder string from
  `gen_struct` / Wannier90 template generation, even when they
  otherwise had everything needed for fine-tuning. The new behavior
  treats an unparseable value as "no shift requested" with a
  visible `RuntimeWarning`.


## [0.4.11]

### Changed
- **`prepare_finetune_target` and
  `prepare_finetune_targets_from_directory` now auto-read the
  Fermi energy from each material's .win file and shift its on-site
  energies so `E_F = 0`** — matching the convention used to build
  the training data in `CleanedDataset.ipynb`. The Tailwater model
  is trained on `E_F = 0`-referenced Hamiltonians, so user fine-tune
  targets need to follow the same convention; without the shift the
  loss is dominated by the constant Fermi offset rather than the
  band-structure detail the user actually wants to refine.

  Per-material, the `fermi_energy` keyword in the .win file
  (whatever its sign) is subtracted from every on-site diagonal
  entry before the target tensor is built. The cost is zero — it's a
  single subtraction on the already-walked entries of
  `hop[(0,0,0)]`. Both functions print one line per material noting
  the value that got picked up:

      [prepare_finetune_target] [Bi2Se3] using fermi_energy = -0.432100 eV
        from wannier90.win (subtracted from on-sites to set E_F = 0)

  Override semantics:

  - `fermi_shift=None` (default) → read `fermi_energy` from each .win.
  - `fermi_shift=0.0`            → disable shifting entirely.
  - `fermi_shift=<float>`        → apply that shift to every material,
                                    ignoring the .win values.

### Added
- **`tailwater.parse_win_fermi_energy(win_path)`** — public parser
  that returns the .win's `fermi_energy` (eV) or `None` if absent.
  Handles all three Wannier90 separator conventions
  (`fermi_energy = X`, `fermi_energy : X`, `fermi_energy  X`),
  ignores comments, skips lines inside explicit `begin … end` blocks.


## [0.4.10]

### Added
- **`tailwater.prepare_finetune_targets_from_directory(root_dir,
  ...)`** — bulk auto-discovery of (embedding, hr, .win) triples
  from a tree of per-material subdirectories.

  Layout convention:

      datasets/train/
      ├── Bi2Se3/
      │   ├── embeddings.pt
      │   ├── wannier90.win
      │   └── wannier90_hr.dat
      ├── Bi2Te3/
      │   └── ...
      └── ...

  One subdirectory per material; the **subdirectory name becomes the
  material name** in training logs and the cached `.pt` filename. The
  function walks the tree, finds each triple via glob patterns
  (defaults: `*embeddings*.pt`, `*.win`, `*_hr.dat` / `*_hr.hdf5`),
  and calls `prepare_finetune_target` once per discovered
  subdirectory. Returns the list of prepared items ready to hand to
  `finetune_heads_multi`.

  Quick recipe:

  ```python
  from tailwater import (
      prepare_finetune_targets_from_directory,
      finetune_heads_multi,
  )
  train_items = prepare_finetune_targets_from_directory("datasets/train",
                                                         out_dir="cache")
  val_items   = prepare_finetune_targets_from_directory("datasets/val",
                                                         out_dir="cache")
  finetune_heads_multi(train_targets=train_items, val_targets=val_items,
                       start_lr=5e-5, end_lr=5e-7, num_epochs=50,
                       energy_range=(-2.0, 2.0), decay_sigma=1.0,
                       device="cuda", save_path="finetune_out")
  ```

  Subdirectories missing any of the three required files are skipped
  with a warning by default; pass `strict=True` to make missing files
  raise instead. Glob patterns are user-customisable for non-standard
  filename conventions.

  Verified end-to-end on a synthetic 3-subdirectory tree (2 complete +
  1 missing .win): the discovery skips the bad one with a warning,
  the full multi-finetune loop runs to completion, and `strict=True`
  raises with a clear "missing .win" message on the same input.


## [0.4.9]

### Changed
- **`prepare_finetune_target` now accepts a `.win` file directly** —
  the per-atom active-orbital layout is parsed straight out of the
  Wannier90 projection + atoms_cart blocks, using the same
  convention as the API's server-side `process_win`. Customers no
  longer need to type per-atom orbital lists by hand:

  ```python
  item = prepare_finetune_target(
      embed_path        = "outputs/Bi2Se3_embeddings.pt",
      hr_path_or_model  = "wannier_data/Bi2Se3_hr.dat",
      win_path          = "wannier_data/Bi2Se3.win",
      name              = "Bi2Se3",
  )
  ```

  `active_orbitals` remains available as an explicit override for the
  rare case where the fine-tune should run on a subspace smaller than
  the full Wannier projection. The signature change is backward
  compatible — `active_orbitals` is still keyword-accessible — but
  is no longer required.

### Added
- **`tailwater.active_orbitals_from_win(win_path)`** — public helper
  that returns the per-atom spatial-orbital list a .win file implies.
  Output matches the API server-side convention so customers can use
  the same layout the API saw at embedding time.
- **`tailwater.parse_win_projections(win_path)`** and
  **`tailwater.parse_win_atoms(win_path)`** — lower-level parsers for
  the projection and atoms_cart blocks. Useful for sanity-checking
  that the .win the user is fine-tuning against matches the structure
  they uploaded to the API.

Verified: the parsed active mask matches the active mask stored in
the API embedding to the bit for the test material (KInTe), and the
4-epoch CPU smoke run produces an identical training trajectory to
the explicit-list variant.


## [0.4.8]

### Added
- **Multi-material heads fine-tune against user-supplied Wannier
  targets** — complementing single-material `subspace_projection`,
  which uses the API's own full-model output as a self-distillation
  target. The new path lets users refine the heads on a set of N
  materials whose ground-truth Hamiltonians they computed
  themselves.

  Three new public symbols (`from tailwater import ...`):

  - `prepare_finetune_target(embed_path, hr_path_or_model,
    active_orbitals, *, fermi_shift=0.0, out_path=None, name=None)`
    Merges an API embedding `.pt` with a user-supplied `tbmodels.Model`
    (or hr-file path), producing a per-material training item whose
    `gdata.edge_targets` is filled from the user's Hamiltonian and
    whose `gdata.node_features[:, 109:127]` carries the user's
    active-orbital mask. Critically, the user's hr need NOT have the
    full 18 spatial orbitals per atom — each atom can carry an
    arbitrary subset of `{s, pz, px, py, dz2, dxz, dyz, dx2-y2, dxy}`.
    The (0,0,0)-block doubling convention is handled transparently
    so the on-site values in `gdata.edge_targets` are physical.

  - `finetune_heads_multi(train_targets, *, start_lr, end_lr,
    num_epochs, energy_range, decay_sigma, device, save_path,
    val_targets=None, val_every=5, loss_mode="subspace",
    heads_checkpoint=None, h_mse_weight=0.001, eig_weight=1.0,
    kgrid_n=4, grad_clip=1.0)`
    Cosine-annealed AdamW over the N training materials per epoch
    (full-batch gradient accumulation across materials). Subspace
    eigenvalue loss masked outside `energy_range` per material;
    optional validation set with mean validation eigenvalue loss
    reported every `val_every` epochs; best-val checkpoint kept
    alongside the final one. Saves to
    `HeadsFT_multi_final.pth` / `HeadsFT_multi_best.pth`.

  - `build_active_mask`, `build_edge_targets_from_hr`,
    `SPATIAL_LABEL_TO_INDEX` — lower-level building blocks for users
    who want to bypass `prepare_finetune_target` and stitch their
    own per-material items together.

  See `examples/10_multi_material_finetune.py` for the canonical
  recipe.


## [0.4.7]

### Added
- **`tailwater.wb_system_with_spin(model, ...)`** — builds a
  `wannierberri.system.System_R` with spin matrix elements (`SS_R`)
  populated, enabling spin-resolved calculators (most importantly
  `wannierberri.calculators.static.SHC`) that WannierBerri's own
  `System_R.from_tbmodels(spin=True)` doesn't support out of the box.
  The function exploits the fact that every Wannier function in the
  Tailwater 18-orbital basis is an exact σ_z eigenstate with the
  convention `orbital_index = spatial_index * 2 + spin_index` to
  construct `SS_R[(0,0,0)]` analytically from Pauli matrices.

  Three modes for resolving the σ_z eigenstate doublets, in order of
  priority:

  1. Caller passes `pairs=[(up_idx, down_idx), ...]` explicitly.
  2. Caller passes `basis_json_path=...` (the JSON written by
     `subspace_projection`).
  3. **(default)** Inferred from the model's atomic-position topology
     via the new `spin_pairs_from_model_topology(model)` helper:
     orbitals at the same lattice position are grouped per atom, then
     paired as consecutive Kramers partners, with an on-site-energy
     match check.

  Example: end-to-end intrinsic spin Hall conductivity on Bi₂Se₃
  lands in ~13 s on a laptop and shows the expected in-gap plateau
  of σ^z_xy ≈ -285 (ℏ/e)·S/cm (canonical SHC topological signature):

  ```python
  import numpy as np, wannierberri as wb
  from tailwater import tb_model, wb_system_with_spin

  model = tb_model.load("wannier90_hr.hdf5")
  sys   = wb_system_with_spin(model)

  Efermi = np.linspace(-2.0, 2.0, 41)
  grid   = wb.Grid(sys, NK=(8, 8, 8), NKFFT=(4, 4, 4))
  result = wb.run(sys, grid=grid, calculators={
      "shc": wb.calculators.static.SHC(
          Efermi=Efermi,
          kwargs_formula={"spin_current_type": "simple"},
      ),
  }, parallel=False, symmetrize=False, dump_results=False)
  ```

  See `examples/07_spin_hall_conductivity.py` for a full sweep with
  the DOS panel for context.


## [0.4.6]

### Fixed
- **`subspace_projection(device='cuda', ...)` crashed at the hr-write
  step** with a cross-device error like
  ``RuntimeError: expected device cuda:0 but got device cpu`` (the
  exact message varies by torch version). The `build_hr_model` /
  `build_hr_model_fast` builders allocate the dense intermediate
  `HoppT` tensor on CPU, then fancy-index-assign from `edge_pred` /
  `onsite_pred`. When the surrounding workflow runs on CUDA those
  inputs are GPU-resident and the assignment errors. The builders now
  detach + move every torch-tensor input to CPU at the boundary, so
  callers can pass GPU tensors without the build step caring.

  Verified end-to-end on the equivalent macOS `mps` device (the only
  non-CPU device available on the build host): the resulting
  `tbmodels.Model` produces a Hamiltonian bit-identical to the CPU
  path (max diff = 0 at every k).


## [0.4.5]

### Added
- **`model.to_kwant()`** — instance-bound converter from the loaded
  `tbmodels.Model` into a `kwant.Builder` with a 3D
  `kwant.TranslationalSymmetry`. Parallel in spirit to
  `model.to_pb()` and `model.to_pythtb()`. After wrapping the
  returned Builder via `kwant.wraparound.wraparound(...).finalized()`
  and sampling H(k), the eigenvalues match `model.hamilton(k_frac)`
  to **float64 precision** (~1e-13 eV) at every k.

  Quick usage:

  ```python
  import numpy as np, kwant
  from tailwater import tb_model

  model = tb_model.load("wannier90_hr.hdf5")
  syst  = kwant.wraparound.wraparound(model.to_kwant()).finalized()

  # Kwant's wraparound k-parameters are 2π·k_frac (per-cell Bloch
  # phase), NOT Cartesian rad/length like pybinding's.
  phase = 2 * np.pi * np.array([0.5, 0.0, 0.0])
  H = syst.hamiltonian_submatrix(
      params=dict(k_x=phase[0], k_y=phase[1], k_z=phase[2]),
  )
  ```

  The returned Builder is *unfinalised* so it can also serve as the
  base for transport calculations (attach leads + call
  `kwant.smatrix` / `kwant.greens_function`).

  Requires `conda install -c conda-forge kwant` (kept out of
  tailwater's deps; a clear ImportError points at the install command
  if the user calls `.to_kwant()` without it).


## [0.4.4]

### Added
- **`model.to_pythtb()`** — instance-bound converter from the loaded
  `tbmodels.Model` into a `pythtb.tb_model`, parallel in spirit to
  `model.to_pb()`. After conversion,
  `py_model.solve_one(k_frac)` and
  `np.linalg.eigvalsh(model.hamilton(k_frac))` match to **float64
  precision** (~5e-14 eV) at every k. Unlike the pybinding path,
  PythTB uses fractional k and fractional orbital positions directly,
  so no companion `k_cart_from_frac` is needed — the same recipe
  that worked with `tbmodels.Model.hamilton(k)` works with the
  PythTB model.

  Quick usage:

  ```python
  from tailwater import tb_model
  model    = tb_model.load("wannier90_hr.hdf5")
  py_model = model.to_pythtb()
  eig      = py_model.solve_one([0.0, 0.0, 0.0])     # Γ
  py_slab  = py_model.cut_piece(num=6, fin_dir=2)    # 6-layer slab
  ```

  Requires `pip install pythtb` (kept out of tailwater's dependency
  set; a clear ImportError points at the install command if the user
  calls `.to_pythtb()` without it).


## [0.4.3]

### Fixed
- **`model.to_pb()` now produces a pybinding lattice whose H(k) eigenvalues
  match `model.hamilton(k)` to float32 precision** (~1e-6 eV) at every k.
  Two bugs were behind the previously-observed band-structure discrepancy:

  1. **On-site doubling.** tbmodels' `hamilton()` constructs H(k) via
     ``Σ_R stored[R] e^{ikR}`` followed by ``H += H.c.``. The H.c. step
     supplies the missing -R half for R ≠ 0, but at R=(0,0,0) it
     doubles the stored block on top of itself — so tbmodels' internal
     `hop[(0,0,0)]` is exactly **half** the physical on-site block.
     `to_pb` previously fed the half-stored block to pybinding,
     producing a Hamiltonian missing half of every on-site
     contribution. Now multiplies `hop[(0,0,0)]` by 2 before feeding
     pybinding.

  2. **Position basis.** Pybinding's `add_one_sublattice` expects
     positions in Cartesian (length units); `tbmodels.Model.pos` stores
     them in fractional (lattice) coordinates. `to_pb` previously
     passed the fractional values through unchanged. Eigenvalues were
     invariant under this (just a per-orbital unitary phase) but
     pybinding's real-space geometry routines and Brillouin-zone
     calculations were wrong. Now converts with `pos_cart = pos_frac @ LM`.

### Added
- **`tailwater.k_cart_from_frac(k_frac, lattice_vectors)`** — converts
  a fractional k-point (or batch) to pybinding's Cartesian (rad/length)
  convention via ``k_cart = 2π · inv(LM) @ k_frac``. Pair with
  `pb.Model.set_wave_vector(...)` to sample bands on the pybinding side
  at the same k as `model.hamilton(k_frac)` on the tbmodels side.

If you used `model.to_pb()` in 0.4.0–0.4.2 and observed band-structure
discrepancies vs. `model.hamilton(k)`, **upgrade to 0.4.3** — no API
changes, just correct numbers.


## [0.4.2]

### Performance
- **`SurfaceGreensFunction` and `FermiArcMap` are now 3–6× faster** in their
  default configurations and 8–12× faster with multi-process parallelism on
  a typical 16-core CPU. Two new knobs control the speed/memory trade-off:

  * **Batched Lopez-Sancho recursion** (always on). For each k-point the
    recursion is now run as a single batched LAPACK pass over all energies
    (or chunks of size ``chunk_size``, default ``256``) rather than one
    serial solve per energy. The two `solve` calls per Lopez-Sancho step
    that share the same coefficient matrix are also combined into a single
    multi-RHS solve, halving the LU factorization work.
  * **k-point parallelism** via the new ``n_jobs`` kwarg (default ``1`` —
    no behavior change). Pass ``n_jobs=-1`` (all cores) or any integer to
    fan k-points out across worker processes through ``joblib``. Each
    worker pins itself to one BLAS thread to avoid oversubscription, so
    scaling is close to linear in the number of physical cores until
    pickling overhead bites (typically at ~16 workers for this problem).

    Recommended recipe for any nontrivial run::

        SurfaceGreensFunction(model, ..., n_jobs=-1)
        FermiArcMap(model, ..., n_jobs=-1)

  All results are bit-exact relative to the previous serial implementation
  (verified against a Bi₂Se₃ slab: max |Δspectral density| = 0.0).

### Added
- ``joblib >= 1.0`` as a runtime dependency (used only when ``n_jobs != 1``;
  ``joblib`` is also a transitive dependency of ``scipy`` and ``sklearn``,
  so it's almost always already installed).


## [0.4.1]

### Fixed
- **`SurfaceGreensFunction.run()` and `FermiArcMap.run()`** — both crashed with
  `AttributeError: 'Model' object has no attribute '_gen_ham'` on every
  `tbmodels` ≥ 1.4 install. `_gen_ham` was a private helper in an older
  internal fork; the public, supported API is `Model.hamilton(k)`. Both
  call sites now use `hamilton(k)`, so surface-state and Fermi-arc
  calculations work again out of the box.

If you installed `0.4.0`, upgrade with `pip install -U tailwater` — every
other 0.4.0 feature (VBM alignment, bundled HeadsOnly checkpoint, default
production API URL) is preserved.


## [0.4.0]

### Added
- **`compute_band_edges(model, k_mesh=(4,4,4))`** — locates VBM / CBM / gap
  on a uniform Monkhorst-Pack k-mesh by taking `VBM = max(eigs < 0)` and
  `CBM = min(eigs > 0)` from the model's spectrum (assuming the training
  convention `E_F = 0`). Returns `{"vbm","cbm","gap","is_metal"}`.
- **`align_to_vbm(model, k_mesh=(4,4,4), fermi_level=None, if_metal="warn")`**
  — returns a deep copy of the model with on-site energies shifted by
  `-VBM`, so the valence band maximum sits at zero. This is the natural
  reference for band-edge plots / DOS / surface-state calculations on
  semiconductors and insulators. Pass `fermi_level=<float>` to override
  the auto-detected VBM with a known value. For metals (no clean gap
  around E=0), the default `if_metal="warn"` emits a `RuntimeWarning`
  and returns the unshifted model so downstream code still runs;
  `"raise"` and `"skip"` are also accepted.

Recommended workflow on a non-metal:
```python
from tailwater import tb_model, align_to_vbm, BulkDOS

model = tb_model.load("wannier90_hr.hdf5")
model = align_to_vbm(model)             # VBM is now at E=0 across all calculators
dos   = BulkDOS(model, energies=(-3, 3), k_mesh=(8,8,8)).run()
```


## [0.3.2]

### Changed
- **Dropped the `[pybinding]` extra.** Customers no longer need the awkward
  bracket-syntax install. The recommended pattern is:

      pip install tailwater                # base install
      pip install pybinding-dev            # only if you call subspace_projection or tb_model.to_pb()

  `pybinding-dev` is a regular PyPI package, installed the same way it would
  be on its own. The `ImportError` raised by `build_hr_model` /
  `build_hr_model_fast` now points at `pip install pybinding-dev` directly.


## [0.3.1]

### Fixed
- **Bundled `HeadsOnly_MACE.pth` now matches the production backbone.**
  The 0.3.0 ship was built from an older `WanE3MACE.irreps_mid` (dim 820) and
  raised `RuntimeError: Embedding dim 851 != heads' irreps_in.dim 820` when
  fed real embeddings from the production API. Regenerated the heads from
  the production `evMace_Epoch_51.pth` checkpoint against the current
  `WanE3MACE` (`irreps_mid = 64x0e+64x0o+32x1o+16x1e+12x2o+25x2e+18x3o+9x3e+
  4x4o+9x4e+4x5o+4x5e`, dim 851) so the heads' `irreps_in` lines up with
  the embeddings the API returns. No code changes — just the bundled
  checkpoint.


## [0.3.0]

### Added
- **Bundled MACE-compatible HeadsOnly checkpoint.** `HeadsOnly_MACE.pth`
  (~1.9 MB, built from the production `WanE3MACE` backbone via
  `API/make_heads_only.py`) now ships *inside* the installed package via
  `[tool.setuptools.package-data]`. `subspace_projection(...)` defaults to
  loading it automatically, so customers don't have to source a HeadsOnly
  checkpoint themselves.

### Fixed
- **`subspace_projection` was defaulting to a Lite-era heads checkpoint.**
  Previously `heads_checkpoint: str = "HeadsOnly.pth"` looked for a file in
  the caller's CWD that (a) usually didn't exist, or (b) if it did, was the
  retired `WanE3Lite`-backbone checkpoint that's incompatible with the
  embeddings the API now returns from `WanE3MACE` — leading to silently
  wrong fine-tuned heads. The default is now `None` and resolves at
  call time to the bundled `HeadsOnly_MACE.pth`. Pass an explicit
  `heads_checkpoint=` only if you're starting from a custom checkpoint.


## [0.2.2]

### Fixed
- **`pip install tailwater` (no extras) is now importable.** Previous releases
  put `import pybinding as pb` at the top of `tailwater.hr_export`, so just
  `import tailwater` raised `ModuleNotFoundError: No module named 'pybinding'`
  for any user who hadn't installed the `[pybinding]` extra. The import is
  now lazy: `import tailwater` succeeds with no extras, and a clear
  `ImportError` (with the install hint) fires only when `build_hr_model` or
  `build_hr_model_fast` is actually called without `pybinding` installed.

  If you call those builders, install pybinding directly:

      pip install pybinding-dev

  Most users don't need it — they upload structures and consume the returned
  HDF5 through `tb_model.load(...)`.


## [0.2.1]

### Docs
- Removed every customer-facing reference to non-default API endpoints
  from `README.md`, `docs/installation.rst`, `docs/quickstart.rst`, and the
  `tw_api_call` docstring / `DEFAULT_API_URL` comment in `tailwater.client`.
  The default endpoint (`https://api.tailwater.io`) is what every normal
  user should hit, and the docs no longer suggest otherwise.

### Unchanged (still supported, just not advertised)
- The `api_url=` keyword argument on `tw_api_call(...)` and
  `remaining_credits(...)`.
- The `TW_API_URL` environment variable.

Both remain functional for the rare case the Tailwater team points a user
at a non-default endpoint; they're simply not surfaced in the user docs.


## [0.2.0]

### Changed
- **Default API endpoint is now the hosted Tailwater inference API**
  (`https://api.tailwater.io`). The client talks to it automatically — no
  configuration needed beyond your credentials. The `api_url=` argument and
  `TW_API_URL` environment variable remain available for the rare case the
  Tailwater team points you at a non-default endpoint.

### Removed
- Stale `TW_API` legacy-callable reference dropped from `README.md` and
  `docs/api/client.rst` (the symbol itself was removed in an earlier change,
  but the doc references lingered and produced Sphinx warnings).

### Docs
- Added an "API access" section to `README.md` and a "Getting API access"
  section to `docs/installation.rst` covering the default endpoint,
  credentials flow (HTTP Basic, request from the Tailwater team), credit
  metering, and how to check the balance.
- `docs/quickstart.rst` opens with a brief "you need credentials" note.

## [0.1.0]

Initial release.

# Changelog

All notable changes to the `tailwater` package. This project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

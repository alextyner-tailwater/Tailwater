# Changelog

All notable changes to the `tailwater` package. This project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

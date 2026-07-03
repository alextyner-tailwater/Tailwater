# tailwater

Client + post-processing toolkit for the **Tailwater** Wannier-Hamiltonian inference API.

`tailwater` lets you upload a crystal structure to the Tailwater API, receive a tight-binding Hamiltonian, optionally fine-tune the output heads on customer-side targets, and run band-structure / DOS / surface-state analyses locally — all from one pip-installable package.

---

## Installation

```bash
pip install tailwater
```

Optional extras:

```bash
pip install "tailwater[scatter]"        # if torch_scatter import fails
pip install "tailwater[seekpath]"       # enables auto k-path mode of bulk_band_structure
pip install "tailwater[dev]"            # pytest, ruff, build, twine
```

The tight-binding-library converters (`to_pb` / `to_pythtb` / `to_kwant`, and
`tb_model.load(...).to_pb()`) need their target package, imported lazily so the
base install stays light:

```bash
pip install pybinding-dev                 # pybinding  (a separate package, not a tailwater extra)
pip install pythtb                        # PythTB
conda install -c conda-forge kwant        # Kwant
```

`as_tbmodels` / `to_hr_dat` / `to_hdf5` need only `tbmodels`, which is a core
dependency (installed automatically).

Tested on Python 3.9–3.12.

---

## API access

The Tailwater inference API is hosted at **`https://api.tailwater.io`** — this
is the default endpoint `tw_api_call(...)` talks to, so the basic usage below
needs no extra configuration beyond your credentials.

- **Credentials.** Authentication is **HTTP Basic** (username + password).
  Email the Tailwater team to request an account; you'll be issued a username
  and a one-time-displayed password.
- **Billing.** Each successful inference call decrements your server-side
  credit balance by one. Health checks (`/healthz`) and balance lookups
  (`/credits/`) are free.
- **Checking your balance:**

  ```python
  from tailwater import remaining_credits
  print(remaining_credits("user", "pw"))   # -> int
  ```

---

## Three workflow layers

### 1. HTTP client — talk to the API

```python
from pymatgen.core import Structure
from tailwater import tw_api_call

structure = Structure.from_file("MyMaterial.cif")

# tw_api_call ALWAYS returns a dict of extracted paths. Every response
# includes a "win" key — the canonical wannier90.win file the server
# actually ran inference on — alongside the mode-specific artifact(s).

# (a) default: tbmodels HDF5 hr-model + .win
paths = tw_api_call(structure, "user", "pw", "./outputs", "my_mat")
# paths = {"hdf5": "...", "win": "..."}

# (b) backbone embeddings + .win
paths = tw_api_call(structure, "user", "pw", "./outputs", "my_mat",
                    return_embeddings=True)
# paths = {"embeddings": "...", "win": "..."}

# (c) project bundle: all artifacts + .win in a single call
paths = tw_api_call(structure, "user", "pw", "./outputs", "my_mat",
                    project=True)
# paths = {"hdf5": "...", "embeddings": "...", "npz": "...", "win": "..."}

# (d) sparse output — always keep the O(N) SparseHR .npz (large systems)
paths = tw_api_call(structure, "user", "pw", "./outputs", "my_mat",
                    output_format="sparse")
# paths = {"npz": "...", "meta": "...", "win": "..."}
```

Five output modes are available — `return_embeddings`, `return_input`, `return_graph_output`, `project`, or default HDF5. Set them as keyword arguments to `tw_api_call`. See `tailwater.tw_api_call.__doc__` for the priority order.

**`output_format`** (`"auto"` | `"sparse"` | `"hdf5"`, default `"auto"`) controls
how the Hamiltonian is transported and delivered:

- `"auto"` — request the sparse `wannier90_hr.npz` (O(N) egress). Systems **below
  30 atoms** are transparently converted back to dense HDF5 (`r["hdf5"]` still
  works, and `r["npz"]` is kept too); larger systems stay sparse under `r["npz"]`,
  with a printed note on how to convert / analyse them.
- `"sparse"` — always keep the raw `.npz` (a [`SparseHR`](#sparse-hamiltonians-sparsehr--format-conversion)), whatever the size.
- `"hdf5"` — always deliver dense tbmodels HDF5 (the pre-0.9 behaviour).

A server that predates the sparse backend ignores the flag and returns HDF5, so
every mode degrades cleanly. See **Sparse Hamiltonians** below for what to do with
the `.npz`.

Each successful call decrements your server-side credit balance by one. Failures surface as `PermissionError` (401, bad password) or `RuntimeError` (402, out of credits / other 5xx).

### 2. Subspace projection — fine-tune heads on supplier-side embeddings

Once you have the project bundle, you can fine-tune the output heads to fit a narrow energy window near the Fermi level:

```python
from tailwater import subspace_projection

subspace_projection(
    start_lr          = 5e-5,
    end_lr            = 5e-7,
    num_epochs        = 20,
    energy_range      = (-2.0, 2.0),       # eV, relative to E_F
    decay_sigma       = 1.0,
    device            = "cpu",
    save_path         = "./projection_out",
    embed_path        = paths["embeddings"],
    graph_output_path = paths["graph_output"],
    loss_mode         = "subspace",         # default
)
```

Per epoch the script prints the mean eigenvalue loss. When done, three files are written to `save_path`:

| File | Contents |
|---|---|
| `HeadsFT_final.pth` | fine-tuned heads weights + metadata |
| `{stem}_pred.hdf5` | projected, subspace-restricted `tbmodels.Model` |
| `{stem}.basis.json` | mapping from subspace indices to `(atom, spatial, spin)` labels |

Three loss modes are exposed:
- `"subspace"` (default) — H-MSE + weighted eigenvalue loss within the energy window
- `"eig_only"` — eigenvalue-only fine-tune; no Hamiltonian targets needed
- `"full"` — plain H-MSE across all orbitals

**Fine-tuning from the sparse `.npz`.** If you took the sparse output
(`output_format="sparse"`, or a large-system `project=True` bundle), pass the
`.npz` as **`hr_npz_path`** instead of `graph_output_path`. The projection then
fits the heads to the **in-window eigenvalues of the sparse Hamiltonian**
(eigenvalue-only downfolding), so the whole workflow needs only `embeddings.pt`
+ the `.npz` — no dense `graph_output.pt` / HDF5:

```python
subspace_projection(
    start_lr=1e-4, end_lr=1e-5, num_epochs=20,
    energy_range=(-2.0, 2.0), decay_sigma=0.5, device="cpu",
    save_path="./projection_out",
    embed_path  = paths["embeddings"],
    hr_npz_path = paths["npz"],          # sparse target (instead of graph_output_path)
)
```

Provide **exactly one** of `graph_output_path` (dense self-distillation target)
or `hr_npz_path` (sparse target); with `hr_npz_path` the loss mode is forced to
`"eig_only"`.

### 3. Post-processing — bulk DOS, surface states, Fermi arcs

```python
import numpy as np
from tailwater import (
    tb_model,
    BulkDOS,
    SurfaceSpectralDensity,
    SurfaceGreensFunction,
    FermiArcMap,
)

# Load the HDF5 the API produced — returns a tbmodels.Model with .to_pb()
model = tb_model.load("outputs/wannier90_hr.hdf5")

# Bulk DOS (KPM, k-mesh averaged)
result = BulkDOS(model, k_mesh=(8, 8, 8), energies=(-4, 4),
                 NC=2048, NV=4).run()
result.figure.savefig("bulk_dos.png")
np.savez("bulk_dos.npz", **result.as_dict())

# Surface spectral density along a k-path (KPM)
result = SurfaceSpectralDensity(
    model, surface=np.eye(3), LZ=5,
    energies=(-1, 1),
    k_path=[[0, 0.5, 0], [0, 0, 0], [0.333, 0.333, 0]],
    k_labels=["M", r"$\Gamma$", "K"],
    N_path=101, NC=2**12, NV=4,
).run()
result.figure_top.savefig("surface_top.png")
result.figure_bottom.savefig("surface_bottom.png")

# Surface Green's function (Lopez-Sancho)
result = SurfaceGreensFunction(
    model, surface=np.eye(3),
    energies=np.linspace(-1, 1, 201),
    k_path=[[0, 0.5, 0], [0, 0, 0], [0.333, 0.333, 0]],
    k_labels=["M", r"$\Gamma$", "K"],
    N_path=101, thickness=6, NN=5, eps=0.005,
).run()
np.savez("surface_gf.npz", **result.as_dict())

# 2D Fermi-arc map at one energy
result = FermiArcMap(
    model, surface=np.eye(3), energy=0.0,
    Nx=50, Ny=50, thickness=6,
).run()
result.figure_top_interpolated.savefig("fermi_arc_top.png")

# Bulk band structure along a manual k-path
from tailwater import bulk_band_structure
fig = bulk_band_structure(
    model,
    k_points = [[0, 0.5, 0], [0, 0, 0], [0.333, 0.333, 0], [0, 0, 0]],
    k_labels = ["M", r"$\Gamma$", "K", r"$\Gamma$"],
    spacing  = 0.01,
    fermi_level = 0.0,
    e_range  = (-3, 3),
)
fig.savefig("bands.png")

# Or use seekpath to auto-determine the high-symmetry path
from pymatgen.core import Structure
structure = Structure.from_file("MyMaterial.cif")
fig = bulk_band_structure(model, auto=True, structure=structure,
                          spacing=0.02, e_range=(-3, 3))
fig.savefig("bands_auto.png")
```

Each post-processing class accepts either an HDF5 path (`str`) or an in-memory `tbmodels.Model`. The `.run()` method returns a typed `Result` dataclass with raw NumPy arrays and matplotlib `Figure` objects.

---

## Sparse Hamiltonians (`SparseHR`) & format conversion

For large systems the API returns the Hamiltonian in **sparse** form — a
`wannier90_hr.npz` (COO hops + on-site diagonal + geometry) that is O(N) in
RAM/egress instead of O(N²). You get it from `output_format="sparse"` (always),
or from `"auto"` for systems ≥ 30 atoms (see the client section above).

Load a `.npz` with **`SparseHR`**, then compute spectra directly — including for
`num_wann` far larger than a dense H(k) could hold — or convert to any supported
tight-binding format:

```python
from tailwater import SparseHR

shr = SparseHR.load("outputs/wannier90_hr.npz")
shr.num_wann, shr.nnz                       # size / number of stored hops
Hk  = shr.Hk([0.0, 0.0, 0.0])               # scipy sparse H(k) at fractional k (Γ)
w   = shr.eigsh_near_fermi([0, 0, 0], e_fermi=0.0, num=20)  # 20 states nearest E_F (shift-invert)
Rd  = shr.hr_dict()                         # {R_tuple: scipy.sparse.csr_matrix}
ev  = shr.eigvals_grid([[0,0,0], [0.5,0,0]])# dense eigenvalues on a list of k-points
```

### Convert to any format — one call, auto-detecting the input

The top-level converters accept **either** a sparse input (a `SparseHR` or a
`.npz` path) **or** a dense one (a `tbmodels.Model`, an `.hdf5`, or a Wannier90
`_hr.dat`) and dispatch automatically — the same call works regardless of what
you're holding:

```python
from tailwater import as_tbmodels, to_hr_dat, to_hdf5, to_pb, to_pythtb, to_kwant

npz = "outputs/wannier90_hr.npz"
model     = as_tbmodels(npz)                 # tbmodels.Model (dense)
to_hr_dat(npz, "wannier90_hr.dat")           # Wannier90 _hr.dat
to_hdf5(npz,   "wannier90_hr.hdf5")          # tbmodels HDF5
pb_lattice = to_pb(npz)                       # pybinding.Lattice
py_model   = to_pythtb(npz)                    # pythtb model
syst, lat  = to_kwant(npz)                     # kwant (Builder, lattice)

# the identical calls work on a dense input, too:
to_hr_dat("wannier90_hr.hdf5", "wannier90_hr.dat")
pb_lattice = to_pb(model)
```

| Target | Top-level converter | `SparseHR` method | Requires |
|---|---|---|---|
| `tbmodels.Model` | `as_tbmodels(src)` | `.to_tbmodels()` | tbmodels (core dep) |
| Wannier90 `_hr.dat` | `to_hr_dat(src, path)` | `.to_hr_dat(path)` | tbmodels |
| tbmodels HDF5 | `to_hdf5(src, path)` | `.to_hdf5(path)` | tbmodels |
| pybinding | `to_pb(src)` | `.to_pb()` | `pybinding-dev` |
| PythTB | `to_pythtb(src)` | — | `pythtb` |
| Kwant | `to_kwant(src)` | `.to_kwant()` | `kwant` |

- **pybinding & Kwant are built straight from the sparse COO** (no dense
  matrix), so they scale to large `num_wann`. `_hr.dat` and HDF5 are **dense**
  on-disk formats (size ≈ `num_R · num_wann²`) and are guarded for very large
  systems — pass `max_wann=` to override the guard if you really mean it.
- pybinding / Kwant / PythTB are optional and imported lazily:
  `pip install pybinding-dev`, `conda install -c conda-forge kwant`,
  `pip install pythtb`.
- `SparseHR` also has `to_tbmodels()` / `to_hdf5()` / `to_hr_dat()` / `to_pb()` /
  `to_kwant()` methods if you already hold the object; the top-level functions are
  the format-agnostic entry points.

Once converted (or via `as_tbmodels`), every post-processing calculator above
(`BulkDOS`, `SurfaceGreensFunction`, `bulk_band_structure`, …) works unchanged.

---

## API reference (top-level imports)

```python
# HTTP client + HDF5 loader
tw_api_call(structure, user, password, output_path, filename,
            output_format="auto", ...)      # "auto" | "sparse" | "hdf5"
tb_model.load(path_to_hdf5)
remaining_credits(user, password)

# Sparse Hamiltonian (from output_format="sparse") + format-detecting converters
SparseHR.load(path_to_npz)          # -> SparseHR: .Hk, .eigsh_near_fermi, .hr_dict, .eigvals_grid,
                                    #    .to_tbmodels/.to_hdf5/.to_hr_dat/.to_pb/.to_kwant
as_tbmodels(src)                    # src = SparseHR/.npz  OR  tbmodels.Model/.hdf5/_hr.dat
to_hr_dat(src, path)   to_hdf5(src, path)
to_pb(src)   to_pythtb(src)   to_kwant(src)

# Heads-only inference model
HeadsOnly(irreps_in)
CovariantOnsiteHead(irreps_in)
CovariantEdgeHead(irreps_in)
load_heads_only_checkpoint(path)
save_heads_only_checkpoint(full_state_dict, irreps_in_str, save_path)

# Subspace fine-tuning  (give ONE of graph_output_path / hr_npz_path)
subspace_projection(start_lr, end_lr, num_epochs, energy_range,
                    decay_sigma, device, save_path, embed_path,
                    graph_output_path=None, loss_mode="subspace",
                    *, hr_npz_path=None)     # hr_npz_path = fine-tune from the sparse .npz

# Subspace losses (advanced)
Subspace_H_MSE_Loss(gdata, edge_pred, onsite_pred, e_lo, e_hi)
Subspace_EigLoss(gdata, edge_pred, onsite_pred, kvec, neighbrs, e_lo, e_hi)
Eigenvalue_Only_Loss(gdata, edge_pred, onsite_pred, e_lo, e_hi)
make_eigenvalue_only_data(gdata, kvecs, eigs_per_k, e_lo, e_hi)
build_subspace_active_mask(node_features, onsite_target, e_lo, e_hi)
write_subspace_basis_file(out_path, active_mask, atoms, LM, ...)

# tbmodels.Model assembly from raw head output
build_hr_model     (edge_pred, onsite_pred, gdata, LM, atoms)
build_hr_model_fast(edge_pred, onsite_pred, gdata, LM, atoms)   # vectorized
write_hr_output(hr_model, out_path, fmt="hdf5"|"hr_dat")

# Post-processing calculators (each has a .run() method returning a Result)
BulkDOS(model_or_path, k_mesh, energies, NC, NV, device)
SurfaceSpectralDensity(model_or_path, surface, LZ, energies, k_path, ...)
SurfaceGreensFunction(model_or_path, surface, energies, k_path, thickness, NN, eps, ...)
FermiArcMap(model_or_path, surface, energy, Nx, Ny, thickness, NN, eps, ...)
generate_k_path(k_points, N_path, labels=None, rec_vecs=None)

# Fermi / band-edge helpers (non-metals)
compute_band_edges(model_or_path, k_mesh=(4,4,4))            # -> {"vbm","cbm","gap","is_metal"}
align_to_vbm(model_or_path, k_mesh=(4,4,4),                  # -> new model with VBM = 0
             fermi_level=None, if_metal="warn")

# Constants
NUM_ELEMENTS   # 109
NeighBrs       # [17, 3] integer R-vector table
```

---

## End-to-end example

```python
import numpy as np
from pymatgen.core import Structure
from tailwater import (
    tw_api_call, subspace_projection, tb_model, SurfaceGreensFunction,
)

# 1. Send the structure to the API (one credit, three artifacts)
structure = Structure.from_file("MyMaterial.cif")
paths = tw_api_call(
    structure, user="user", password="pw",
    output_path="./outputs", filename="my_mat",
    project=True,
)

# 2. Fine-tune the heads to fit a near-Fermi window
subspace_projection(
    start_lr=5e-5, end_lr=5e-7, num_epochs=20,
    energy_range=(-2.0, 2.0), decay_sigma=1.0,
    device="cpu",
    save_path="./out_subspace",
    embed_path=paths["embeddings"],
    graph_output_path=paths["graph_output"],
)

# 3. Run surface-GF analysis on the projected hr-model
model = tb_model.load("./out_subspace/embeddings_pred.hdf5")
result = SurfaceGreensFunction(
    model, surface=np.eye(3),
    energies=np.linspace(-1, 1, 201),
    k_path=[[0, 0.5, 0], [0, 0, 0], [0.333, 0.333, 0]],
    k_labels=["M", r"$\Gamma$", "K"],
).run()
result.figure_top.savefig("surface_top.png")
```

See `examples/` for runnable scripts covering each layer in isolation.

---

## License

Apache 2.0.

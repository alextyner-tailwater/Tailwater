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

If you intend to use `subspace_projection` (writes a projected hr-model HDF5)
or `tb_model.load(...).to_pb()`, also install pybinding directly:

```bash
pip install pybinding-dev
```

`pybinding-dev` is a separate package, not a `tailwater` extra — installs the
same way it does on its own.

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

# (c) project bundle: all three artifacts + .win in a single call
paths = tw_api_call(structure, "user", "pw", "./outputs", "my_mat",
                    project=True)
# paths = {"hdf5": "...", "embeddings": "...",
#          "graph_output": "...", "win": "..."}
```

Five output modes are available — `return_embeddings`, `return_input`, `return_graph_output`, `project`, or default HDF5. Set them as keyword arguments to `tw_api_call`. See `tailwater.tw_api_call.__doc__` for the priority order.

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

## API reference (top-level imports)

```python
# HTTP client + HDF5 loader
tw_api_call(structure, user, password, output_path, filename, ...)
tb_model.load(path_to_hdf5)
remaining_credits(user, password)

# Heads-only inference model
HeadsOnly(irreps_in)
CovariantOnsiteHead(irreps_in)
CovariantEdgeHead(irreps_in)
load_heads_only_checkpoint(path)
save_heads_only_checkpoint(full_state_dict, irreps_in_str, save_path)

# Subspace fine-tuning
subspace_projection(start_lr, end_lr, num_epochs, energy_range,
                    decay_sigma, device, save_path,
                    embed_path, graph_output_path, loss_mode="subspace")

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

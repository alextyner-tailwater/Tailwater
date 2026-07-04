"""Minimal API call: upload a pymatgen Structure, download the Hamiltonian.

By default the Hamiltonian comes back as a sparse ``wannier90_hr.npz``
(a ``tailwater.SparseHR``); for small systems (< 30 atoms) it is also
auto-converted to a dense tbmodels HDF5. Pass ``output_format="hdf5"`` to
force dense, or ``"sparse"`` to always keep the ``.npz``.

Prerequisites: server-side account provisioned with one or more credits.
"""

from pymatgen.core import Structure

from tailwater import tw_api_call


def main():
    structure = Structure.from_file("MyMaterial.cif")

    # tw_api_call always returns a dict of paths. The default ("auto")
    # keeps the sparse .npz (under "npz") and, for small systems, adds a
    # dense HDF5 (under "hdf5"). "win" is the canonical .win the server
    # parsed and ran inference on.
    paths = tw_api_call(
        structure   = structure,
        user        = "acme-research",
        password    = "...",
        output_path = "./outputs",
        filename    = "my_material",
    )
    print(f"sparse .npz -> {paths['npz']}")
    if "hdf5" in paths:                       # small systems (< 30 atoms)
        print(f"dense HDF5  -> {paths['hdf5']}")
    print(f".win        -> {paths['win']}")


if __name__ == "__main__":
    main()

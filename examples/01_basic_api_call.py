"""Minimal API call: upload a pymatgen Structure, download an HDF5 hr-model.

Prerequisites: server-side account provisioned with one or more credits.
"""

from pymatgen.core import Structure

from tailwater import tw_api_call


def main():
    structure = Structure.from_file("MyMaterial.cif")

    # tw_api_call now ALWAYS returns a dict of paths. The default mode
    # returns the HDF5 hr-model alongside the canonical .win file the
    # server parsed and ran inference on.
    paths = tw_api_call(
        structure   = structure,
        user        = "acme-research",
        password    = "...",
        output_path = "./outputs",
        filename    = "my_material",
    )
    print(f"hr-model -> {paths['hdf5']}")
    print(f".win     -> {paths['win']}")


if __name__ == "__main__":
    main()

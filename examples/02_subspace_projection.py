"""Full end-to-end: API project bundle -> subspace fine-tune -> projected hr-model.

One credit pulls the project bundle (embeddings.pt + the sparse
wannier90_hr.npz); the fine-tune refines the heads to reproduce the
predicted Hamiltonian's eigenvalues inside a near-Fermi energy window
(eigenvalue-only downfolding).
"""

from pymatgen.core import Structure

from tailwater import subspace_projection, tw_api_call


def main():
    structure = Structure.from_file("MyMaterial.cif")

    # 1) One API call, one credit -> embeddings.pt + sparse wannier90_hr.npz.
    paths = tw_api_call(
        structure   = structure,
        user        = "acme-research",
        password    = "...",
        output_path = "./outputs",
        filename    = "my_material",
        project     = True,
    )
    print(f"API bundle paths: {paths}")

    # 2) Fine-tune heads + project to [-2, 2] eV around E_F. The sparse .npz
    #    Hamiltonian is the fit target (its in-window eigenvalues).
    final_ckpt = subspace_projection(
        start_lr     = 1e-4,
        end_lr       = 1e-5,
        num_epochs   = 20,
        energy_range = (-2.0, 2.0),
        decay_sigma  = 0.5,
        device       = "cpu",
        save_path    = "./projection_out",
        embed_path   = paths["embeddings"],
        hr_npz_path  = paths["npz"],
    )
    print(f"Fine-tuned heads -> {final_ckpt}")


if __name__ == "__main__":
    main()

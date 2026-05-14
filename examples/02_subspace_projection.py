"""Full end-to-end: API project bundle -> subspace fine-tune -> projected hr-model.

One credit pulls all three artifacts from the API; the fine-tune refines
the heads to fit a near-Fermi energy window using the model's own
pre-fine-tune predictions as the target (self-distillation downfolding).
"""

from pymatgen.core import Structure

from tailwater import subspace_projection, tw_api_call


def main():
    structure = Structure.from_file("MyMaterial.cif")

    # 1) One API call, one credit, all three artifacts on disk.
    paths = tw_api_call(
        structure   = structure,
        user        = "acme-research",
        password    = "...",
        output_path = "./outputs",
        filename    = "my_material",
        project     = True,
    )
    print(f"API bundle paths: {paths}")

    # 2) Fine-tune heads + project to [-2, 2] eV around E_F.
    final_ckpt = subspace_projection(
        start_lr          = 5e-5,
        end_lr            = 5e-7,
        num_epochs        = 20,
        energy_range      = (-2.0, 2.0),
        decay_sigma       = 1.0,
        device            = "cpu",
        save_path         = "./projection_out",
        embed_path        = paths["embeddings"],
        graph_output_path = paths["graph_output"],
    )
    print(f"Fine-tuned heads -> {final_ckpt}")


if __name__ == "__main__":
    main()

"""Smoke test: verify every top-level export is importable.

Doesn't exercise the API or any heavy compute — just imports the
package and asserts the headline names exist. Catches packaging /
relative-import regressions early.

Run with: pytest tests/
"""

import importlib


def test_top_level_import():
    mod = importlib.import_module("tailwater")
    assert hasattr(mod, "__version__")


def test_client_exports():
    from tailwater import (
        remaining_credits,
        tb_model,
        tw_api_call,
    )
    assert callable(tw_api_call)
    assert callable(tb_model.load)
    assert callable(remaining_credits)


def test_heads_only_exports():
    from tailwater import (
        CovariantEdgeHead,
        CovariantOnsiteHead,
        HeadsOnly,
        load_heads_only_checkpoint,
        save_heads_only_checkpoint,
    )
    assert callable(HeadsOnly)
    assert callable(load_heads_only_checkpoint)
    assert callable(save_heads_only_checkpoint)
    assert callable(CovariantOnsiteHead)
    assert callable(CovariantEdgeHead)


def test_subspace_exports():
    from tailwater import (
        Eigenvalue_Only_Loss,
        Subspace_EigLoss,
        Subspace_H_MSE_Loss,
        build_subspace_active_mask,
        make_eigenvalue_only_data,
        subspace_projection,
        write_subspace_basis_file,
    )
    for fn in (
        subspace_projection, Subspace_H_MSE_Loss, Subspace_EigLoss,
        Eigenvalue_Only_Loss, make_eigenvalue_only_data,
        build_subspace_active_mask, write_subspace_basis_file,
    ):
        assert callable(fn), fn


def test_post_processing_exports():
    from tailwater import (
        BandStructure,
        BulkDOS,
        FermiArcMap,
        SurfaceGreensFunction,
        SurfaceSpectralDensity,
        bulk_band_structure,
        generate_k_path,
    )
    for cls in (BulkDOS, SurfaceSpectralDensity,
                SurfaceGreensFunction, FermiArcMap, BandStructure):
        assert callable(cls)
    assert callable(generate_k_path)
    assert callable(bulk_band_structure)


def test_constants():
    from tailwater import NUM_ELEMENTS, NeighBrs
    assert NUM_ELEMENTS == 109
    assert NeighBrs.shape == (17, 3)

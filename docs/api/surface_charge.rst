Surface charge density
======================

Real-space surface charge-density heat maps of a general ``(hkl)`` slab,
built directly from a Wannier Hamiltonian's real-space ``H(R)``. Works on
**any** Wannier tight-binding model — a Tailwater prediction *or* a
DFT-generated Wannier90 Hamiltonian — because the only interchange format
it needs is ``H(R)``.

Pipeline: re-express ``H(R)`` in an integer supercell whose first two
lattice vectors lie in the ``(hkl)`` plane (an exact, determinant-preserving
remap), stack ``size`` cells along the surface normal and drop hoppings that
leave the slab, integrate :math:`|\psi|^2` of the occupied states over the
surface BZ to get a per-orbital occupation, then render
:math:`\rho(\mathbf{r}) = \sum_g n_g\,\mathcal{G}(\mathbf{r}-\mathbf{r}_g)`
with the Wannier centres as :math:`\mathbf{r}_g`.

Quick example
-------------

.. code-block:: python

    from tailwater import surface_charge_density, load_hr, supercell_self_check

    # `model` accepts a tbmodels.Model, an HDF5 path, a Wannier90 *_hr.dat
    # (DFT output), or the dict returned by load_hr().
    HR_PATH = "outputs/wannier90_hr.hdf5"
    MILLER  = (0, 0, 1)     # surface Miller index
    SIZE    = 4             # slab thickness in unit cells

    # Sanity-gate the general-(hkl) supercell remap (expect ~1e-13 eV).
    model = load_hr(HR_PATH)
    assert supercell_self_check(model, MILLER) < 1e-8

    # Top-view + side cross-section heat maps; everything past `size` is optional.
    res = surface_charge_density(
        model, MILLER, SIZE,
        mu=0.0, nk=12, sigma=0.6, tile=3,
        savepath="surface_charge_001.png",
    )
    rho, top_img, side_img = res["rho"], res["top_img"], res["side_img"]

Image a topological surface state by restricting the occupation to a narrow
window around :math:`E_F`:

.. code-block:: python

    surface_charge_density(model, MILLER, SIZE, energy_window=(-0.1, 0.1),
                           savepath="surface_charge_001_tss.png")

A DFT Wannier90 model drops in unchanged — pass the ``*_hr.dat`` path
directly:

.. code-block:: python

    surface_charge_density("path/to/wannier90_hr.dat", (1, 1, 1), 5)

See ``examples/11_surface_charge_density.py`` for the full runnable script.

.. note::

   The per-k diagonalisation loop can stall under some OpenBLAS builds. The
   function limits BLAS threads via ``threadpoolctl`` when it is installed;
   otherwise set ``OMP_NUM_THREADS=1`` in the environment.

API
---

.. autofunction:: tailwater.surface_charge.surface_charge_density

.. autofunction:: tailwater.surface_charge.load_hr

.. autofunction:: tailwater.surface_charge.supercell_self_check

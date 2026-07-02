"""Format-detecting Hamiltonian converters.

One function per target format that accepts *either* representation and
dispatches automatically:

* a sparse :class:`tailwater.sparse.SparseHR` (or a ``.npz`` path) — the
  optimized-inference output; converters build straight from the COO where the
  target library is sparse-native (pybinding / Kwant), so large systems never
  densify.
* a dense ``tbmodels.Model`` (or a ``.hdf5`` / Wannier90 ``_hr.dat`` path) — the
  classic dense output; converters reuse the existing tbmodels-based helpers
  (``tb_model.load``'s ``to_pb`` / ``to_pythtb`` / ``to_kwant`` methods and
  ``write_hr_output``).

This means the SAME command works regardless of which format the user is
holding::

    from tailwater import to_hr_dat, to_pb, to_kwant
    to_hr_dat("wannier90_hr.npz",  "wannier90_hr.dat")   # sparse input
    to_hr_dat("wannier90_hr.hdf5", "wannier90_hr.dat")   # dense input, same call
    pb_model  = to_pb("wannier90_hr.npz")
    syst, lat = to_kwant("wannier90_hr.npz")

Detection mirrors the existing ``surface_charge.load_hr`` extension idiom. No
existing public name is shadowed: the dense path already exposed ``to_pb`` /
``to_pythtb`` / ``to_kwant`` as *instance methods* on the model returned by
``tb_model.load`` and ``write_hr_output`` as the dense H(R) writer; these
functions are new top-level dispatchers layered on top, and both underlying
paths keep working unchanged.
"""
import os
import warnings

__all__ = ["to_pb", "to_pythtb", "to_kwant", "to_hr_dat", "to_hdf5", "as_tbmodels"]


def _resolve(src):
    """Normalise ``src`` to ``("sparse", SparseHR)`` or ``("dense", tbmodels.Model)``.

    ``src`` may be a :class:`SparseHR`, a ``tbmodels.Model``, or a path whose
    extension selects the loader (``.npz`` -> sparse; ``.hdf5``/``.h5`` or a
    Wannier90 ``_hr.dat`` -> dense).
    """
    from .sparse import SparseHR
    if isinstance(src, SparseHR):
        return "sparse", src

    try:
        import tbmodels
        if isinstance(src, tbmodels.Model):
            return "dense", src
    except ImportError:
        pass

    p = str(src)
    low = p.lower()
    if low.endswith(".npz"):
        return "sparse", SparseHR.load(p)
    if low.endswith((".hdf5", ".h5")):
        import tbmodels
        return "dense", tbmodels.Model.from_hdf5_file(p)
    if low.endswith(".dat") or "_hr" in os.path.basename(low):
        import tbmodels
        return "dense", tbmodels.Model.from_wannier_files(hr_file=p)
    raise ValueError(
        f"Cannot infer Hamiltonian format from {src!r}. Pass a SparseHR, a "
        "tbmodels.Model, or a path ending in .npz (sparse) / .hdf5 / _hr.dat "
        "(dense).")


def _warn_ignored_lattice(kind, lattice_vectors):
    if kind == "sparse" and lattice_vectors is not None:
        warnings.warn(
            "lattice_vectors is only honored for dense (tbmodels) inputs; a "
            "SparseHR uses its own stored cell. Ignoring lattice_vectors.",
            stacklevel=3)


def to_pb(src, lattice_vectors=None, hop_threshold=1e-12):
    """Convert ``src`` to a pybinding ``pb.Lattice`` (auto-detects sparse/dense).

    For a sparse input the lattice is built straight from the COO (no dense
    matrix); ``lattice_vectors`` is honored only for the dense path."""
    kind, m = _resolve(src)
    _warn_ignored_lattice(kind, lattice_vectors)
    if kind == "sparse":
        return m.to_pb()
    from .client import _to_pb_method
    return _to_pb_method(m, lattice_vectors=lattice_vectors,
                         hop_threshold=hop_threshold)


def to_pythtb(src, lattice_vectors=None, hop_threshold=1e-12):
    """Convert ``src`` to a PythTB model (auto-detects sparse/dense).

    PythTB is inherently dense, so a sparse input is densified via tbmodels
    first (small/medium systems only)."""
    from .client import _to_pythtb_method
    kind, m = _resolve(src)
    if kind == "sparse":
        m = m.to_tbmodels()
    return _to_pythtb_method(m, lattice_vectors=lattice_vectors,
                             hop_threshold=hop_threshold)


def to_kwant(src, lattice_vectors=None, hop_threshold=1e-12):
    """Convert ``src`` to a Kwant ``kwant.Builder`` (auto-detects sparse/dense).

    For a sparse input the builder carries matrix-valued blocks built straight
    from the COO (scales to large num_wann) and returns ``(Builder, lattice)``;
    ``lattice_vectors`` is honored only for the dense path."""
    kind, m = _resolve(src)
    _warn_ignored_lattice(kind, lattice_vectors)
    if kind == "sparse":
        return m.to_kwant()
    from .client import _to_kwant_method
    return _to_kwant_method(m, lattice_vectors=lattice_vectors,
                            hop_threshold=hop_threshold)


def to_hr_dat(src, path, **kw):
    """Write ``src`` to a Wannier90 ``_hr.dat`` (auto-detects sparse/dense).

    DENSE on-disk format (``~ num_R * num_wann**2``). For a sparse input the
    ``max_wann`` guard applies (pass ``max_wann=`` to override); prefer keeping
    large systems sparse."""
    kind, m = _resolve(src)
    if kind == "sparse":
        return m.to_hr_dat(path, **kw)
    from .hr_export import write_hr_output
    return write_hr_output(m, str(path), fmt="hr_dat")


def to_hdf5(src, path, **kw):
    """Write ``src`` to a tbmodels HDF5 (auto-detects sparse/dense).

    DENSE on-disk format. For a sparse input the ``max_wann`` guard applies
    (pass ``max_wann=`` to override)."""
    kind, m = _resolve(src)
    if kind == "sparse":
        return m.to_hdf5(path, **kw)
    from .hr_export import write_hr_output
    return write_hr_output(m, str(path), fmt="hdf5")


def as_tbmodels(src, uc=None):
    """Return ``src`` as a ``tbmodels.Model`` (the sparse -> dense bridge).

    A dense input is returned unchanged; a sparse input is densified (``uc``
    defaults to its stored cell)."""
    kind, m = _resolve(src)
    if kind == "sparse":
        return m.to_tbmodels(uc=uc)
    return m

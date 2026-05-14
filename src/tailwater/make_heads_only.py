"""Supplier-side helper: NOT included in the customer-facing API surface.

`make_heads_only.py` builds a `HeadsOnly.pth` checkpoint by stripping the
proprietary backbone weights out of a full WanE3Lite training checkpoint.
It needs access to `model_lite.py` (the proprietary backbone code), which
is NOT shipped with this package — so the script is intentionally a stub
on the customer side.

Customers receive a pre-built `HeadsOnly.pth` from the supplier alongside
this package and load it with `tailwater.load_heads_only_checkpoint(...)`.

If you ARE the supplier and need to build a HeadsOnly checkpoint, run the
full version of this script that lives in the supplier's repo, not here.

The companion helper IS shipped — `tailwater.save_heads_only_checkpoint`
takes a raw `state_dict` and writes the checkpoint without requiring
`model_lite`, so a downstream toolchain that already has the irreps
string in hand can build the checkpoint directly:

    from tailwater import save_heads_only_checkpoint
    save_heads_only_checkpoint(
        full_state_dict = checkpoint["model_state_dict"],
        irreps_in_str   = "96x0e+48x0o+32x1o+32x1e+...",
        save_path       = "HeadsOnly.pth",
    )
"""

raise ImportError(
    "tailwater.make_heads_only is a supplier-side tool that depends on "
    "`model_lite` (the proprietary backbone code) and is intentionally "
    "stubbed in the customer-facing package. Use "
    "`tailwater.save_heads_only_checkpoint(...)` directly if you have "
    "the state_dict and irreps string in hand."
)

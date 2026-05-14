"""Heads-only deployment model.

This file is the customer-shippable counterpart to model_lite.py. It
contains ONLY the equivariant output heads (CovariantOnsiteHead and
CovariantEdgeHead) and a thin wrapper (HeadsOnly) that consumes
pre-computed embeddings produced by the backbone.

Self-contained on purpose
-------------------------
The head classes are duplicated from model_lite.py rather than imported,
so this file can be shipped to customers without also shipping the
proprietary backbone code in model_lite.py. The duplicated code is the
universal physics part — Clebsch-Gordan-based basis construction,
time-reversal projection, Hermiticity for onsite, gate nonlinearities.
It contains no architectural decisions specific to your backbone.

Deployment contract
-------------------
The customer receives:
  1. This file (heads_only_model.py)
  2. A HeadsOnly checkpoint (head weights + irreps metadata)
  3. An embeddings dataset — PyG-style Data objects with extra fields
     `f_out` (per-atom backbone features) and `edge_feat` (per-edge
     backbone features). All other Data fields (node_features,
     edge_index, edge_vectors, edge_targets, inv_data, atom_number, idn)
     are passed through unchanged.

The HeadsOnly forward signature is the same as a full model — takes a
Data object and returns (edge_pred, onsite_pred) — except that the Data
object MUST contain `f_out` and `edge_feat` attributes pre-populated by
the backbone. No backbone forward pass happens inside HeadsOnly.
"""

import math

import torch
import torch.nn as torch_nn
import torch.serialization
torch.serialization.add_safe_globals([slice])
from e3nn import o3, nn as e3nn_nn


# =====================================================
# EQUIVARIANT OUTPUT HEADS  (duplicated from model_lite.py)
# =====================================================

class CovariantOnsiteHead(torch.nn.Module):
    def __init__(self, irreps_in, device=None):
        super().__init__()
        self.device = device

        C_mat_total_local, total_irreps_str = self._build_18x18_basis()
        self.total_irreps = o3.Irreps(total_irreps_str).simplify()
        self.register_buffer("C_mat_total", C_mat_total_local)

        U_T_2x2 = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.complex64)
        U_T_18x18 = torch.block_diag(*[U_T_2x2 for _ in range(9)])
        self.register_buffer("U_T_18x18", U_T_18x18)

        irreps_scalars = o3.Irreps("64x0e")
        irreps_gated = o3.Irreps(
            "64x0o + 32x1o + 32x1e + 25x2o + 25x2e + "
            "18x3o + 18x3e + 9x4o + 9x4e + 4x5o + 4x5e"
        )
        irreps_gates = o3.Irreps("240x0e")

        self.lin1 = o3.Linear(irreps_in, irreps_scalars + irreps_gates + irreps_gated)
        self.gate = e3nn_nn.Gate(
            irreps_scalars, [torch.nn.functional.silu],
            irreps_gates,   [torch.sigmoid],
            irreps_gated
        )
        self.lin2 = o3.Linear(self.gate.irreps_out, self.total_irreps + self.total_irreps)

    def _build_18x18_basis(self):
        basis = [(0, 1), (1, -1), (2, 1)]  # (L, parity) for s, p, d

        spatial_irreps_list = []
        C_mat_spat = torch.zeros(81, 9, 9)
        idx_spat = 0
        row_start = 0
        for l1, p1 in basis:
            dim1 = 2 * l1 + 1
            col_start = 0
            for l2, p2 in basis:
                dim2 = 2 * l2 + 1
                for L in range(abs(l1 - l2), l1 + l2 + 1):
                    p_out = p1 * p2
                    par = "e" if p_out == 1 else "o"
                    spatial_irreps_list.append((f"1x{L}{par}", L, p_out))
                    w3j = o3.wigner_3j(l1, l2, L)
                    cg = w3j * math.sqrt(2 * L + 1)
                    dim_L = 2 * L + 1
                    C_mat_spat[idx_spat: idx_spat + dim_L,
                               row_start: row_start + dim1,
                               col_start: col_start + dim2] = cg.permute(2, 0, 1)
                    idx_spat += dim_L
                col_start += dim2
            row_start += dim1

        perm = torch.tensor([0, 2, 3, 1, 6, 7, 5, 8, 4])
        C_mat_spat = C_mat_spat[:, perm, :][:, :, perm]

        C_mat_I = torch.zeros(81, 18, 18, dtype=torch.complex64)
        I_mat = torch.eye(2, dtype=torch.complex64)
        for k in range(81):
            spatial_k = C_mat_spat[k].to(torch.complex64)
            C_mat_I[k] = torch.einsum('ij, ab -> iajb', spatial_k, I_mat).reshape(18, 18)

        total_irreps_str = "+".join([t[0] for t in spatial_irreps_list])

        sigma_y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
        sigma_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)
        sigma_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
        sigma_e3nn = torch.stack([sigma_y, sigma_z, sigma_x])

        C_mat_sigma_list = []
        idx_spat = 0
        for irrep_str, L1, p1 in spatial_irreps_list:
            dim1 = 2 * L1 + 1
            for L_out in range(abs(L1 - 1), L1 + 1 + 1):
                p_out = p1 * 1
                par = "e" if p_out == 1 else "o"
                total_irreps_str += f"+ 1x{L_out}{par}"
                w3j = o3.wigner_3j(L1, 1, L_out)
                cg = (w3j * math.sqrt(2 * L_out + 1)).permute(2, 0, 1)
                dim_out = 2 * L_out + 1
                C_mat_block = torch.zeros(dim_out, 18, 18, dtype=torch.complex64)
                for m_out in range(dim_out):
                    mat_18x18 = torch.zeros(18, 18, dtype=torch.complex64)
                    for m1 in range(dim1):
                        for m_spin in range(3):
                            c_val = cg[m_out, m1, m_spin]
                            if abs(c_val) > 1e-7:
                                spatial_mat = C_mat_spat[idx_spat + m1].to(torch.complex64)
                                spin_mat = sigma_e3nn[m_spin]
                                term = torch.einsum('ij, ab -> iajb', spatial_mat, spin_mat).reshape(18, 18)
                                mat_18x18 += c_val * term
                    C_mat_block[m_out] = mat_18x18
                C_mat_sigma_list.append(C_mat_block)
            idx_spat += dim1

        C_mat_sigma = torch.cat(C_mat_sigma_list, dim=0)
        C_mat_total = torch.cat([C_mat_I, C_mat_sigma], dim=0)
        return C_mat_total, total_irreps_str

    def forward(self, f_in):
        x = self.lin1(f_in)
        x = self.gate(x)
        coeffs = self.lin2(x)
        n = coeffs.shape[-1] // 2
        coeffs_c = coeffs[:, :n].to(torch.complex64) + 1j * coeffs[:, n:].to(torch.complex64)
        H_c = torch.einsum('n k, k i j -> n i j', coeffs_c, self.C_mat_total)

        H_TR = torch.matmul(self.U_T_18x18,
                            torch.matmul(torch.conj(H_c), -self.U_T_18x18))
        H_c = 0.5 * (H_c + H_TR)
        H_c = 0.5 * (H_c + torch.conj(H_c).transpose(-1, -2))
        return torch.stack([H_c.real, H_c.imag], dim=-1)


class CovariantEdgeHead(torch.nn.Module):
    def __init__(self, irreps_in, device=None):
        super().__init__()
        self.device = device

        C_mat_total_local, total_irreps_str = self._build_18x18_basis()
        self.total_irreps = o3.Irreps(total_irreps_str).simplify()
        self.register_buffer("C_mat_total", C_mat_total_local)

        U_T_2x2 = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.complex64)
        U_T_18x18 = torch.block_diag(*[U_T_2x2 for _ in range(9)])
        self.register_buffer("U_T_18x18", U_T_18x18)

        irreps_scalars = o3.Irreps("64x0e")
        irreps_gated = o3.Irreps(
            "64x0o + 32x1o + 32x1e + 25x2o + 25x2e + "
            "18x3o + 18x3e + 9x4o + 9x4e + 4x5o + 4x5e"
        )
        irreps_gates = o3.Irreps("240x0e")

        self.lin1 = o3.Linear(irreps_in, irreps_scalars + irreps_gates + irreps_gated)
        self.gate = e3nn_nn.Gate(
            irreps_scalars, [torch.nn.functional.silu],
            irreps_gates,   [torch.sigmoid],
            irreps_gated
        )
        self.lin2 = o3.Linear(self.gate.irreps_out, self.total_irreps + self.total_irreps)

    _build_18x18_basis = CovariantOnsiteHead._build_18x18_basis

    def forward(self, f_in):
        x = self.lin1(f_in)
        x = self.gate(x)
        coeffs = self.lin2(x)
        n = coeffs.shape[-1] // 2
        coeffs_c = coeffs[:, :n].to(torch.complex64) + 1j * coeffs[:, n:].to(torch.complex64)
        H_c = torch.einsum('n k, k i j -> n i j', coeffs_c, self.C_mat_total)
        H_TR = torch.matmul(self.U_T_18x18,
                            torch.matmul(torch.conj(H_c), -self.U_T_18x18))
        H_c = 0.5 * (H_c + H_TR)
        return torch.stack([H_c.real, H_c.imag], dim=-1)


# =====================================================
# HEADS-ONLY MODEL (the wrapper customers see)
# =====================================================

class HeadsOnly(torch.nn.Module):
    """Customer-facing model: consumes embeddings, produces H predictions.

    Replaces the full WanE3Lite at the deployment boundary. The customer
    instantiates this with the same `irreps_in` that the backbone produced,
    loads head weights from a HeadsOnly checkpoint, and trains / runs
    inference using PyG Data objects that have `f_out` and `edge_feat`
    populated by the backbone in advance.

    Forward signature: matches the full model so downstream training and
    extraction scripts (extract_hr_subspace.py, finetune_subspace.py)
    can use the HeadsOnly model in place of WanE3Lite without changes
    apart from the data-loading path.
    """
    def __init__(self, irreps_in):
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in) if isinstance(irreps_in, str) else irreps_in
        self.CovariantOnsiteHead = CovariantOnsiteHead(irreps_in=self.irreps_in)
        self.CovariantEdgeHead   = CovariantEdgeHead(irreps_in=self.irreps_in)

    def forward(self, data):
        """Forward pass on pre-computed embeddings.

        Args
        ----
        data : PyG Data object with REQUIRED attributes
                 .f_out      : [N, irreps_in.dim] per-atom backbone features
                 .edge_feat  : [E, irreps_in.dim] per-edge backbone features
               (all other Data fields like node_features, edge_index, etc.
               are not used by the heads but pass through for downstream
               loss / H(k) construction.)

        Returns
        -------
        (edge_pred, onsite_pred) — same shapes as the full model.
        """
        if not hasattr(data, "f_out") or not hasattr(data, "edge_feat"):
            raise RuntimeError(
                "HeadsOnly requires data.f_out and data.edge_feat to be "
                "populated. Did you forget to run the backbone-side "
                "embedding extraction first? See extract_embeddings.py."
            )
        # Shape sanity check — catches accidental mismatches when the
        # checkpoint and the embedding dataset were built with different
        # backbone irreps.
        if data.f_out.shape[-1] != self.irreps_in.dim:
            raise RuntimeError(
                f"Embedding dim {data.f_out.shape[-1]} != heads' irreps_in.dim "
                f"{self.irreps_in.dim}. The HeadsOnly checkpoint was built for "
                f"a backbone with different irreps than the embeddings you're "
                f"feeding in."
            )

        onsite_pred = self.CovariantOnsiteHead(data.f_out)
        edge_pred   = self.CovariantEdgeHead(data.edge_feat)
        return edge_pred, onsite_pred


# =====================================================
# CHECKPOINT HELPERS
# =====================================================

def save_heads_only_checkpoint(full_state_dict, irreps_in_str, save_path):
    """Build a HeadsOnly-only checkpoint by stripping backbone weights.

    Args
    ----
    full_state_dict : dict — the model_state_dict from your full WanE3Lite
                      (or compatible) training checkpoint.  May or may not
                      have the DDP 'module.' prefix.
    irreps_in_str   : str — the backbone's output irreps as a string,
                      e.g. "192x0e + 96x0o + 32x1o + 32x1e + ...".
                      This is what HeadsOnly will use to size its head
                      Linear layers and must match the embedding tensor's
                      last-dim shape.
    save_path       : output .pth filename.
    """
    cleaned = {}
    for k, v in full_state_dict.items():
        key = k.replace("module.", "", 1) if k.startswith("module.") else k
        # Keep only the head parameters and their internal buffers
        # (C_mat_total, U_T_18x18). Backbone keys (node_embed, tp1, blocks,
        # tp_edge, etc.) are dropped here — the customer never sees them.
        if key.startswith("CovariantOnsiteHead.") or key.startswith("CovariantEdgeHead."):
            cleaned[key] = v.detach().cpu()

    torch.save({
        "heads_state_dict": cleaned,
        "irreps_in":        irreps_in_str,
    }, save_path)
    return cleaned


def load_heads_only_checkpoint(path, map_location=None):
    """Construct + load a HeadsOnly model from a HeadsOnly checkpoint.

    Returns the HeadsOnly module with weights loaded.
    """
    ckpt = torch.load(path, map_location=map_location)
    irreps_in = ckpt["irreps_in"]
    model = HeadsOnly(irreps_in=irreps_in)
    missing, unexpected = model.load_state_dict(ckpt["heads_state_dict"],
                                                strict=False)
    if len(unexpected) > 0:
        print(f"[HeadsOnly] unexpected keys in checkpoint: {unexpected}")
    # `missing` is expected to contain the registered buffers C_mat_total
    # and U_T_18x18 if the checkpoint omitted them (they're deterministic
    # and recomputed at construction), so we don't warn on it.
    return model

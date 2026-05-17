"""Quantile positioning schemes for QR-DQN.

QR-DQN parametrizes a return distribution by predicting its inverse CDF at a
set of quantile fractions tau_i in [0, 1]. The expected value is

    E[Z] = integral_0^1 F^{-1}(tau) d tau

which we approximate with a quadrature rule defined by (tau_i, w_i). The
default vanilla QR-DQN uses the midpoint rule (uniform tau, uniform w).
This module also supports:

  - midpoint       : tau_i = (i + 0.5) / N, w_i = 1/N  (vanilla QR-DQN)
  - gauss_legendre : Gauss-Legendre nodes on [0,1], non-uniform tau and w
  - trapezoidal    : uniform tau including endpoints; fixed q_min/q_max at
                     tau=0/1; network only predicts the N-2 interior points

Risk-aware truncation (truncate_upper_quantiles): when used with the CVaR
risk policy, the network only predicts the bottom k = ceil(N * risk_fraction)
quantiles uniformly in [0, risk_fraction]. The upper part of the distribution
is never consulted at action time, so dropping it concentrates network
capacity on the part that actually matters for the risk policy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class QuantileScheme:
    """Concrete quantile placement and integration weights.

    Attributes:
        tau:            Quantile fractions used in the QR loss
                        (only for the network-predicted quantiles). Shape
                        [num_predicted].
        mean_weights:   Integration weights for the FULL distribution (i.e.,
                        predicted + fixed endpoints if any). Sum to 1. Shape
                        [num_total].
        cvar_weights:   Integration weights for the bottom-risk_fraction lobe
                        (zero above the cutoff), renormalized so they sum to
                        1 over the cutoff. Shape [num_total]. None if no
                        risk_fraction was provided.
        num_predicted:  How many quantile values the network's head outputs.
        fixed_lo:       Value of F^{-1}(0). Only set for trapezoidal mode.
        fixed_hi:       Value of F^{-1}(1). Only set for trapezoidal mode.
        mode:           Human-readable name (for logging / checkpoints).
    """

    tau: torch.Tensor
    mean_weights: torch.Tensor
    cvar_weights: torch.Tensor | None
    num_predicted: int
    fixed_lo: float | None
    fixed_hi: float | None
    mode: str

    @property
    def num_total(self) -> int:
        return int(self.mean_weights.numel())

    @property
    def has_fixed_endpoints(self) -> bool:
        return self.fixed_lo is not None

    def assemble_full(self, predicted: torch.Tensor) -> torch.Tensor:
        """Prepend / append fixed endpoint values if this scheme uses them.

        Args:
            predicted: shape [..., num_predicted].

        Returns:
            Shape [..., num_total]. For schemes without fixed endpoints this
            is the input unchanged.
        """
        if not self.has_fixed_endpoints:
            return predicted
        lead = predicted.shape[:-1]
        lo = torch.full(
            (*lead, 1),
            float(self.fixed_lo),
            dtype=predicted.dtype,
            device=predicted.device,
        )
        hi = torch.full(
            (*lead, 1),
            float(self.fixed_hi),
            dtype=predicted.dtype,
            device=predicted.device,
        )
        return torch.cat([lo, predicted, hi], dim=-1)

    def expectation(self, predicted: torch.Tensor) -> torch.Tensor:
        """Quadrature estimate of E[Z] over the assembled distribution."""
        full = self.assemble_full(predicted)
        return (full * self.mean_weights).sum(dim=-1)

    def cvar(self, predicted: torch.Tensor) -> torch.Tensor:
        """Quadrature estimate of CVaR_{risk_fraction} over the assembled
        distribution. Falls back to expectation() if no cvar_weights were
        precomputed (this should not be called in that case)."""
        if self.cvar_weights is None:
            return self.expectation(predicted)
        full = self.assemble_full(predicted)
        return (full * self.cvar_weights).sum(dim=-1)


def _compute_cvar_weights(
    full_tau: torch.Tensor,
    full_weights: torch.Tensor,
    risk_fraction: float,
) -> torch.Tensor:
    """Build CVaR_{rf} weights from a full quadrature (tau, w).

    Mass is kept on the support points whose tau <= risk_fraction and
    renormalized to sum to 1.
    """
    mask = full_tau <= risk_fraction
    masked = torch.where(mask, full_weights, torch.zeros_like(full_weights))
    total = masked.sum()
    if total <= 0:
        # Degenerate (no support below risk_fraction): keep first node only.
        out = torch.zeros_like(full_weights)
        out[0] = 1.0
        return out
    return masked / total


def build_scheme(
    mode: str,
    num_quantiles: int,
    device: torch.device,
    *,
    q_min: float = 0.0,
    q_max: float = 50.0,
    risk_type: str = "mean",
    risk_fraction: float = 0.1,
    truncate_upper: bool = False,
) -> QuantileScheme:
    """Construct a QuantileScheme from config knobs.

    Args:
        mode: 'midpoint' | 'gauss_legendre' | 'trapezoidal'.
        num_quantiles: Total grid size N. For trapezoidal the network only
            predicts N-2 of these (the interior).
        device: Torch device for the returned tensors.
        q_min, q_max: Fixed endpoints used only in trapezoidal mode.
        risk_type: 'mean' or 'cvar'. Controls whether cvar_weights are built.
        risk_fraction: For risk_type='cvar', the bottom fraction of the
            distribution to integrate over.
        truncate_upper: If True (only valid with mode='midpoint' and
            risk_type='cvar'), drop the upper (1 - risk_fraction) of the
            quantile grid and place the remaining k = ceil(N * risk_fraction)
            quantiles uniformly in [0, risk_fraction].
    """
    if truncate_upper:
        if risk_type != "cvar":
            raise ValueError(
                "truncate_upper_quantiles=True only makes sense with "
                "risk_type='cvar'"
            )
        if mode != "midpoint":
            raise ValueError(
                "truncate_upper_quantiles=True is only supported with "
                f"quantile_mode='midpoint' (got {mode!r})"
            )
        k = max(1, int(math.ceil(num_quantiles * risk_fraction)))
        tau_np = np.array(
            [(i + 0.5) * risk_fraction / k for i in range(k)],
            dtype=np.float64,
        )
        w_np = np.full(k, 1.0 / k, dtype=np.float64)
        tau = torch.tensor(tau_np, dtype=torch.float32, device=device)
        weights = torch.tensor(w_np, dtype=torch.float32, device=device)
        # cvar over the full (truncated) support IS the mean by construction.
        return QuantileScheme(
            tau=tau,
            mean_weights=weights,
            cvar_weights=weights.clone(),
            num_predicted=k,
            fixed_lo=None,
            fixed_hi=None,
            mode="midpoint_truncated",
        )

    if mode == "midpoint":
        n = num_quantiles
        tau_np = np.array(
            [(i + 0.5) / n for i in range(n)], dtype=np.float64,
        )
        w_np = np.full(n, 1.0 / n, dtype=np.float64)
        tau = torch.tensor(tau_np, dtype=torch.float32, device=device)
        weights = torch.tensor(w_np, dtype=torch.float32, device=device)
        cvar_w = (
            _compute_cvar_weights(tau, weights, risk_fraction)
            if risk_type == "cvar"
            else None
        )
        return QuantileScheme(
            tau=tau,
            mean_weights=weights,
            cvar_weights=cvar_w,
            num_predicted=n,
            fixed_lo=None,
            fixed_hi=None,
            mode="midpoint",
        )

    if mode == "gauss_legendre":
        if num_quantiles < 2:
            raise ValueError("gauss_legendre requires num_quantiles >= 2")
        nodes_np, w_np = np.polynomial.legendre.leggauss(num_quantiles)
        tau_np = (nodes_np + 1.0) / 2.0
        w_np = w_np / 2.0
        # Sort by tau (leggauss returns them ordered, but be explicit).
        order = np.argsort(tau_np)
        tau_np, w_np = tau_np[order], w_np[order]
        tau = torch.tensor(tau_np, dtype=torch.float32, device=device)
        weights = torch.tensor(w_np, dtype=torch.float32, device=device)
        cvar_w = (
            _compute_cvar_weights(tau, weights, risk_fraction)
            if risk_type == "cvar"
            else None
        )
        return QuantileScheme(
            tau=tau,
            mean_weights=weights,
            cvar_weights=cvar_w,
            num_predicted=num_quantiles,
            fixed_lo=None,
            fixed_hi=None,
            mode="gauss_legendre",
        )

    if mode == "trapezoidal":
        n = num_quantiles
        if n < 3:
            raise ValueError("trapezoidal requires num_quantiles >= 3")
        # Full grid: tau_i = i / (n - 1)
        full_tau_np = np.linspace(0.0, 1.0, n, dtype=np.float64)
        # Trapezoidal weights on a uniform grid of n points over [0,1]:
        # interior = 1/(n-1), endpoints = 1/(2(n-1)). Sum = 1.
        full_w_np = np.full(n, 1.0 / (n - 1), dtype=np.float64)
        full_w_np[0] = full_w_np[-1] = 0.5 / (n - 1)
        # The network only predicts interior quantiles (tau_1, ..., tau_{n-2}).
        interior_tau_np = full_tau_np[1:-1]
        full_tau = torch.tensor(full_tau_np, dtype=torch.float32, device=device)
        interior_tau = torch.tensor(
            interior_tau_np, dtype=torch.float32, device=device,
        )
        weights = torch.tensor(full_w_np, dtype=torch.float32, device=device)
        cvar_w = (
            _compute_cvar_weights(full_tau, weights, risk_fraction)
            if risk_type == "cvar"
            else None
        )
        return QuantileScheme(
            tau=interior_tau,
            mean_weights=weights,
            cvar_weights=cvar_w,
            num_predicted=n - 2,
            fixed_lo=float(q_min),
            fixed_hi=float(q_max),
            mode="trapezoidal",
        )

    raise ValueError(f"Unknown quantile_mode: {mode!r}")

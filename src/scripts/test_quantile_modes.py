"""Smoke + correctness tests for quantile_modes.

Run as a script (no pytest harness):
    ./venv-RL/bin/python3 src/scripts/test_quantile_modes.py
"""

import math
import os
import sys

import torch

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
)

from src.distrl.agents.distributional.quantile_modes import build_scheme  # noqa: E402


def _check(name: str, ok: bool, info: str = "") -> None:
    status = "  OK " if ok else " FAIL"
    print(f"  [{status}] {name}{(' — ' + info) if info else ''}")
    assert ok, name


def test_shapes_and_weights() -> None:
    print("test_shapes_and_weights")
    device = torch.device("cpu")
    N = 50

    s = build_scheme("midpoint", N, device)
    _check("midpoint num_predicted=N", s.num_predicted == N)
    _check("midpoint weights sum=1", torch.isclose(s.mean_weights.sum(), torch.tensor(1.0)))

    s = build_scheme("gauss_legendre", N, device)
    _check("gauss_legendre num_predicted=N", s.num_predicted == N)
    _check("gauss_legendre weights sum=1", torch.isclose(s.mean_weights.sum(), torch.tensor(1.0)))
    _check("gauss_legendre tau monotonic", bool((s.tau[1:] > s.tau[:-1]).all()))

    s = build_scheme("trapezoidal", N, device, q_min=0.0, q_max=10.0)
    _check("trapezoidal num_predicted=N-2", s.num_predicted == N - 2)
    _check("trapezoidal weights sum=1", torch.isclose(s.mean_weights.sum(), torch.tensor(1.0)))
    _check("trapezoidal has fixed endpoints", s.has_fixed_endpoints)

    s = build_scheme("midpoint", N, device, risk_type="cvar", risk_fraction=0.1, truncate_upper=True)
    _check("truncate_upper k = ceil(N*rf)", s.num_predicted == math.ceil(N * 0.1))
    _check("truncate_upper tau in [0, rf]", float(s.tau.max()) <= 0.1)


def test_integration_correctness() -> None:
    print("test_integration_correctness")
    device = torch.device("cpu")
    # Linear: int_0^1 tau dtau = 0.5  (exact for all sane quadratures, N>=2)
    for mode in ("midpoint", "gauss_legendre"):
        s = build_scheme(mode, 50, device)
        est = float(s.expectation(s.tau.unsqueeze(0)))
        _check(f"{mode} integrates tau", abs(est - 0.5) < 1e-5, f"got {est:.6f}")

    s = build_scheme("trapezoidal", 50, device, q_min=0.0, q_max=1.0)
    est = float(s.expectation(s.tau.unsqueeze(0)))
    _check("trapezoidal integrates tau", abs(est - 0.5) < 1e-5, f"got {est:.6f}")

    # Quadratic: int_0^1 tau^2 dtau = 1/3
    # GL with N>=2 is exact for tau^2 mathematically; the residual error is
    # the float32 precision of the (tau, weight) tensors (~3e-8). Midpoint
    # is O(1/N^2) ~ 1/12N^2 = 3.3e-5 for N=50.
    for mode, tol in (("midpoint", 5e-4), ("gauss_legendre", 1e-6)):
        s = build_scheme(mode, 50, device)
        est = float(s.expectation((s.tau ** 2).unsqueeze(0)))
        _check(f"{mode} integrates tau^2", abs(est - 1.0/3) < tol, f"got {est:.7f}")


def test_cvar_weights() -> None:
    print("test_cvar_weights")
    device = torch.device("cpu")
    for mode in ("midpoint", "gauss_legendre", "trapezoidal"):
        s = build_scheme(
            mode, 50, device,
            q_min=0.0, q_max=1.0,
            risk_type="cvar", risk_fraction=0.1,
        )
        assert s.cvar_weights is not None
        _check(f"{mode} cvar_weights sum=1",
               torch.isclose(s.cvar_weights.sum(), torch.tensor(1.0)))


def test_assemble_full_trapezoidal() -> None:
    print("test_assemble_full_trapezoidal")
    device = torch.device("cpu")
    s = build_scheme("trapezoidal", 10, device, q_min=0.0, q_max=100.0)
    # 8 interior predictions in [10, 90]
    preds = torch.linspace(10.0, 90.0, 8).unsqueeze(0).unsqueeze(0)
    full = s.assemble_full(preds).squeeze().tolist()
    _check("first = q_min", abs(full[0] - 0.0) < 1e-6)
    _check("last  = q_max", abs(full[-1] - 100.0) < 1e-6)
    interior = torch.linspace(10.0, 90.0, 8).tolist()
    _check(
        "middle preserved",
        all(abs(a - b) < 1e-5 for a, b in zip(full[1:-1], interior)),
    )


def main() -> None:
    test_shapes_and_weights()
    test_integration_correctness()
    test_cvar_weights()
    test_assemble_full_trapezoidal()
    print("\nAll quantile_modes smoke tests passed.")


if __name__ == "__main__":
    main()

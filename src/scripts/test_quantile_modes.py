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
    one = torch.tensor(1.0)

    s = build_scheme("midpoint", N, device)
    _check("midpoint num_predicted=N", s.num_predicted == N)
    _check("midpoint mean_w sum=1", torch.isclose(s.mean_weights.sum(), one))
    _check("midpoint pred_w sum=1", torch.isclose(s.predictor_weights.sum(), one))
    _check("midpoint pred_w == mean_w", torch.allclose(s.predictor_weights, s.mean_weights))

    s = build_scheme("gauss_legendre", N, device)
    _check("gauss_legendre num_predicted=N", s.num_predicted == N)
    _check("gauss_legendre mean_w sum=1", torch.isclose(s.mean_weights.sum(), one))
    _check("gauss_legendre pred_w sum=1", torch.isclose(s.predictor_weights.sum(), one))
    _check("gauss_legendre pred_w == mean_w", torch.allclose(s.predictor_weights, s.mean_weights))
    _check("gauss_legendre tau monotonic", bool((s.tau[1:] > s.tau[:-1]).all()))
    _check("gauss_legendre weights NOT uniform",
           not torch.allclose(s.mean_weights, torch.full_like(s.mean_weights, 1.0/N)))

    s = build_scheme("trapezoidal", N, device, q_min=0.0, q_max=10.0)
    _check("trapezoidal num_predicted=N-2", s.num_predicted == N - 2)
    _check("trapezoidal mean_w sum=1", torch.isclose(s.mean_weights.sum(), one))
    _check("trapezoidal pred_w sum=1", torch.isclose(s.predictor_weights.sum(), one))
    _check("trapezoidal pred_w uniform 1/(N-2)",
           torch.allclose(s.predictor_weights, torch.full((N-2,), 1.0/(N-2))))
    _check("trapezoidal has fixed endpoints", s.has_fixed_endpoints)

    # Simpson needs odd N. Use N=51 for parity-ish with N=50 elsewhere.
    Ns = 51
    s = build_scheme("simpson", Ns, device, q_min=0.0, q_max=10.0)
    _check("simpson num_predicted=N-2", s.num_predicted == Ns - 2)
    _check("simpson mean_w sum=1", torch.isclose(s.mean_weights.sum(), one))
    _check("simpson pred_w sum=1", torch.isclose(s.predictor_weights.sum(), one))
    _check("simpson has fixed endpoints", s.has_fixed_endpoints)
    # Endpoint atoms should have the smallest mean_weight (1 / (3(N-1))).
    _check(
        "simpson endpoint weight = 1/(3(N-1))",
        torch.isclose(s.mean_weights[0], torch.tensor(1.0 / (3 * (Ns - 1)))),
    )
    # Odd-indexed interior should have weight 4 / (3(N-1)).
    _check(
        "simpson odd-interior weight = 4/(3(N-1))",
        torch.isclose(s.mean_weights[1], torch.tensor(4.0 / (3 * (Ns - 1)))),
    )
    # Even-interior should have weight 2 / (3(N-1)).
    _check(
        "simpson even-interior weight = 2/(3(N-1))",
        torch.isclose(s.mean_weights[2], torch.tensor(2.0 / (3 * (Ns - 1)))),
    )
    # Even N must raise.
    raised = False
    try:
        build_scheme("simpson", 50, device)
    except ValueError:
        raised = True
    _check("simpson rejects even N", raised)

    s = build_scheme("midpoint", N, device, risk_type="cvar", risk_fraction=0.1, truncate_upper=True)
    _check("truncate_upper k = ceil(N*rf)", s.num_predicted == math.ceil(N * 0.1))
    _check("truncate_upper pred_w sum=1", torch.isclose(s.predictor_weights.sum(), one))
    _check("truncate_upper tau in [0, rf]", float(s.tau.max()) <= 0.1)


def test_loss_backward_compat() -> None:
    """The new weighted loss must produce identical numbers for midpoint
    (uniform weights) and differ for gauss_legendre (non-uniform)."""
    print("test_loss_backward_compat")
    torch.manual_seed(123)
    B, N = 4, 20
    pinball_loss = torch.rand(B, N, N)

    # Old plain .mean() over (B, predictor, target).
    old = pinball_loss.mean()

    # New: weighted sum over predictor and target, then batch mean.
    device = torch.device("cpu")
    s = build_scheme("midpoint", N, device)
    per_pred = (pinball_loss * s.mean_weights.view(1, 1, -1)).sum(dim=2)
    per_sample = (per_pred * s.predictor_weights.view(1, -1)).sum(dim=1)
    new_midpoint = per_sample.mean()
    _check("midpoint .mean() == weighted sum", torch.allclose(old, new_midpoint),
           f"old={float(old):.6f}, new={float(new_midpoint):.6f}")

    # Gauss-Legendre should DIFFER from the uniform .mean().
    s_gl = build_scheme("gauss_legendre", N, device)
    per_pred = (pinball_loss * s_gl.mean_weights.view(1, 1, -1)).sum(dim=2)
    per_sample = (per_pred * s_gl.predictor_weights.view(1, -1)).sum(dim=1)
    new_gl = per_sample.mean()
    _check("gauss_legendre loss differs from uniform .mean()",
           not torch.allclose(old, new_gl, atol=1e-4),
           f"old={float(old):.6f}, gl={float(new_gl):.6f}")


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

    # Simpson: exact for cubics, so trivially for tau (with the fixed
    # endpoints 0 and 1 actually present in the assembled distribution).
    s = build_scheme("simpson", 51, device, q_min=0.0, q_max=1.0)
    est = float(s.expectation(s.tau.unsqueeze(0)))
    _check("simpson integrates tau", abs(est - 0.5) < 1e-5, f"got {est:.6f}")

    # Quadratic: int_0^1 tau^2 dtau = 1/3
    # GL with N>=2 is exact for tau^2 mathematically; the residual error is
    # the float32 precision of the (tau, weight) tensors (~3e-8). Midpoint
    # is O(1/N^2) ~ 1/12N^2 = 3.3e-5 for N=50.
    for mode, tol in (("midpoint", 5e-4), ("gauss_legendre", 1e-6)):
        s = build_scheme(mode, 50, device)
        est = float(s.expectation((s.tau ** 2).unsqueeze(0)))
        _check(f"{mode} integrates tau^2", abs(est - 1.0/3) < tol, f"got {est:.7f}")

    # Simpson is mathematically exact for tau^2 and tau^3; residual is just
    # float32 quantisation of the tau/weight tensors. Note: we pass tau^k of
    # the INTERIOR nodes and the fixed endpoints 0 and 1 are auto-prepended
    # via assemble_full -> 0^k and 1^k contribute correctly.
    s = build_scheme("simpson", 51, device, q_min=0.0, q_max=1.0)
    est_sq = float(s.expectation((s.tau ** 2).unsqueeze(0)))
    _check(
        "simpson integrates tau^2", abs(est_sq - 1.0/3) < 1e-5,
        f"got {est_sq:.7f}",
    )
    est_cu = float(s.expectation((s.tau ** 3).unsqueeze(0)))
    _check(
        "simpson integrates tau^3", abs(est_cu - 0.25) < 1e-5,
        f"got {est_cu:.7f}",
    )


def test_cvar_weights() -> None:
    print("test_cvar_weights")
    device = torch.device("cpu")
    for mode, n in (
        ("midpoint", 50),
        ("gauss_legendre", 50),
        ("trapezoidal", 50),
        ("simpson", 51),
    ):
        s = build_scheme(
            mode, n, device,
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
    test_loss_backward_compat()
    print("\nAll quantile_modes smoke tests passed.")


if __name__ == "__main__":
    main()

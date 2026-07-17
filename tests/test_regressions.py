#!/usr/bin/env python3
"""
Regression tests locking in the audit fixes:
- Merton_process excess-kurtosis formula (3*sigJ**4 term, not 3*sigJ**3)
- scipy>=1.12 compatibility (scp.mean removed from scipy top level)
- LSM 'order' parameter actually used in the polynomial regression
plus cheap model invariants (put-call parity, MC vs closed form,
Merton closed formula vs Fourier inversion).

Run from the repo root with:  PYTHONPATH=src python -m pytest tests/ -x -q
"""

import numpy as np
import pytest

from FMNM.Processes import Diffusion_process, Merton_process
from FMNM.Parameters import Option_param
from FMNM.BS_pricer import BS_pricer
from FMNM.Merton_pricer import Merton_pricer

# importing these locks in the scipy-compat fix (scp.mean -> np.mean)
import FMNM.VG_pricer  # noqa: F401
import FMNM.NIG_pricer  # noqa: F401
import FMNM.Heston_pricer  # noqa: F401
import FMNM.Solvers  # noqa: F401  # locks in the scipy.linalg.misc -> scipy.linalg fix


def test_merton_kurtosis():
    """The 4th cumulant of compound-Poisson-normal jumps is
    lam * (muJ**4 + 6 muJ**2 sigJ**2 + 3 sigJ**4); the old code had 3*sigJ**3."""
    lam, muJ, sigJ, sig = 0.8, 0.1, 0.5, 0.2
    proc = Merton_process(r=0.1, sig=sig, lam=lam, muJ=muJ, sigJ=sigJ)

    var = sig**2 + lam * sigJ**2 + lam * muJ**2  # 2nd cumulant
    c4 = lam * (muJ**4 + 6 * muJ**2 * sigJ**2 + 3 * sigJ**4)  # 4th cumulant
    expected_kurt = c4 / var**2  # excess kurtosis

    assert proc.kurt == pytest.approx(expected_kurt, abs=1e-12)
    assert proc.kurt == pytest.approx(2.635, abs=0.01)  # sanity, hand-derived


def test_scipy_compat():
    """All pricer modules import, and BS Monte Carlo (which used scp.mean)
    agrees with the closed formula within 3 standard errors."""
    np.random.seed(seed=42)
    opt = Option_param(S0=100, K=100, T=1, payoff="call", exercise="European")
    diff = Diffusion_process(r=0.1, sig=0.2)
    pricer = BS_pricer(opt, diff)

    closed = pricer.closed_formula()
    mc, std_err = pricer.MC(N=10000, Err=True)
    mc = np.ravel(mc)[0].item()  # MC returns a shape-(1,) array
    std_err = np.ravel(std_err)[0].item()

    assert abs(mc - closed) < 3 * std_err


def test_put_call_parity():
    """BS closed-form: call - put == S0 - K*exp(-r T) within 1e-8."""
    S0, K, T, r, sigma = 100.0, 110.0, 2.0, 0.05, 0.25
    call = BS_pricer.BlackScholes("call", S0, K, T, r, sigma)
    put = BS_pricer.BlackScholes("put", S0, K, T, r, sigma)
    assert call - put == pytest.approx(S0 - K * np.exp(-r * T), abs=1e-8)


def test_lsm_order_sensitivity():
    """The 'order' argument must be live: with identical seeds, order=2 and
    order=4 regressions give different American put prices."""
    opt = Option_param(S0=100, K=110, T=1, payoff="put", exercise="American")
    diff = Diffusion_process(r=0.1, sig=0.3)
    pricer = BS_pricer(opt, diff)

    np.random.seed(seed=7)
    p2 = pricer.LSM(N=200, paths=2000, order=2)
    np.random.seed(seed=7)
    p2_again = pricer.LSM(N=200, paths=2000, order=2)
    np.random.seed(seed=7)
    p4 = pricer.LSM(N=200, paths=2000, order=4)

    assert p2 == pytest.approx(p2_again)  # same seed, same order -> identical
    assert p2 != p4  # the order parameter changes the regression


def test_merton_closed_vs_fourier():
    """Merton closed formula (BS series) vs Fourier inversion of the cf."""
    opt = Option_param(S0=100, K=100, T=1, payoff="call", exercise="European")
    proc = Merton_process(r=0.05, sig=0.2, lam=0.8, muJ=0.0, sigJ=0.3)
    pricer = Merton_pricer(opt, proc)

    closed = pricer.closed_formula()
    fourier = pricer.Fourier_inversion()

    assert fourier == pytest.approx(closed, rel=1e-4)

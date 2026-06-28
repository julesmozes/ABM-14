"""Pytest-discoverable tests for the fishing ABM.

Run with: pytest -q
"""
import numpy as np

from model import (
    K_MAX,
    LAMBDA_MAX,
    LAMBDA_MIN,
    chebyshev,
    mutate_lambda,
    scrounge_probability,
    FishingModel,
)


def test_chebyshev_non_torus():
    assert chebyshev((0, 0), (3, 4), 10, 10, torus=False) == 4


def test_chebyshev_torus():
    assert chebyshev((0, 0), (9, 0), 10, 10, torus=True) == 1


def test_mutate_lambda_clipped():
    rng = np.random.default_rng(123)
    out = mutate_lambda(4.9, rng, sigma=0.5)
    assert LAMBDA_MIN <= out <= LAMBDA_MAX


def test_scrounge_probability_monotone_in_beta():
    p_low = scrounge_probability(1.0, 2.0, beta=0.5)
    p_high = scrounge_probability(1.0, 2.0, beta=5.0)
    assert p_high > p_low


def test_scrounge_probability_prefers_scrounge_when_advantage_positive():
    assert scrounge_probability(0.0, 1.0, beta=2.0) > 0.5


def test_fishingmodel_init_properties():
    m = FishingModel(width=10, height=8, n_agents=5, v=2, q=0.2, rng=123)
    assert m.density.data.shape == (10, 8)
    assert m.capacity.data.shape == (10, 8)
    assert float(m.capacity.data.min()) >= 0.1 * K_MAX - 1e-12
    assert float(m.capacity.data.max()) <= K_MAX + 1e-12
    assert float(m.density.data.min()) >= 0.0
    assert float(m.density.data.max()) <= float(m.capacity.data.max()) + 1e-12
    assert len(m.agents) == 5
    assert m.K == K_MAX
    for agent in m.agents:
        assert LAMBDA_MIN <= agent.loss_aversion <= LAMBDA_MAX


def test_model_step_runs():
    m = FishingModel(width=5, height=5, n_agents=3, v=2, q=0.3, rng=123)
    before = m.density.data.copy()
    m.step()
    assert len(m.agents) >= 0
    assert m.density.data.shape == before.shape

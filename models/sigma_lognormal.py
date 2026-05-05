# models/sigma_lognormal.py
import numpy as np


def single_lognormal(t, D, t0, mu, sigma):
    """One lognormal velocity component."""
    dt    = t - t0
    v     = np.zeros_like(t, dtype=np.float64)
    valid = dt > 1e-9
    v[valid] = (D / (sigma * np.sqrt(2 * np.pi) * dt[valid])) * \
               np.exp(-0.5 * ((np.log(dt[valid]) - mu) / sigma) ** 2)
    return v


def sigma_lognormal_velocity(t, params):
    """
    Sum of K lognormal components.

    Args:
        t:      (N,) time array
        params: (K, 4) [[D, t0, mu, sigma], ...]
    Returns:
        v: (N,) velocity profile
    """
    v = np.zeros(len(t))
    for D, t0, mu, sigma in params:
        v += single_lognormal(t, D, t0, mu, sigma)
    return v


def extract_params(points, K=3):
    """
    Naïve initialization of K Sigma-Lognormal components from a stroke.

    Args:
        points: (N, 4) [x, y, time, pen_up]
        K:      number of components
    Returns:
        params: (K, 4) [D, t0, mu, sigma]
    """
    t      = points[:, 2]
    t_norm = (t - t.min()) / (t.max() - t.min() + 1e-8)

    dx    = np.gradient(points[:, 0])
    dy    = np.gradient(points[:, 1])
    dtn   = np.gradient(t_norm) + 1e-8
    speed = np.sqrt(dx**2 + dy**2) / dtn
    speed = speed / (speed.max() + 1e-8)

    params = []
    for k in range(K):
        t_center = (k + 0.5) / K
        D    = speed.max() / K
        t0   = max(0.0, t_center - 0.1)
        mu   = np.log(max(t_center - t0, 1e-4))
        sigma = 0.4
        params.append([D, t0, mu, sigma])

    return np.array(params, dtype=np.float32)


def lognormal_features(points, K=3):
    """Flat feature vector of Sigma-Lognormal params, shape (K*4,)."""
    return extract_params(points, K=K).flatten()

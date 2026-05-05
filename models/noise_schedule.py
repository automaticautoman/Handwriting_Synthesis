# models/noise_schedule.py
import torch, math


def cosine_schedule(T=1000, s=0.008):
    """
    Cosine noise schedule (Nichol & Dhariwal, 2021).

    Returns:
        betas:     (T,) per-step variance
        alpha_bar: (T,) cumulative signal retention
    """
    steps     = torch.arange(T + 1, dtype=torch.float64)
    f         = torch.cos(((steps / T) + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = f / f[0]
    betas     = 1 - alpha_bar[1:] / alpha_bar[:-1]
    betas     = torch.clamp(betas, 0, 0.999).float()
    return betas, alpha_bar[:-1].float()


def q_sample(x0, t, alpha_bar):
    """
    Forward process: x_t = sqrt(ā_t)·x0 + sqrt(1−ā_t)·ε

    Args:
        x0:        (B, L, C) clean sequence
        t:         (B,)      timestep indices
        alpha_bar: (T,)      precomputed cumulative alphas

    Returns:
        x_t:  noisy sequence
        eps:  the noise that was added (training target)
    """
    ab  = alpha_bar[t].view(-1, 1, 1)
    eps = torch.randn_like(x0)
    x_t = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
    return x_t, eps

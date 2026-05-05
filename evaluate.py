# evaluate.py
import torch
import numpy as np
from tqdm import tqdm
from scipy.signal import find_peaks
from models.noise_schedule import cosine_schedule
from models.text_encoder import tokenize
from config import CONFIG


def make_char_positions(text, seq_len, device):
    """
    Generate character position signal for inference.

    For a sequence of length seq_len, each point is assigned
    a normalized position [0, 1] based on which character it
    corresponds to in the text string.

    Args:
        text:    target text string
        seq_len: number of stroke points to generate
        device:  torch device

    Returns:
        (1, seq_len) float tensor
    """
    n_chars   = max(len(text), 1)
    positions = np.zeros(seq_len, dtype=np.float32)
    for i in range(seq_len):
        char_idx      = int((i / seq_len) * n_chars)
        char_idx      = min(char_idx, n_chars - 1)
        positions[i]  = char_idx / n_chars
    return torch.from_numpy(positions).unsqueeze(0).to(device)  # (1, L)


@torch.no_grad()
def sample(score_net, text_enc, writer_enc,
           phi, seq_len, betas, alpha_bar, device,
           text="hello world", writer_id=0, n_steps=50):
    """
    DDIM sampling — stable with far fewer steps than DDPM.

    Uses 50 denoising steps instead of 1000, which prevents
    error accumulation and avoids the inf/nan explosion we
    saw with the raw DDPM sampler.

    Args:
        score_net:  trained ScoreNet
        text_enc:   trained TextEncoder
        writer_enc: trained WriterEmbedding
        phi:        (1, 5) or (5,) biophysical conditioning
        seq_len:    number of stroke points to generate
        betas:      from cosine_schedule()
        alpha_bar:  from cosine_schedule()
        text:       string to write
        writer_id:  integer writer ID
        n_steps:    number of DDIM denoising steps (default 50)

    Returns:
        (seq_len, 6) numpy array — generated stroke sequence
    """
    score_net.eval(); text_enc.eval(); writer_enc.eval()

    T         = len(betas)
    step_size = T // n_steps
    timesteps = list(reversed(range(0, T, step_size)))[:n_steps]

    betas     = betas.to(device)
    alpha_bar = alpha_bar.to(device)

    # Encode text
    ids, mask = tokenize(text, max_len=CONFIG['text_max_len'])
    ids       = ids.unsqueeze(0).to(device)
    mask      = mask.unsqueeze(0).to(device)

    # Safety: ensure at least one text token is unmasked
    if not mask.any():
        mask[0, 0] = True

    text_emb   = text_enc(ids, mask)
    writer_emb = writer_enc(torch.tensor([writer_id], device=device))

    phi_batch = phi.to(device)
    if phi_batch.dim() == 1:
        phi_batch = phi_batch.unsqueeze(0)

    seq_mask    = torch.ones(1, seq_len, dtype=torch.bool, device=device)
    char_pos    = make_char_positions(text, seq_len, device)

    # Start from pure Gaussian noise
    x = torch.randn(1, seq_len, 6, device=device)

    for i, t_idx in enumerate(tqdm(timesteps, desc="Sampling", leave=False)):
        t      = torch.tensor([t_idx], device=device)
        ab_t   = alpha_bar[t_idx]
        ab_prev = (alpha_bar[timesteps[i + 1]]
                   if i + 1 < len(timesteps)
                   else torch.tensor(1.0, device=device))

        eps_pred = score_net(x, t, phi_batch, writer_emb, text_emb,
                             text_mask=mask, seq_mask=seq_mask,
                             char_positions=char_pos)

        # DDIM update — deterministic, no noise accumulation
        x0_pred = (x - (1 - ab_t).sqrt() * eps_pred) / ab_t.sqrt()
        x0_pred = torch.clamp(x0_pred, -5.0, 5.0)
        x       = ab_prev.sqrt() * x0_pred + (1 - ab_prev).sqrt() * eps_pred

    return x.squeeze(0).cpu().numpy()  # (seq_len, 6)


# ─── Evaluation metrics ───────────────────────────────────────────────────────

def check_velocity_shape(seq):
    """
    Fraction of strokes with a bell-shaped velocity profile.
    Target: > 0.7
    """
    velocity   = seq[:, 2]
    pen_up     = seq[:, 4]
    up_indices = np.where(pen_up > 0.5)[0]
    if len(up_indices) == 0:
        return 0.0
    start, valid = 0, 0
    for end in up_indices:
        stroke_v = velocity[start:end + 1]
        if len(stroke_v) > 5:
            peaks, _ = find_peaks(stroke_v, prominence=0.03)
            if len(peaks) > 0:
                valid += 1
        start = end + 1
    return valid / len(up_indices)


def check_pressure_velocity_anticorrelation(seq):
    """
    Pearson r between velocity and pressure.
    Target: -0.4 to -0.8 (negative = physically correct)
    """
    v = seq[:, 2]
    p = seq[:, 3]
    if v.std() < 1e-6 or p.std() < 1e-6:
        return 0.0
    return float(np.corrcoef(v, p)[0, 1])


def check_fatigue_trend(seq):
    """
    Velocity slope over the sequence.
    Target: < 0 (fatigued writers slow down)
    """
    v = seq[:, 2]
    n = len(v)
    if n < 10:
        return 0.0
    return float(np.polyfit(np.arange(n), v, 1)[0])


def full_eval(score_net, text_enc, writer_enc, val_dl,
              betas, alpha_bar, device, n_samples=50):
    """Run all three biomechanical metrics on n_samples generated sequences."""
    results = {'velocity_shape': [], 'pv_corr': [], 'fatigue_slope': []}

    for batch in val_dl:
        phi       = batch['phi'][0]
        writer_id = int(batch['writer_id'][0].item())
        text      = "hello world"

        gen = sample(score_net, text_enc, writer_enc,
                     phi, CONFIG['max_seq_len'], betas, alpha_bar,
                     device, text=text, writer_id=writer_id)

        results['velocity_shape'].append(check_velocity_shape(gen))
        results['pv_corr'].append(check_pressure_velocity_anticorrelation(gen))
        results['fatigue_slope'].append(check_fatigue_trend(gen))

        if len(results['velocity_shape']) >= n_samples:
            break

    print("=== Biomechanical Evaluation ===")
    print(f"Velocity bell shape: {np.mean(results['velocity_shape']):.3f}  (target: > 0.7)")
    print(f"Pressure-velocity r: {np.mean(results['pv_corr']):.3f}          (target: -0.4 to -0.8)")
    print(f"Fatigue slope:       {np.mean(results['fatigue_slope']):.6f}    (target: < 0)")

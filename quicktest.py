# quicktest.py
import sys
sys.path.insert(0, '.')

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.diffusion import ScoreNet
from models.text_encoder import TextEncoder, WriterEmbedding, tokenize
from models.noise_schedule import cosine_schedule
from config import CONFIG
from evaluate import sample
from train import get_device


def render_handwriting(seq, title="Generated", save_path=None):
    """
    Render a stroke sequence as a handwriting image.
    Pen-up events create breaks between strokes.
    """
    x      = seq[:, 0]
    y      = seq[:, 1]
    pen_up = seq[:, 4]

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.set_facecolor('white')
    ax.set_title(title, fontsize=13)
    ax.invert_yaxis()
    ax.axis('off')

    start = 0
    for i in range(len(seq)):
        if pen_up[i] > 0.5 or i == len(seq) - 1:
            end      = i + 1
            stroke_x = x[start:end]
            stroke_y = y[start:end]
            if len(stroke_x) > 1:
                ax.plot(stroke_x, stroke_y,
                        color='black', linewidth=1.5,
                        solid_capstyle='round',
                        solid_joinstyle='round')
            start = i + 1

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def visualize(seq, title="Generated handwriting", save_path=None):
    """
    4-panel diagnostic plot showing all 6 channels.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=14)

    x, y  = seq[:, 0], seq[:, 1]
    vel   = seq[:, 2]
    pres  = seq[:, 3]
    pen_u = seq[:, 4]
    fat   = seq[:, 5]
    N     = len(seq)

    # Trajectory coloured by velocity
    ax = axes[0, 0]
    up = np.where(pen_u > 0.5)[0]
    start = 0
    for end in np.append(up, N - 1):
        seg = slice(start, int(end) + 1)
        sc  = ax.scatter(x[seg], y[seg], c=vel[seg],
                         cmap='plasma', s=3, vmin=vel.min(), vmax=vel.max())
        start = int(end) + 1
    ax.set_title("Trajectory (colour = velocity)")
    ax.set_aspect('equal')
    ax.invert_yaxis()
    plt.colorbar(sc, ax=ax, label='speed')

    # Velocity
    axes[0, 1].plot(vel, color='steelblue', linewidth=1.2)
    axes[0, 1].set_title("Velocity profile")
    axes[0, 1].set_xlabel("point index")
    axes[0, 1].set_ylabel("normalized speed")

    # Pressure
    axes[1, 0].plot(pres, color='darkorange', linewidth=1.2)
    axes[1, 0].set_title("Pen pressure")
    axes[1, 0].set_xlabel("point index")
    axes[1, 0].set_ylabel("pressure (0-1)")

    # Fatigue
    axes[1, 1].plot(fat, color='mediumseagreen', linewidth=1.2)
    axes[1, 1].set_title("Fatigue level")
    axes[1, 1].set_xlabel("point index")
    axes[1, 1].set_ylabel("phi_f (0-1)")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


if __name__ == '__main__':
    device    = get_device()
    betas, ab = cosine_schedule(T=CONFIG['T'])

    # ── Load models ───────────────────────────────────────────────────
    ckpt = torch.load('best_model.pt', map_location=device)

    score_net = ScoreNet(
        seq_dim          = CONFIG['seq_dim'],
        d_model          = CONFIG['d_model'],
        n_layers         = CONFIG['n_layers'],
        n_heads          = CONFIG['n_heads'],
        phi_dim          = CONFIG['phi_dim'],
        writer_embed_dim = CONFIG['writer_embed_dim'],
        text_dim         = CONFIG['text_embed_dim'],
    ).to(device)

    text_enc = TextEncoder(
        vocab_size = CONFIG['vocab_size'],
        d_model    = CONFIG['text_d_model'],
        n_heads    = CONFIG['text_n_heads'],
        n_layers   = CONFIG['text_n_layers'],
        max_len    = CONFIG['text_max_len'],
        out_dim    = CONFIG['text_embed_dim'],
    ).to(device)

    writer_enc = WriterEmbedding(
        n_writers  = CONFIG['n_writers'],
        embed_dim  = CONFIG['writer_embed_dim'],
    ).to(device)
    score_net.load_state_dict(ckpt['score_net'])

    text_enc.load_state_dict(ckpt['text_enc'])
    writer_enc.load_state_dict(ckpt['writer_enc'])

    # ── Define biophysical states ─────────────────────────────────────
    # All values normalized to [0, 1]:
    # [fatigue_rate/0.01, tremor_freq/12, tremor_amp, base_pressure, writer_id/500]

    phi_healthy = torch.tensor([[0.2, 0.667, 0.0, 0.5, 0.1]])   # healthy writer
    phi_elderly = torch.tensor([[0.8, 0.417, 0.4, 0.4, 0.2]])   # tremor + fast fatigue
    phi_tired   = torch.tensor([[0.9, 0.667, 0.0, 0.4, 0.3]])   # fatigued, no tremor

    # ── Generate ──────────────────────────────────────────────────────
    text = "hello world"
    print(f'\nGenerating handwriting for: "{text}"')
    print("(Note: output looks random until GPU training completes)\n")

    gen_healthy = sample(score_net, text_enc, writer_enc,
                         phi_healthy, CONFIG['max_seq_len'],
                         betas, ab, device,
                         text=text, writer_id=0)

    gen_elderly = sample(score_net, text_enc, writer_enc,
                         phi_elderly, CONFIG['max_seq_len'],
                         betas, ab, device,
                         text=text, writer_id=1)

    gen_tired = sample(score_net, text_enc, writer_enc,
                       phi_tired, CONFIG['max_seq_len'],
                       betas, ab, device,
                       text=text, writer_id=2)

    # ── Print stats ───────────────────────────────────────────────────
    for name, gen in [("Healthy", gen_healthy),
                      ("Elderly", gen_elderly),
                      ("Tired",   gen_tired)]:
        print(f"{name} writer:")
        print(f"  x range:        {gen[:,0].min():.2f} to {gen[:,0].max():.2f}")
        print(f"  velocity range: {gen[:,2].min():.2f} to {gen[:,2].max():.2f}")
        print(f"  pressure range: {gen[:,3].min():.2f} to {gen[:,3].max():.2f}")
        print(f"  any NaN/Inf:    {not np.isfinite(gen).all()}")
        print()

    # ── Visualize ─────────────────────────────────────────────────────
    render_handwriting(gen_healthy, title=f'Healthy writer — "{text}"',
                       save_path="healthy_handwriting.png")

    render_handwriting(gen_elderly, title=f'Elderly writer (tremor) — "{text}"',
                       save_path="elderly_handwriting.png")

    render_handwriting(gen_tired, title=f'Tired writer — "{text}"',
                       save_path="tired_handwriting.png")

    visualize(gen_healthy, title="Healthy writer — diagnostics",
              save_path="healthy_diagnostics.png")
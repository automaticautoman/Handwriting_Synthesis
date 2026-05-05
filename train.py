# train.py
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb, os, sys

sys.path.insert(0, os.path.dirname(__file__))
from config import CONFIG
from models.diffusion import ScoreNet
from models.text_encoder import TextEncoder, WriterEmbedding
from models.noise_schedule import cosine_schedule, q_sample
from datasets.handwriting import HandwritingDataset


def get_device():
    if torch.cuda.is_available():         return torch.device('cuda')
    if torch.backends.mps.is_available(): return torch.device('mps')
    return torch.device('cpu')


def compute_loss(score_net, text_enc, writer_enc,
                 batch, alpha_bar, device):
    """
    DDPM training loss with full conditioning (text + writer + φ).
    """
    x0         = batch['sequence'].to(device)     # (B, L, 6)
    phi        = batch['phi'].to(device)          # (B, 5)
    seq_mask   = batch['seq_mask'].to(device)     # (B, L) bool
    text_ids   = batch['text_ids'].to(device)     # (B, T_text)
    text_mask  = batch['text_mask'].to(device)    # (B, T_text) bool
    writer_ids = batch['writer_id'].to(device)    # (B,)

    B = x0.shape[0]
    all_masked = ~text_mask.any(dim=-1)  # (B,) True = fully empty text
    if all_masked.any():
      text_mask = text_mask.clone()
      text_mask[all_masked, 0] = True   # unmask at least one token

    # Encode text and writer
    text_emb   = text_enc(text_ids, text_mask)    # (B, T_text, text_dim)
    writer_emb = writer_enc(writer_ids)            # (B, writer_dim)

    # Sample random diffusion timestep
    t = torch.randint(0, CONFIG['T'], (B,), device=device)

    # Forward process: add noise
    alpha_bar_dev = alpha_bar.to(device)
    x_t, eps_true = q_sample(x0, t, alpha_bar_dev)

    # Predict noise
    eps_pred = score_net(
        x_t, t, phi, writer_emb, text_emb,
        text_mask=text_mask,
        seq_mask=seq_mask,
    )

    # MSE only on real (unmasked) positions
    mask_exp = seq_mask.unsqueeze(-1).float()     # (B, L, 1)
    loss = F.mse_loss(eps_pred * mask_exp, eps_true * mask_exp, reduction='sum')
    loss = loss / seq_mask.float().sum()

    return loss


def train():
    device = get_device()
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────
    train_ds = HandwritingDataset(CONFIG['data_root'], CONFIG['max_seq_len'], split='train')
    val_ds   = HandwritingDataset(CONFIG['data_root'], CONFIG['max_seq_len'], split='val')
    train_dl = DataLoader(train_ds, batch_size=CONFIG['batch_size'],
                          shuffle=True, num_workers=0, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=CONFIG['batch_size'],
                          shuffle=False, num_workers=0, pin_memory=True)

    # ── Noise schedule ────────────────────────────────────────────────
    betas, alpha_bar = cosine_schedule(T=CONFIG['T'])

    # ── Models ────────────────────────────────────────────────────────
    score_net = ScoreNet(
        seq_dim        = CONFIG['seq_dim'],
        d_model        = CONFIG['d_model'],
        n_layers       = CONFIG['n_layers'],
        n_heads        = CONFIG['n_heads'],
        phi_dim        = CONFIG['phi_dim'],
        writer_embed_dim = CONFIG['writer_embed_dim'],
        text_dim       = CONFIG['text_embed_dim'],
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

    all_params = (list(score_net.parameters()) +
                  list(text_enc.parameters()) +
                  list(writer_enc.parameters()))

    n_params = sum(p.numel() for p in all_params if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # ── Optimizer ─────────────────────────────────────────────────────
    optimizer = AdamW(all_params, lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'], eta_min=1e-6)

    # ── Logging ───────────────────────────────────────────────────────
    wandb.init(project='handwriting-biophysical', config=CONFIG)

    # ── Training loop ─────────────────────────────────────────────────
    best_val = float('inf')

    for epoch in range(CONFIG['epochs']):
        # Train
        score_net.train(); text_enc.train(); writer_enc.train()
        train_loss = 0.0
        for batch in tqdm(train_dl, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}", leave=False):
            loss = compute_loss(score_net, text_enc, writer_enc,
                                batch, alpha_bar, device)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, CONFIG['grad_clip'])
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dl)

        # Validate
        score_net.eval(); text_enc.eval(); writer_enc.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dl:
                val_loss += compute_loss(score_net, text_enc, writer_enc,
                                         batch, alpha_bar, device).item()
        val_loss /= len(val_dl)

        scheduler.step()
        wandb.log({'train_loss': train_loss, 'val_loss': val_loss, 'epoch': epoch + 1})
        print(f"Epoch {epoch+1:3d}  train={train_loss:.4f}  val={val_loss:.4f}")

        # Checkpoint
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                'epoch':      epoch,
                'score_net':  score_net.state_dict(),
                'text_enc':   text_enc.state_dict(),
                'writer_enc': writer_enc.state_dict(),
                'optimizer':  optimizer.state_dict(),
                'config':     CONFIG,
            }, 'best_model.pt')
            print(f"  ✓ saved (val={best_val:.4f})")

    wandb.finish()
    print(f"\nTraining complete. Best val loss: {best_val:.4f}")


if __name__ == '__main__':
    train()

# models/diffusion.py
import torch, torch.nn as nn, math


def timestep_embedding(t, d_model):
    half  = d_model // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
    )
    args  = t[:, None].float() * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class AdaLN(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.norm = nn.LayerNorm(d)

    def forward(self, x, scale, shift):
        return self.norm(x) * (1 + scale) + shift


class PhiConditioner(nn.Module):
    def __init__(self, phi_dim, writer_embed_dim, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(phi_dim + writer_embed_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model * 2),
        )

    def forward(self, phi, writer_emb):
        cond         = torch.cat([phi, writer_emb], dim=-1)
        out          = self.net(cond)
        scale, shift = out.chunk(2, dim=-1)
        return scale.unsqueeze(1), shift.unsqueeze(1)


class CharPositionEmbedding(nn.Module):
    """
    Maps a normalized character index (0=first char, 1=last char)
    per stroke point to a d_model embedding.

    This is the alignment signal — it tells the model WHERE in space
    each stroke point should appear based on which character it belongs to.
    """
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, d_model),
        )

    def forward(self, char_positions):
        # char_positions: (B, L) -> (B, L, d_model)
        return self.proj(char_positions.unsqueeze(-1))


class Block(nn.Module):
    def __init__(self, d_model, n_heads, text_dim, dropout=0.1):
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(d_model, n_heads,
                                                 dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads,
                                                 kdim=text_dim, vdim=text_dim,
                                                 dropout=dropout, batch_first=True)
        self.ff   = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm1 = AdaLN(d_model)
        self.norm2 = AdaLN(d_model)
        self.norm3 = AdaLN(d_model)

    def forward(self, x, scale, shift,
                text_kv, text_key_mask=None, seq_pad_mask=None):
        # Self-attention
        h, _ = self.self_attn(
            self.norm1(x, scale, shift),
            self.norm1(x, scale, shift),
            self.norm1(x, scale, shift),
            key_padding_mask=seq_pad_mask,
        )
        x = x + h

        # Cross-attention — guard against fully-masked text
        if text_key_mask is not None and text_key_mask.all(dim=-1).any():
            all_masked    = text_key_mask.all(dim=-1, keepdim=True)
            text_key_mask = text_key_mask.clone()
            text_key_mask[all_masked.squeeze(-1)] = False

        h, _ = self.cross_attn(
            self.norm2(x, scale, shift),
            text_kv, text_kv,
            key_padding_mask=text_key_mask,
        )
        x = x + h

        x = x + self.ff(self.norm3(x, scale, shift))
        return x


class ScoreNet(nn.Module):
    """
    Full score network with character alignment conditioning.

    Inputs:
        x_t:            (B, L, seq_dim)  noisy stroke sequence
        t:              (B,)             diffusion timestep
        phi:            (B, phi_dim)     biophysical state
        writer_emb:     (B, writer_dim)  writer style
        text_emb:       (B, L_text, text_dim)  character embeddings
        text_mask:      (B, L_text) bool
        seq_mask:       (B, L) bool
        char_positions: (B, L) float in [0,1]  character alignment signal

    Output:
        (B, L, seq_dim) predicted noise
    """

    def __init__(self, seq_dim=6, d_model=256, n_layers=6, n_heads=8,
                 phi_dim=5, writer_embed_dim=64, text_dim=128, dropout=0.1):
        super().__init__()

        self.in_proj      = nn.Linear(seq_dim, d_model)
        self.pos_emb      = nn.Embedding(512, d_model)
        self.char_pos_emb = CharPositionEmbedding(d_model)

        self.t_proj = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )

        self.phi_cond = PhiConditioner(phi_dim, writer_embed_dim, d_model)

        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, text_dim, dropout)
            for _ in range(n_layers)
        ])

        self.out = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, seq_dim),
        )

    def forward(self, x_t, t, phi, writer_emb, text_emb,
                text_mask=None, seq_mask=None, char_positions=None):

        B, L, _ = x_t.shape
        device  = x_t.device

        pos = torch.arange(L, device=device).unsqueeze(0)
        h   = self.in_proj(x_t) + self.pos_emb(pos)

        t_emb = self.t_proj(timestep_embedding(t, h.shape[-1]))
        h     = h + t_emb.unsqueeze(1)

        # Character alignment signal
        if char_positions is not None:
            h = h + self.char_pos_emb(char_positions.to(device).float())

        scale, shift = self.phi_cond(phi, writer_emb)

        text_key_mask = ~text_mask if text_mask is not None else None
        seq_pad_mask  = ~seq_mask  if seq_mask  is not None else None

        for block in self.blocks:
            h = block(h, scale, shift, text_emb,
                      text_key_mask=text_key_mask,
                      seq_pad_mask=seq_pad_mask)

        return self.out(h)

# models/text_encoder.py
import torch
import torch.nn as nn

# ─── Tokenizer ────────────────────────────────────────────────────────────────

VOCAB_OFFSET = 32   # ASCII 32 = space (first printable char)
PAD_ID = 0          # reserved: padding
UNK_ID = 1          # reserved: unknown character


def char_to_id(c):
    code = ord(c)
    if 32 <= code <= 127:
        return code - VOCAB_OFFSET + 2   # +2: leave 0=PAD, 1=UNK
    return UNK_ID


def tokenize(text, max_len=128):
    """
    Convert a string to padded integer tensors.

    Returns:
        ids:  (max_len,) LongTensor
        mask: (max_len,) BoolTensor — True = real token, False = padding
    """
    ids    = [char_to_id(c) for c in text[:max_len]]
    length = len(ids)
    ids   += [PAD_ID] * (max_len - length)
    ids    = torch.tensor(ids, dtype=torch.long)
    mask   = torch.zeros(max_len, dtype=torch.bool)
    mask[:length] = True
    return ids, mask


# ─── Text Encoder ─────────────────────────────────────────────────────────────

class TextEncoder(nn.Module):
    """
    Character-level Transformer encoder.

    Reads a sequence of character IDs and outputs one contextual
    embedding per character. The score network cross-attends to
    these embeddings to know what characters to write.

    Input:  (B, L_text) LongTensor
    Output: (B, L_text, out_dim)
    """

    def __init__(self, vocab_size=98, d_model=128, n_heads=4,
                 n_layers=3, max_len=128, out_dim=128, dropout=0.1):
        super().__init__()

        self.char_embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_embed  = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = n_heads,
            dim_feedforward = d_model * 4,
            dropout         = dropout,
            batch_first     = True,
            norm_first      = True,   # Pre-LN: more stable
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_proj    = nn.Linear(d_model, out_dim) if d_model != out_dim else nn.Identity()
        self.norm        = nn.LayerNorm(out_dim)

    def forward(self, ids, mask=None):
        """
        Args:
            ids:  (B, L) LongTensor — character token IDs
            mask: (B, L) BoolTensor — True = real token

        Returns:
            (B, L, out_dim) contextual character embeddings
        """
        B, L   = ids.shape
        device = ids.device

        pos = torch.arange(L, device=device).unsqueeze(0)
        x   = self.char_embed(ids) + self.pos_embed(pos)

        # TransformerEncoder wants True = IGNORE, so invert our mask
        pad_mask = ~mask if mask is not None else None
        x = self.transformer(x, src_key_padding_mask=pad_mask)

        return self.norm(self.out_proj(x))   # (B, L, out_dim)


# ─── Writer Style Embedding ───────────────────────────────────────────────────

class WriterEmbedding(nn.Module):
    """
    Learned per-writer style vector (lookup table).

    Pass writer_id = -1 to get the average/unknown style.
    """

    def __init__(self, n_writers=500, embed_dim=64):
        super().__init__()
        self.embed      = nn.Embedding(n_writers + 1, embed_dim)
        self.unknown_id = n_writers

    def forward(self, writer_ids):
        """
        Args:
            writer_ids: (B,) LongTensor — use -1 for unknown writer

        Returns:
            (B, embed_dim)
        """
        ids = writer_ids.clone()
        ids[ids < 0] = self.unknown_id
        return self.embed(ids)

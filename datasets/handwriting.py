# datasets/handwriting.py
import torch
from torch.utils.data import Dataset
import numpy as np, os, glob, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.text_encoder import tokenize


class HandwritingDataset(Dataset):
    """
    Loads preprocessed .npy samples.

    Each sample dict contains:
        sequence:       (N, 6)    [x, y, velocity, pressure, pen_up, fatigue]
        biophysical:    (5,)      [fatigue_rate, tremor_freq, tremor_amp,
                                   base_pressure, writer_id_norm]
        text:           str
        writer_id:      int
        char_positions: (max_len,) float32 in [0, 1]  — character alignment
    """

    def __init__(self, data_root, max_len=200, text_max_len=128, split='train'):
        pattern    = os.path.join(data_root, split, '*.npy')
        self.files = sorted(glob.glob(pattern))
        self.max_len      = max_len
        self.text_max_len = text_max_len
        print(f"[{split}] {len(self.files)} samples found")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True).item()

        seq       = data['sequence']
        phi       = data['biophysical']
        text      = str(data.get('text', ''))
        writer_id = int(data.get('writer_id', 0))

        # Character positions — fall back to evenly spaced if not in file
        if 'char_positions' in data:
            char_pos = data['char_positions'].astype(np.float32)
        else:
            # Generate on the fly for old preprocessed files
            N_raw    = len(seq)
            n_chars  = max(len(text), 1)
            char_pos = np.zeros(self.max_len, dtype=np.float32)
            N_use    = min(N_raw, self.max_len)
            for i in range(N_use):
                ci           = int((i / N_use) * n_chars)
                ci           = min(ci, n_chars - 1)
                char_pos[i]  = ci / n_chars

        N = min(len(seq), self.max_len)

        seq_padded      = np.zeros((self.max_len, 6), dtype=np.float32)
        seq_padded[:N]  = seq[:N]

        seq_mask      = np.zeros(self.max_len, dtype=bool)
        seq_mask[:N]  = True

        # Pad / truncate char_positions to max_len
        cp = np.zeros(self.max_len, dtype=np.float32)
        cp[:min(len(char_pos), self.max_len)] = char_pos[:self.max_len]

        text_ids, text_mask = tokenize(text, max_len=self.text_max_len)

        return {
            'sequence':       torch.from_numpy(seq_padded),
            'phi':            torch.from_numpy(phi),
            'seq_mask':       torch.from_numpy(seq_mask),
            'length':         N,
            'text_ids':       text_ids,
            'text_mask':      text_mask,
            'writer_id':      torch.tensor(writer_id, dtype=torch.long),
            'char_positions': torch.from_numpy(cp),
        }

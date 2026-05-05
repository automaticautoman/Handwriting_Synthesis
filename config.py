CONFIG = {
    # Data
    'data_root':    './data/processed',
    'max_seq_len':  200,
    'seq_dim':      6,
    'phi_dim':      5,

    # Text encoder
    'vocab_size':       98,
    'text_d_model':     128,
    'text_n_heads':     4,
    'text_n_layers':    3,
    'text_max_len':     128,
    'text_embed_dim':   128,

    # Writer style
    'n_writers':        500,
    'writer_embed_dim': 64,

    # Diffusion
    'T':             1000,
    'beta_schedule': 'cosine',

    # Score network — full size
    'd_model':   256,
    'n_layers':  6,
    'n_heads':   8,

    # Training
    'batch_size':   32,
    'lr':           3e-6,
    'weight_decay': 1e-4,
    'grad_clip':    1.0,
    'epochs':       150,

    # Biophysical defaults
    'fatigue_rate':       0.002,
    'tremor_freq':        8.0,
    'tremor_amplitude':   0.0,
    'base_pressure':      0.5,
}
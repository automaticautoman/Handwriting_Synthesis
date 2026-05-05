# preprocess.py
import xml.etree.ElementTree as ET
import numpy as np
import os, glob, sys

sys.path.insert(0, os.path.dirname(__file__))
from config import CONFIG
from models.biophysical import FatigueModel, PressureModel


def parse_xml(xml_path):
    """
    Parse one IAM Online XML file.
    Returns:
        points: (N, 4) [x, y, time_ms, pen_up]
        text:   string label
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    all_points = []
    for stroke in root.iter('Stroke'):
        stroke_pts = []
        for pt in stroke.iter('Point'):
            x = float(pt.get('x'))
            y = float(pt.get('y'))
            t = float(pt.get('time'))
            stroke_pts.append([x, y, t])

        for i, p in enumerate(stroke_pts):
            pen_up = 1.0 if i == len(stroke_pts) - 1 else 0.0
            all_points.append([p[0], p[1], p[2], pen_up])

    text = ''
    for tag in ['TextLine', 'text', 'Text']:
        node = root.find(f'.//{tag}')
        if node is not None:
            text = (node.get('text') or node.text or '').strip()
            if text:
                break

    return np.array(all_points, dtype=np.float32), text


def compute_velocity(points):
    """Normalized tangential speed from finite differences."""
    x, y, t = points[:, 0], points[:, 1], points[:, 2]
    dx = np.gradient(x)
    dy = np.gradient(y)
    dt = np.gradient(t) + 1e-8
    speed = np.sqrt(dx**2 + dy**2) / dt
    max_s = speed.max()
    return (speed / max_s).astype(np.float32) if max_s > 0 else speed.astype(np.float32)


def normalize(points):
    """Zero-mean unit-std for x,y; [0,1] for time."""
    pts = points.copy()
    for col in [0, 1]:
        mu  = pts[:, col].mean()
        std = pts[:, col].std() + 1e-8
        pts[:, col] = (pts[:, col] - mu) / std
    t_min, t_max = pts[:, 2].min(), pts[:, 2].max()
    pts[:, 2] = (pts[:, 2] - t_min) / (t_max - t_min + 1e-8)
    return pts


def compute_char_positions(n_points, text, max_len=200):
    """
    For each point in the stroke sequence, assign a normalized
    x-position based on which character it belongs to.

    Strategy: divide the sequence evenly among characters.
    Point i belongs to character floor(i/N * n_chars).
    Position = char_index / n_chars, so values are in [0, 1].

    This is the alignment signal that tells the model WHERE
    each stroke point should appear in space.

    Args:
        n_points: actual number of points in the sequence
        text:     the text label for this sample
        max_len:  padded sequence length

    Returns:
        char_positions: (max_len,) float32 in [0, 1]
    """
    n_chars = max(len(text), 1)
    N       = min(n_points, max_len)

    char_positions = np.zeros(max_len, dtype=np.float32)
    for i in range(N):
        char_idx          = int((i / N) * n_chars)
        char_idx          = min(char_idx, n_chars - 1)
        char_positions[i] = char_idx / n_chars

    return char_positions


def preprocess_dataset(raw_dir, out_dir, val_every=10):
    xml_files  = sorted(glob.glob(os.path.join(raw_dir, '**/*.xml'), recursive=True))
    print(f"Found {len(xml_files)} XML files in {raw_dir}")

    writer_map, writer_counter = {}, 0
    saved = 0

    for i, xml_path in enumerate(xml_files):
        parts      = xml_path.replace('\\', '/').split('/')
        writer_str = parts[-3] if len(parts) >= 3 else 'unknown'
        if writer_str not in writer_map:
            writer_map[writer_str] = writer_counter
            writer_counter += 1
        writer_id = writer_map[writer_str]

        try:
            raw, text = parse_xml(xml_path)
        except Exception as e:
            print(f"  skip {xml_path}: {e}")
            continue

        if len(raw) < 10:
            continue

        norm     = normalize(raw)
        velocity = compute_velocity(norm)

        fatigue_model  = FatigueModel(rate=CONFIG['fatigue_rate'])
        fatigue_per_pt = np.zeros(len(norm), dtype=np.float32)
        stroke_count   = 0
        for j in range(len(norm)):
            fatigue_per_pt[j] = fatigue_model(stroke_count)
            if raw[j, 3] == 1.0:
                stroke_count += 1

        pressure_model = PressureModel(base_pressure=CONFIG['base_pressure'])
        pressure       = pressure_model.compute(norm, fatigue_level=float(fatigue_per_pt.mean()))

        sequence = np.stack([
            norm[:, 0],
            norm[:, 1],
            velocity,
            pressure,
            raw[:, 3],
            fatigue_per_pt,
        ], axis=1).astype(np.float32)

        # Normalized phi — all values in [0, 1]
        phi = np.array([
            CONFIG['fatigue_rate'] / 0.01,
            CONFIG['tremor_freq']  / 12.0,
            CONFIG['tremor_amplitude'],
            CONFIG['base_pressure'],
            writer_id / max(writer_counter, 1),
        ], dtype=np.float32)

        # Character alignment positions — NEW
        char_positions = compute_char_positions(
            n_points = len(norm),
            text     = text,
            max_len  = CONFIG['max_seq_len'],
        )

        split    = 'val' if (i % val_every == 0) else 'train'
        save_dir = os.path.join(out_dir, split)
        os.makedirs(save_dir, exist_ok=True)

        np.save(os.path.join(save_dir, f'sample_{i:06d}.npy'), {
            'sequence':       sequence,
            'biophysical':    phi,
            'text':           text,
            'writer_id':      writer_id,
            'char_positions': char_positions,   # NEW
        })
        saved += 1

        if (i + 1) % 500 == 0:
            print(f"  processed {i+1}/{len(xml_files)} ...")

    print(f"\nDone. Saved {saved} samples  |  {writer_counter} unique writers")


if __name__ == '__main__':
    preprocess_dataset(
        raw_dir = './data/raw/lineStrokes',
        out_dir = CONFIG['data_root'],
    )

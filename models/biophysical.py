# models/biophysical.py
import numpy as np


class FatigueModel:
    """
    φ_f(n) = 1 − exp(−rate × n)

    Fatigue accumulates over strokes. As φ_f → 1:
      - Strokes slow down (lognormal μ increases)
      - Timing becomes noisier (σ widens)
      - Amplitude drops (D reduces)
    """

    def __init__(self, rate=0.002, writer_resistance=1.0):
        self.rate = rate / max(writer_resistance, 1e-6)

    def level(self, stroke_index):
        return float(1.0 - np.exp(-self.rate * stroke_index))

    def __call__(self, stroke_index):
        return self.level(stroke_index)

    def modulate(self, params, stroke_index):
        """
        Apply fatigue to Sigma-Lognormal params (K, 4) = [D, t0, mu, sigma].
        """
        phi_f  = self.level(stroke_index)
        params = params.copy()
        params[:, 0] *= (1 - 0.20 * phi_f)    # D  drops up to 20%
        params[:, 2] += (0.30 * phi_f)          # mu shifts later
        params[:, 3] *= (1 + 0.50 * phi_f)     # sigma widens
        return params


class TremorModel:
    """
    Band-limited oscillatory perturbation on (x, y).

    Healthy writer: amplitude ≈ 0
    Essential tremor: 4–8 Hz, amplitude 0.1–0.5
    Parkinson's:     3–6 Hz, amplitude 0.3–1.0
    """

    def __init__(self, frequency=8.0, amplitude=0.0,
                 writer_phase=0.0, fatigue_amplification=2.0):
        self.frequency             = frequency
        self.amplitude             = amplitude
        self.phase                 = writer_phase
        self.fatigue_amplification = fatigue_amplification

    def apply(self, points, fatigue_level=0.0):
        """
        Args:
            points:        (N, 4) [x, y, time, pen_up]
            fatigue_level: scalar in [0, 1]
        Returns:
            perturbed points, same shape
        """
        points  = points.copy()
        t       = points[:, 2]
        eff_amp = self.amplitude * (1 + self.fatigue_amplification * fatigue_level)

        if eff_amp < 1e-6:
            return points

        omega = 2 * np.pi * self.frequency
        tremor_x = eff_amp * (
            np.sin(omega * t + self.phase) +
            0.3 * np.sin(2 * omega * t + self.phase * 1.5) +
            0.05 * np.random.randn(len(t))
        )
        tremor_y = eff_amp * (
            np.cos(omega * t + self.phase + np.pi / 4) +
            0.3 * np.cos(2 * omega * t + self.phase * 1.5)
        )
        points[:, 0] += tremor_x
        points[:, 1] += tremor_y
        return points


class PressureModel:
    """
    Pen pressure follows:
      - High speed  → low pressure  (pen skims across)
      - Low speed   → high pressure (pen dwells)
      - High curve  → higher pressure
      - Fatigue     → slight pressure drop
    """

    def __init__(self, base_pressure=0.5, velocity_coupling=0.4,
                 curvature_coupling=0.15, writer_style=1.0):
        self.base             = base_pressure
        self.vel_coupling     = velocity_coupling
        self.curv_coupling    = curvature_coupling
        self.style            = writer_style

    def compute(self, points, fatigue_level=0.0):
        """
        Args:
            points:        (N, 4) [x, y, time, pen_up]
            fatigue_level: scalar in [0, 1]
        Returns:
            pressure: (N,) float32 in [0, 1]
        """
        if len(points) < 3:
            return np.full(len(points), self.base, dtype=np.float32)

        x, y, t = points[:, 0], points[:, 1], points[:, 2]

        dx  = np.gradient(x)
        dy  = np.gradient(y)
        dta = np.gradient(t) + 1e-8
        speed      = np.sqrt(dx**2 + dy**2) / dta
        speed_norm = speed / (speed.max() + 1e-8)

        ddx       = np.gradient(dx)
        ddy       = np.gradient(dy)
        denom     = (dx**2 + dy**2) ** 1.5 + 1e-8
        curvature = np.abs(dx * ddy - dy * ddx) / denom
        curv_norm = np.clip(curvature / (curvature.max() + 1e-8), 0, 1)

        pressure  = self.base
        pressure -= self.vel_coupling  * speed_norm
        pressure += self.curv_coupling * curv_norm
        pressure *= self.style * (1 - 0.15 * fatigue_level)
        pressure += 0.02 * np.random.randn(len(points))

        return np.clip(pressure, 0.0, 1.0).astype(np.float32)

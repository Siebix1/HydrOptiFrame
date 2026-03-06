from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.interpolate


@dataclass
class RFPulse:
    """
    RF pulse object containing control points, timing, and waveform generation.
    Default waveform generation uses B-spline interpolation.
    """
    amp: np.ndarray
    phi: np.ndarray

    T: float = 1.024
    NT: int = 256
    set_edges_to_zero: bool = False

    waveform_type: str = "spline"   # default
    flip: float = 30.0
    spline_order: int = 2

    # Extra parameters for non-spline pulse types
    waveform_params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.amp = np.asarray(self.amp, dtype=float)
        self.phi = np.asarray(self.phi, dtype=float)

        if self.amp.shape != self.phi.shape:
            raise ValueError("amp and phi must have the same shape.")

        if self.NT < 2:
            raise ValueError("NT must be >= 2.")

        if self.spline_order < 1:
            raise ValueError("spline_order must be >= 1.")

    # -------------------------------------------------------------------------
    # Basic pulse/time properties
    # -------------------------------------------------------------------------

    @property
    def DT(self) -> float:
        return self.T / self.NT

    @property
    def TSEQ(self) -> np.ndarray:
        return np.linspace(0, self.T, self.NT)

    @property
    def amp_with_edges(self) -> np.ndarray:
        if not self.set_edges_to_zero:
            return self.amp
        return np.concatenate(([0.0], self.amp, [0.0]))

    @property
    def phi_with_edges(self) -> np.ndarray:
        if not self.set_edges_to_zero:
            return self.phi
        return np.concatenate(([0.0], self.phi, [0.0]))

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def build_waveform(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Build the waveform according to waveform_type.
        """
        waveform_type = self.waveform_type.lower()

        if waveform_type == "spline":
            return self._build_spline()

        if waveform_type == "rect":
            return self._build_rect()

        if waveform_type == "binomial2":
            return self._build_binomial2()

        if waveform_type == "binomial3":
            return self._build_binomial3()

        if waveform_type == "double_sinc":
            return self._build_double_sinc()

        if waveform_type == "sinc":
            return self._build_sinc()

        raise ValueError(f"Unknown waveform_type: {self.waveform_type}")

    def set_waveform_type(self, waveform_type: str) -> None:
        self.waveform_type = waveform_type

    def set_spline_order(self, spline_order: int) -> None:
        if spline_order < 1:
            raise ValueError("spline_order must be >= 1.")
        self.spline_order = spline_order

    @classmethod
    def from_best_params(
        cls,
        best_params: dict[str, float],
        n_points: int,
        T: float,
        NT: int,
        set_edges_to_zero: bool = False,
        waveform_type: str = "spline",
        flip: float = 30.0,
        spline_order: int = 2,
        waveform_params: dict[str, Any] | None = None,
    ) -> "RFPulse":
        amp = np.array([best_params[f"amp{i+1}"] for i in range(n_points)], dtype=float)
        phi = np.array([best_params[f"phi{i+1}"] for i in range(n_points)], dtype=float)

        return cls(
            amp=amp,
            phi=phi,
            T=T,
            NT=NT,
            set_edges_to_zero=set_edges_to_zero,
            waveform_type=waveform_type,
            flip=flip,
            spline_order=spline_order,
            waveform_params={} if waveform_params is None else waveform_params,
        )

    # -------------------------------------------------------------------------
    # Waveform generators
    # -------------------------------------------------------------------------

    def _build_rect(self) -> tuple[np.ndarray, np.ndarray]:
        ntau = int(self.T / self.DT)
        da = (self.flip / 180.0 * np.pi) / ntau

        b1_amp = da * np.ones(ntau)
        b1_phase = np.zeros(ntau)

        return b1_amp, b1_phase

    def _build_binomial2(self) -> tuple[np.ndarray, np.ndarray]:
        freq1 = self.waveform_params["freq1"]
        freq2 = self.waveform_params["freq2"]
        phi1 = self.waveform_params["phi1"]
        phi2 = self.waveform_params["phi2"]

        ninter = 6
        ntau = int((self.NT - ninter) / 2)
        tau = ntau * self.DT

        da = (self.flip / 180.0 * np.pi) / (2 * tau / self.DT)

        ramp = np.linspace(-0.5, 0.5, ntau)
        inter = np.zeros(ninter)
        const = np.ones(ntau)

        b1_amp = np.concatenate((da * const, inter, da * const))
        b1_phase = np.concatenate((phi1 + freq1 * ramp, inter, phi2 + freq2 * ramp))

        return b1_amp, b1_phase

    def _build_binomial3(self) -> tuple[np.ndarray, np.ndarray]:
        phi1 = self.waveform_params["phi1"]
        phi2 = self.waveform_params["phi2"]
        phi3 = self.waveform_params["phi3"]
        phi4 = self.waveform_params["phi4"]
        phi5 = self.waveform_params["phi5"]
        phi6 = self.waveform_params["phi6"]

        ninter = 6
        ntau = int((self.NT - 2 * ninter) / 4)
        tau = ntau * self.DT

        b1 = (self.flip / 180.0 * np.pi) / (4 * tau / self.DT)

        ramp1 = np.linspace(-0.5, 0.5, ntau)
        ramp2 = np.linspace(-0.5, 0.5, 2 * ntau)
        inter = np.zeros(ninter)
        const1 = np.ones(ntau)
        const2 = np.ones(2 * ntau)

        b1_amp = np.concatenate((const1 * b1, inter, const2 * b1, inter, const1 * b1))
        b1_phase = np.concatenate((
            phi4 + phi1 * ramp1,
            inter,
            phi5 + phi2 * ramp2,
            inter,
            phi6 + phi3 * ramp1,
        ))

        return b1_amp, b1_phase

    def _build_spline(self) -> tuple[np.ndarray, np.ndarray]:
        a = self.amp_with_edges
        p = self.phi_with_edges

        t = np.linspace(0, self.T, len(a))
        da = self.flip / 180.0 * np.pi
        k = self.spline_order

        pulse_amp_spline = scipy.interpolate.make_interp_spline(t, a, k)
        pulse_phase_spline = scipy.interpolate.make_interp_spline(t, p, k)

        amp_interp = pulse_amp_spline(self.TSEQ).T
        phase_interp = pulse_phase_spline(self.TSEQ).T

        b1_amp = da * amp_interp / np.sum(np.abs(amp_interp))
        b1_phase = phase_interp

        return b1_amp, b1_phase

    def _build_double_sinc(self) -> tuple[np.ndarray, np.ndarray]:
        start1 = self.waveform_params["start1"]
        stop1 = self.waveform_params["stop1"]
        start2 = self.waveform_params["start2"]
        stop2 = self.waveform_params["stop2"]
        startphase1 = self.waveform_params["startphase1"]
        stopphase1 = self.waveform_params["stopphase1"]
        startphase2 = self.waveform_params["startphase2"]
        stopphase2 = self.waveform_params["stopphase2"]

        flip = self.waveform_params.get("flip", 60.0)

        xsinc1 = np.linspace(start1, stop1, int(self.NT / 2))
        xsinc2 = np.linspace(start2, stop2, int(self.NT / 2))

        alpha = (flip / 180.0 * np.pi) / self.NT * 4.0

        b1_amp = np.concatenate((
            np.abs(alpha * np.sinc(xsinc1)),
            np.abs(alpha * np.sinc(xsinc2)),
        ))
        b1_phase = np.concatenate((
            np.linspace(startphase1, stopphase1, int(self.NT / 2)),
            np.linspace(startphase2, stopphase2, int(self.NT / 2)),
        ))

        return b1_amp, b1_phase

    def _build_sinc(self) -> tuple[np.ndarray, np.ndarray]:
        start1 = self.waveform_params["start1"]
        stop1 = self.waveform_params["stop1"]
        phi1 = self.waveform_params["phi1"]
        phi2 = self.waveform_params["phi2"]

        flip = self.waveform_params.get("flip", 60.0)

        xsinc1 = np.linspace(start1, stop1, int(self.NT))
        alpha = (flip / 180.0 * np.pi) / self.NT

        b1_amp = alpha * np.sinc(xsinc1)
        b1_phase = np.linspace(phi1, phi2, int(self.NT))

        return b1_amp, b1_phase
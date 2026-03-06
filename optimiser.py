from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import optuna
from numba import njit

from pulse import RFPulse

@njit
def _pulse_offset_relax_kernel(B1_amp, B1_phase, dF, dt, T1, T2, nt):
    M0 = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    nf = len(dF)
    M = np.zeros((nf, nt, 3), dtype=np.float64)

    for j in range(nf):
        a, b = _freeprecess(dt, T1, T2, 0.0)

        for i in range(nt):
            rth = _throtoffres(B1_amp[i], B1_phase[i], dF[j], dt)

            if i == 0:
                M[j, i, :] = (a @ rth @ M0.reshape(3, 1) + b).reshape(3,)
            else:
                M[j, i, :] = (a @ rth @ M[j, i - 1, :].reshape(3, 1) + b).reshape(3,)

    return M


@njit
def _zrot(phi):
    return np.array((
        (np.cos(phi), np.sin(phi), 0.0),
        (-np.sin(phi), np.cos(phi), 0.0),
        (0.0, 0.0, 1.0),
    ))


@njit
def _xrot(phi):
    return np.array((
        (1.0, 0.0, 0.0),
        (0.0, np.cos(phi), np.sin(phi)),
        (0.0, -np.sin(phi), np.cos(phi)),
    ))


@njit
def _yrot(phi):
    return np.array((
        (np.cos(phi), 0.0, -np.sin(phi)),
        (0.0, 1.0, 0.0),
        (np.sin(phi), 0.0, np.cos(phi)),
    ))


@njit
def _throtoffres(phi, theta, domega, dt):
    alpha = 2 * np.pi * dt * domega / 1000.0
    alphap = np.sqrt(alpha**2 + phi**2)

    if phi == 0.0:
        beta = np.pi / 2
    else:
        beta = np.arctan(alpha / phi)

    Rz = _zrot(theta)
    Rz_n = _zrot(-theta)
    Ry = _yrot(beta)
    Ry_n = _yrot(-beta)
    Rx = _xrot(alphap)

    return Rz @ Ry @ Rx @ Ry_n @ Rz_n


@njit
def _freeprecess(T, T1, T2, df):
    E1 = np.exp(-T / T1)
    E2 = np.exp(-T / T2)
    alpha = 2 * np.pi * T * df / 1000.0

    A = np.array((
        (E2, 0.0, 0.0),
        (0.0, E2, 0.0),
        (0.0, 0.0, E1),
    ))
    Rz = _zrot(alpha)

    Afp = A @ Rz
    Bfp = np.array((0.0, 0.0, 1.0 - E1)).reshape((3, 1))

    return Afp, Bfp


@njit
def _pulse_offset_relax_kernel(B1_amp, B1_phase, dF, dt, T1, T2, nt):
    M0 = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    nf = len(dF)
    M = np.zeros((nf, nt, 3), dtype=np.float64)

    for j in range(nf):
        a, b = _freeprecess(dt, T1, T2, 0.0)

        for i in range(nt):
            rth = _throtoffres(B1_amp[i], B1_phase[i], dF[j], dt)

            if i == 0:
                M[j, i, :] = (a @ rth @ M0.reshape(3, 1) + b).reshape(3,)
            else:
                M[j, i, :] = (a @ rth @ M[j, i - 1, :].reshape(3, 1) + b).reshape(3,)

    return M

@dataclass
class PulseOptimiser:
    pulse_template: RFPulse

    n_points: int = 15
    n_epochs: int = 1000
    amp_lim_low: float = 0.01
    amp_lim_high: float = 1.0
    phi_lim: float = 4 * np.pi
    sigma0: float = 1 / 25

    GYR: float = 42577478.518
    SYSFIELD: float = 3.0
    DELTAFREQFAT: float = -0.0000034

    TE: float = 4.78
    TR: float = 2.35
    T1: float = 1480.0
    T2: float = 50.0

    NF: int = 200
    NZ: int = 100
    F: float = 1000.0
    FOV: float = 0.44

    FLIPMIN: float = 1.0
    FLIPMAX: float = 90.0
    NFLIP: int = 100

    CORRCOEF: float = 1.0

    L1: float = 100.0
    L2: float = 5.0
    L3: float = 30.0

    FATBAND: float = 50.0
    WATBAND: float = 10.0

    study: optuna.Study | None = field(default=None, init=False)
    best_pulse: RFPulse | None = field(default=None, init=False)

    @property
    def SYSFREQ(self) -> float:
        return self.GYR * self.SYSFIELD

    @property
    def FATFREQ(self) -> float:
        return self.SYSFREQ * self.DELTAFREQFAT

    @property
    def DF(self) -> np.ndarray:
        return np.linspace(-self.F, self.F, self.NF)

    @property
    def DZ(self) -> np.ndarray:
        return np.linspace(-self.FOV, self.FOV, self.NZ)

    @property
    def SLICETHICKNESS(self) -> float:
        return self.FOV / 3.5

    def _suggest_pulse(self, trial: optuna.Trial) -> RFPulse:
        amp = np.array(
            [
                trial.suggest_float(
                    f"amp{i+1}",
                    self.amp_lim_low,
                    self.amp_lim_high,
                    step=0.001,
                )
                for i in range(self.n_points)
            ],
            dtype=float,
        )

        phi = np.array(
            [
                trial.suggest_float(
                    f"phi{i+1}",
                    -self.phi_lim,
                    self.phi_lim,
                    step=0.001,
                )
                for i in range(self.n_points)
            ],
            dtype=float,
        )

        return RFPulse(
            amp=amp,
            phi=phi,
            T=self.pulse_template.T,
            NT=self.pulse_template.NT,
            set_edges_to_zero=self.pulse_template.set_edges_to_zero,
            waveform_type=self.pulse_template.waveform_type,
            flip=self.pulse_template.flip,
            spline_order=self.pulse_template.spline_order,
            waveform_params=self.pulse_template.waveform_params.copy(),
        )

    def pulse_offset_relax(
        self,
        B1_amp: np.ndarray,
        B1_phase: np.ndarray,
        dF: np.ndarray | None = None,
    ) -> np.ndarray:
        if dF is None:
            dF = self.DF

        return _pulse_offset_relax_kernel(
            B1_amp=B1_amp,
            B1_phase=B1_phase,
            dF=dF,
            dt=self.pulse_template.DT,
            T1=self.T1,
            T2=self.T2,
            nt=self.pulse_template.NT,
        )

    def b1_sim_off(
        self,
        pulse: RFPulse,
        dF: np.ndarray | None = None,
    ) -> np.ndarray:
        if dF is None:
            dF = self.DF

        flip_values = np.linspace(self.FLIPMIN, self.FLIPMAX, self.NFLIP)
        M = np.zeros((self.NFLIP, len(dF), self.pulse_template.NT, 3), dtype=np.float64)

        for i, flip in enumerate(flip_values):
            temp_pulse = RFPulse(
                amp=pulse.amp.copy(),
                phi=pulse.phi.copy(),
                T=pulse.T,
                NT=pulse.NT,
                set_edges_to_zero=pulse.set_edges_to_zero,
                waveform_type=pulse.waveform_type,
                flip=float(flip),
                spline_order=pulse.spline_order,
                waveform_params=pulse.waveform_params.copy(),
            )

            B1_amp, B1_phase = temp_pulse.build_waveform()
            M[i, :, :, :] = self.pulse_offset_relax(B1_amp, B1_phase, dF)

        return M
    
    def _fat_water_ratio(self, M_xy: np.ndarray) -> float:
        fatband = self.FATBAND * self.CORRCOEF
        s1 = self.FATFREQ - fatband
        s2 = self.FATFREQ + fatband

        waterband = 20 * self.CORRCOEF
        s3 = -waterband
        s4 = waterband

        d_f = 2 * self.F / self.NF

        start = int((self.F + s1) / d_f)
        stop = int((self.F + s2) / d_f)

        startw = int((self.F + s3) / d_f)
        stopw = int((self.F + s4) / d_f)

        fat_signal = np.mean(M_xy[start:stop])
        water_signal = np.mean(M_xy[startw:stopw])

        return fat_signal / water_signal


    def _water_fat_ratio(self, M_xy: np.ndarray) -> float:
        fatband = 20
        s1 = self.FATFREQ - fatband
        s2 = self.FATFREQ + fatband

        waterband = 50
        s3 = -waterband
        s4 = waterband

        d_f = 2 * self.F / self.NF

        start = int((self.F + s1) / d_f)
        stop = int((self.F + s2) / d_f)

        startw = int((self.F + s3) / d_f)
        stopw = int((self.F + s4) / d_f)

        fat_signal = np.mean(M_xy[start:stop])
        water_signal = np.mean(M_xy[startw:stopw])

        return water_signal / fat_signal


    def _derivative_fat(self, M_xy: np.ndarray) -> float:
        s1 = -540
        s2 = -340
        d_f = 2 * self.F / self.NF

        start = int((self.F + s1) / d_f)
        stop = int((self.F + s2) / d_f)

        dfilter = np.array([-1.0, 0.0, 1.0])
        derivative_sum = np.convolve(M_xy[start:stop], dfilter, mode="same")

        return float(np.mean(derivative_sum**2))


    def _mean_water(self, M_xy: np.ndarray) -> float:
        waterband = 20 * self.CORRCOEF
        s3 = -waterband
        s4 = waterband

        d_f = 2 * self.F / self.NF

        startw = int((self.F + s3) / d_f)
        stopw = int((self.F + s4) / d_f)

        return float(np.mean(M_xy[startw:stopw]))


    def _mean_fat(self, M_xy: np.ndarray) -> float:
        fatband = 20
        s3 = self.FATFREQ - fatband
        s4 = self.FATFREQ + fatband

        d_f = 2 * self.F / self.NF

        start = int((self.F + s3) / d_f)
        stop = int((self.F + s4) / d_f)

        return float(np.mean(M_xy[start:stop]))


    def _l2_fat(self, M_xy: np.ndarray) -> float:
        fatband = self.FATBAND * self.CORRCOEF
        s3 = self.FATFREQ - fatband
        s4 = self.FATFREQ + fatband

        d_f = 2 * self.F / self.NF

        start = int((self.F + s3) / d_f)
        stop = int((self.F + s4) / d_f)

        offset_band = 0
        offset_indice = int(offset_band / d_f)

        weight_fltr = np.ones((stop - start + offset_indice))
        center = int((stop - start) / 2)

        weight_fltr[center - 5:center + 5] = 20
        weight_fltr[center - 2:center + 2] = 140

        M1 = M_xy[start:stop + offset_indice] * weight_fltr
        return float(np.mean(M1**2))


    def _l2_water(self, M_xy: np.ndarray) -> float:
        watband = 50
        s3 = -watband
        s4 = watband

        d_f = 2 * self.F / self.NF

        start = int((self.F + s3) / d_f)
        stop = int((self.F + s4) / d_f)

        offset_band = 0
        offset_indice = int(offset_band / d_f)

        weight_fltr = np.ones((stop - start + offset_indice))
        center = int((stop - start) / 2)

        weight_fltr[center - 5:center + 5] = 20
        weight_fltr[center - 2:center + 2] = 80

        M1 = M_xy[start:stop + offset_indice] * weight_fltr
        return float(np.mean(M1**2))


    def composed_loss(self, M: np.ndarray, verbose: int = 0) -> float:
        M_xy = np.abs(M[:, self.pulse_template.NT - 1, 0] + 1j * M[:, self.pulse_template.NT - 1, 1])

        mean_w = self._mean_water(M_xy)
        mean_l2_f = self._l2_fat(M_xy)
        contrast_f_w = self._fat_water_ratio(M_xy)

        if verbose == 1:
            print("mean water =", self.L2 / mean_w)
            print("L2 fat =", self.L1 * mean_l2_f)
            print("fat_water_ratio =", contrast_f_w)

        return contrast_f_w + self.L2 / mean_w + self.L1 * mean_l2_f

    def objective(self, trial: optuna.Trial) -> float:
        pulse = self._suggest_pulse(trial)
        B1_amp, B1_phase = pulse.build_waveform()
        M = self.pulse_offset_relax(B1_amp, B1_phase, self.DF)
        return float(self.composed_loss(M))

    def optimise(self) -> optuna.Study:
        sampler = optuna.samplers.CmaEsSampler(
            n_startup_trials=int(np.round(self.n_epochs / 10)),
            sigma0=self.sigma0,
        )

        self.study = optuna.create_study(sampler=sampler, direction="minimize")
        self.study.optimize(self.objective, n_trials=self.n_epochs)

        self.best_pulse = RFPulse.from_best_params(
            best_params=self.study.best_params,
            n_points=self.n_points,
            T=self.pulse_template.T,
            NT=self.pulse_template.NT,
            set_edges_to_zero=self.pulse_template.set_edges_to_zero,
            waveform_type=self.pulse_template.waveform_type,
            flip=self.pulse_template.flip,
            spline_order=self.pulse_template.spline_order,
            waveform_params=self.pulse_template.waveform_params.copy(),
        )
        return self.study

    def get_best_pulse(self) -> RFPulse:
        if self.best_pulse is None:
            raise RuntimeError("Run optimise() first.")
        return self.best_pulse

    def simulate_pulse(self, pulse: RFPulse):
        B1_amp, B1_phase = pulse.build_waveform()
        M = self.pulse_offset_relax(B1_amp, B1_phase, self.DF)
        return B1_amp, B1_phase, M

    def simulate_b1_map(self, pulse: RFPulse):
        return self.b1_sim_off(pulse, self.DF)
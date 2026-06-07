from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from pulse import RFPulse
from optimiser import PulseOptimiser


@dataclass
class PulseResults:
    pulse: RFPulse
    optimiser: PulseOptimiser
    title: str | None = None
    output_root: str = "Results"
    output_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        if self.title is None:
            self.title = self.build_title()

        root = Path(self.output_root)
        root.mkdir(parents=True, exist_ok=True)

        self.output_dir = self._create_output_dir(root)

    def build_title(self) -> str:
        return (
            f"spl_{int(self.optimiser.SYSFIELD * 10)}T_"
            f"{self.optimiser.n_points}p_{int(self.pulse.T * 1000)}ms_"
            f"{int(self.optimiser.L2 * 10000)}L2_{self.optimiser.FATBAND}FB"
        )

    def _create_output_dir(self, root: Path) -> Path:
        """
        Create a unique output folder for the current optimisation run.

        Example:
        HydrOptiFrame_Pulse_3T_2p000ms_20260306_153512
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        field_str = f"{self.optimiser.SYSFIELD:g}T".replace(".", "p")
        duration_str = f"{self.pulse.T:.3f}ms".replace(".", "p")

        folder_name = f"HydrOptiFrame_Pulse_{field_str}_{duration_str}_{timestamp}"
        output_dir = root / folder_name
        output_dir.mkdir(parents=True, exist_ok=False)

        return output_dir

    # -------------------------------------------------------------------------
    # Main workflow methods
    # -------------------------------------------------------------------------
    def plot_all(self) -> None:
        b1_amp, b1_phase, M = self.optimiser.simulate_pulse(self.pulse)

        m_xy = np.abs(M[:, :, 0] + 1j * M[:, :, 1])

        self.plot_pulse_amp_phase_freq_spl(
            m_xy=m_xy,
            B1_amp=b1_amp,
            B1_phase=b1_phase,
            amp=self.pulse.amp_with_edges,
            phi=self.pulse.phi_with_edges,
        )

        self.plot_amp_phase(b1_amp, b1_phase)

        mb1_full = self.optimiser.simulate_b1_map(self.pulse)
        mb1 = np.abs(mb1_full[:, :, -1, 0] + 1j * mb1_full[:, :, -1, 1])

        self.plot_b1_map(mb1)

        if self.optimiser.study is not None:
            trial_values = [
                trial.value
                for trial in self.optimiser.study.trials
                if trial.value is not None
            ]

            if trial_values:
                best_loss_over_time = np.minimum.accumulate(np.asarray(trial_values))
                self.plot_loss(best_loss_over_time)

    def export_pta(self, comment: str | None = None) -> None:
        """
        Export the optimised pulse in PTA format.
        """
        b1_amp, b1_phase = self.pulse.build_waveform()

        if comment is None:
            comment = (
                f"fatfreq = {self.optimiser.FATFREQ} "
                f"l1 = {self.optimiser.L1}"
            )

        self._pta_gen(
            title=self.title,
            B1_amp=b1_amp,
            B1_phase=-b1_phase,  # sign convention
            T=self.pulse.T,
            comment=comment,
        )

    def _pta_gen(
        self,
        title: str,
        B1_amp: np.ndarray,
        B1_phase: np.ndarray,
        T: float,
        comment: str = "No comment provided.",
    ) -> None:
        if len(B1_amp) != len(B1_phase):
            raise ValueError("B1_amp and B1_phase must have the same length.")

        gamma = 42577478.518
        refgrad = 1000 * 5.12 / (gamma * 0.01 * T)

        B1_amp = B1_amp / np.max(B1_amp)
        B1_norm = B1_amp * np.exp(1j * B1_phase)

        ampInt = np.abs(np.sum(B1_norm))
        powerInt = np.sum(np.abs(B1_norm**2))
        absInt = np.sum(np.abs(B1_norm))

        output_file = self.output_dir / f"{title}.pta"

        with open(output_file, "w", encoding="utf-8", newline="") as f:
            f.write(f"PULSENAME:\t{title}\r\n")
            f.write(f"COMMENT:\t{comment}\r\n")
            f.write(f"REFGRAD:\t{refgrad}\r\n")
            f.write("MINSLICE:\t10\r\n")
            f.write("MAXSLICE:\t500\r\n")
            f.write(f"AMPINT:\t{ampInt}\r\n")
            f.write(f"POWERINT:\t{powerInt}\r\n")
            f.write(f"ABSINT:\t{absInt}\r\n")
            f.write("\t\r\n")

            for i in range(len(B1_amp)):
                phase_wrapped = (B1_phase[i] + np.pi) % (2 * np.pi) - np.pi

                f.write(
                    f"{np.round(B1_amp[i], 8)}\t"
                    f"{np.round(phase_wrapped, 8)}\t"
                    f";\t({i})\r\n"
                )

        print("Pulse file done")

    def _format_value(self, value) -> str:
        if isinstance(value, np.ndarray):
            return np.array2string(
                value,
                precision=6,
                separator=", ",
                threshold=1000000,
                max_line_width=140,
            )

        if isinstance(value, dict):
            lines = []
            for key, val in value.items():
                lines.append(f"  {key}: {self._format_value(val)}")
            return "\n".join(lines)

        return str(value)

    def write_report(self) -> None:
        """
        Write a text report containing all relevant pulse and simulation
        parameters for this optimisation run.
        """
        report_path = self.output_dir / "report.txt"
        b1_amp, b1_phase = self.pulse.build_waveform()

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("HydrOptiFrame report\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Run folder : {self.output_dir.name}\n")
            f.write(f"Title      : {self.title}\n")
            f.write(
                "Timestamp  : "
                + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                + "\n\n"
            )

            f.write("Pulse parameters\n")
            f.write("-" * 80 + "\n")
            f.write(f"T [ms]            : {self.pulse.T}\n")
            f.write(f"NT                : {self.pulse.NT}\n")
            f.write(f"DT [ms]           : {self.pulse.DT}\n")
            f.write(f"flip [deg]        : {self.pulse.flip}\n")
            f.write(f"waveform_type     : {self.pulse.waveform_type}\n")
            f.write(f"spline_order      : {self.pulse.spline_order}\n")
            f.write(f"set_edges_to_zero : {self.pulse.set_edges_to_zero}\n")
            f.write(f"waveform_params   : {self._format_value(self.pulse.waveform_params)}\n")
            f.write(f"amp               : {self._format_value(self.pulse.amp)}\n")
            f.write(f"phi               : {self._format_value(self.pulse.phi)}\n")
            f.write(f"amp_with_edges    : {self._format_value(self.pulse.amp_with_edges)}\n")
            f.write(f"phi_with_edges    : {self._format_value(self.pulse.phi_with_edges)}\n")
            f.write(f"B1_amp            : {self._format_value(b1_amp)}\n")
            f.write(f"B1_phase          : {self._format_value(b1_phase)}\n\n")

            f.write("Optimiser / simulation parameters\n")
            f.write("-" * 80 + "\n")
            f.write(f"n_points     : {self.optimiser.n_points}\n")
            f.write(f"n_epochs     : {self.optimiser.n_epochs}\n")
            f.write(f"amp_lim_low  : {self.optimiser.amp_lim_low}\n")
            f.write(f"amp_lim_high : {self.optimiser.amp_lim_high}\n")
            f.write(f"phi_lim      : {self.optimiser.phi_lim}\n")
            f.write(f"sigma0       : {self.optimiser.sigma0}\n\n")

            f.write(f"GYR              : {self.optimiser.GYR}\n")
            f.write(f"SYSFIELD [T]     : {self.optimiser.SYSFIELD}\n")
            f.write(f"SYSFREQ [Hz]     : {self.optimiser.SYSFREQ}\n")
            f.write(f"DELTAFREQFAT     : {self.optimiser.DELTAFREQFAT}\n")
            f.write(f"FATFREQ [Hz]     : {self.optimiser.FATFREQ}\n\n")

            f.write(f"TE [ms] : {self.optimiser.TE}\n")
            f.write(f"TR [ms] : {self.optimiser.TR}\n")
            f.write(f"T1 [ms] : {self.optimiser.T1}\n")
            f.write(f"T2 [ms] : {self.optimiser.T2}\n\n")

            f.write(f"NF             : {self.optimiser.NF}\n")
            f.write(f"NZ             : {self.optimiser.NZ}\n")
            f.write(f"F [Hz]         : {self.optimiser.F}\n")
            f.write(f"FOV            : {self.optimiser.FOV}\n")
            f.write(f"SLICETHICKNESS : {self.optimiser.SLICETHICKNESS}\n")
            f.write(f"DF             : {self._format_value(self.optimiser.DF)}\n")
            f.write(f"DZ             : {self._format_value(self.optimiser.DZ)}\n\n")

            f.write(f"FLIPMIN [deg] : {self.optimiser.FLIPMIN}\n")
            f.write(f"FLIPMAX [deg] : {self.optimiser.FLIPMAX}\n")
            f.write(f"NFLIP         : {self.optimiser.NFLIP}\n\n")

            f.write(f"CORRCOEF     : {self.optimiser.CORRCOEF}\n")
            f.write(f"L1           : {self.optimiser.L1}\n")
            f.write(f"L2           : {self.optimiser.L2}\n")
            f.write(f"L3           : {self.optimiser.L3}\n")
            f.write(f"FATBAND [Hz] : {self.optimiser.FATBAND}\n")
            f.write(f"WATBAND [Hz] : {self.optimiser.WATBAND}\n\n")

            if self.optimiser.study is not None:
                f.write("Optimisation results\n")
                f.write("-" * 80 + "\n")
                f.write(f"best_value : {self.optimiser.study.best_value}\n")

                duration = getattr(self.optimiser, "optimisation_duration", None)
                if duration is not None:
                    f.write(f"optimisation_duration_seconds : {duration:.3f}\n")

                f.write("best_params :\n")
                for key, value in self.optimiser.study.best_params.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")

    # -------------------------------------------------------------------------
    # Plot methods
    # -------------------------------------------------------------------------
    @staticmethod
    def _safe_index(index: int, max_len: int) -> int:
        return int(np.clip(index, 0, max_len - 1))

    def _band_indices(self, center_hz: float, half_width_hz: float) -> tuple[int, int]:
        d_f = 2 * self.optimiser.F / self.optimiser.NF

        start = int((self.optimiser.F + center_hz - half_width_hz) / d_f)
        stop = int((self.optimiser.F + center_hz + half_width_hz) / d_f)

        start = self._safe_index(start, self.optimiser.NF)
        stop = self._safe_index(stop, self.optimiser.NF)

        if stop <= start:
            stop = min(start + 1, self.optimiser.NF)

        return start, stop

    def plot_pulse_amp_phase_freq_spl(
        self,
        m_xy: np.ndarray,
        B1_amp: np.ndarray,
        B1_phase: np.ndarray,
        amp: np.ndarray,
        phi: np.ndarray,
    ) -> None:
        """
        Plot only the two frequency-response panels.

        This replaces the previous 2x2 figure. The RF amplitude and phase plots
        are still generated separately by plot_amp_phase().
        """
        dF = self.optimiser.DF
        final_mxy = m_xy[:, self.pulse.NT - 1]

        fat_center = self.optimiser.FATFREQ
        water_center = 0.0

        fat_band = self.optimiser.FATBAND
        water_band = self.optimiser.WATBAND

        fat_start, fat_stop = self._band_indices(fat_center, fat_band)
        water_start, water_stop = self._band_indices(water_center, water_band)

        fat_plot_band = 2 * fat_band
        fat_plot_start, fat_plot_stop = self._band_indices(fat_center, fat_plot_band)

        water_mean = np.mean(final_mxy[water_start:water_stop])
        fat_mean = np.mean(final_mxy[fat_start:fat_stop])
        water_fat_ratio = water_mean / fat_mean if fat_mean > 0 else np.inf

        rect = np.ones(self.pulse.NT)
        ref_flip = self.pulse.flip
        da_ref = (ref_flip / 180 * np.pi) / self.pulse.NT

        sar_relative = np.sum(B1_amp**2) / np.sum((da_ref * rect) ** 2)
        print("SAR =", sar_relative)
        print("Water / fat ratio =", water_fat_ratio)
        print("Fat mean =", fat_mean)

        fig, axs = plt.subplots(1, 2, figsize=(14, 5.2))
        fig.suptitle("Frequency response: " + self.title, fontsize=13)

        # ------------------------------------------------------------------
        # Full |Mxy| response
        # ------------------------------------------------------------------
        axs[0].plot(dF, final_mxy, linewidth=1.8)

        axs[0].axvspan(
            water_center - water_band,
            water_center + water_band,
            color="green",
            alpha=0.18,
            label=f"Water band ±{water_band:g} Hz",
        )

        axs[0].axvspan(
            fat_center - fat_band,
            fat_center + fat_band,
            color="red",
            alpha=0.18,
            label=f"Fat band ±{fat_band:g} Hz",
        )

        axs[0].axvline(
            water_center,
            linestyle="--",
            linewidth=1,
        )

        axs[0].axvline(
            fat_center,
            linestyle="--",
            linewidth=1,
        )

        axs[0].set_xlabel("Frequency shift [Hz]")
        axs[0].set_ylabel("|Mxy| [a.u.]")
        axs[0].set_title("Full frequency response")
        axs[0].set_xlim((-self.optimiser.F, self.optimiser.F))
        axs[0].set_ylim((0, 1.1 * np.max(final_mxy)))
        axs[0].legend(frameon=False, fontsize=8)

        # ------------------------------------------------------------------
        # Fat suppression zoom
        # ------------------------------------------------------------------
        fat_zoom_x = dF[fat_plot_start:fat_plot_stop]
        fat_zoom_y = final_mxy[fat_plot_start:fat_plot_stop]

        axs[1].plot(fat_zoom_x, fat_zoom_y, linewidth=1.8)

        axs[1].axvspan(
            fat_center - fat_band,
            fat_center + fat_band,
            color="red",
            alpha=0.18,
            label=f"Fat band ±{fat_band:g} Hz",
        )

        axs[1].axvline(
            fat_center,
            linestyle="--",
            linewidth=1,
        )

        axs[1].set_xlabel("Frequency shift [Hz]")
        axs[1].set_ylabel("|Mxy| [a.u.]")
        axs[1].set_title("Fat suppression zoom")

        if len(fat_zoom_y) > 0 and np.max(fat_zoom_y) > 0:
            axs[1].set_ylim((0, 1.1 * np.max(fat_zoom_y)))

        axs[1].legend(frameon=False, fontsize=8)

        fig.tight_layout()

        plt.savefig(
            self.output_dir / f"Pulse_{self.title}.png",
            dpi=220,
            bbox_inches="tight",
        )
        plt.close(fig)

        print("Frequency response plots done")

    def plot_amp_phase(self, B1_amp: np.ndarray, B1_phase: np.ndarray) -> None:
        Tseq = self.pulse.TSEQ

        fig, ax1 = plt.subplots(figsize=(5.5, 4.0))

        color = "tab:red"
        ax1.set_xlabel("Time [ms]")
        ax1.set_ylabel("B1 amplitude [T]", color=color)
        ax1.plot(
            Tseq[:],
            B1_amp[:] / (self.optimiser.GYR * self.pulse.DT / 1000),
            color=color,
            linewidth=1.8,
        )
        ax1.tick_params(axis="y", labelcolor=color)
        ax1.set_xlim((0, self.pulse.T))

        max_amp = B1_amp.max() / (self.optimiser.GYR * self.pulse.DT / 1000)
        ax1.set_ylim((0, 1.1 * max_amp))

        ax2 = ax1.twinx()

        color = "tab:blue"
        ax2.set_ylabel("Phase [rad]", color=color)
        ax2.plot(Tseq[:], -B1_phase[:], color=color, linewidth=1.8)
        ax2.tick_params(axis="y", labelcolor=color)
        ax2.set_xlim((0, self.pulse.T))

        ax1.set_title("RF pulse amplitude and phase")

        fig.tight_layout()

        plt.savefig(
            self.output_dir / f"pulse_amp_phase_{self.title}.png",
            dpi=220,
            bbox_inches="tight",
        )
        plt.close(fig)

        print("Amplitude/phase plotted!")

    def plot_amp_phase_normed(self, B1_amp: np.ndarray, B1_phase: np.ndarray) -> None:
        Tseq = self.pulse.TSEQ

        fig, ax1 = plt.subplots(figsize=(5.5, 4.0))

        color = "tab:red"
        ax1.set_xlabel("Time [ms]")
        ax1.set_ylabel("Normalised B1 amplitude", color=color)
        ax1.plot(Tseq[:], B1_amp / np.max(B1_amp), color=color, linewidth=1.8)
        ax1.tick_params(axis="y", labelcolor=color)
        ax1.set_xlim((0, self.pulse.T))

        ax2 = ax1.twinx()

        color = "tab:blue"
        ax2.set_ylabel("Normalised phase", color=color)

        phase_norm = (
            B1_phase / np.max(np.abs(B1_phase))
            if np.max(np.abs(B1_phase)) > 0
            else B1_phase
        )

        ax2.plot(Tseq[:], -phase_norm, color=color, linewidth=1.8)
        ax2.tick_params(axis="y", labelcolor=color)
        ax2.set_xlim((0, self.pulse.T))

        ax1.set_title("Normalised RF pulse amplitude and phase")

        fig.tight_layout()

        plt.savefig(
            self.output_dir / f"pulse_amp_phase_normed_{self.title}.png",
            dpi=220,
            bbox_inches="tight",
        )
        plt.close(fig)

        print("Normalised amplitude/phase plotted!")

    def plot_b1_map(self, m_b1: np.ndarray) -> None:
        fig, ax = plt.subplots(figsize=(5.5, 4.2))

        im = ax.imshow(
            m_b1,
            extent=[
                -self.optimiser.F,
                self.optimiser.F,
                self.optimiser.FLIPMIN,
                self.optimiser.FLIPMAX,
            ],
            aspect="auto",
            origin="lower",
        )

        ax.set_ylabel("Flip angle [°]")
        ax.set_xlabel("RF offset frequency [Hz]")
        ax.set_title("B1 / flip-angle response")

        cbar = ax.figure.colorbar(im)
        cbar.outline.set_visible(False)
        cbar.ax.set_ylabel("|Mxy| [a.u.]", rotation=90, va="bottom")

        fig.tight_layout()

        plt.savefig(
            self.output_dir / f"B1_map_{self.title}.png",
            dpi=220,
            bbox_inches="tight",
        )
        plt.close(fig)

        print("B1_map plotted!")

    def plot_loss(self, loss: np.ndarray) -> None:
        n_epoch = len(loss)

        fig, ax = plt.subplots(figsize=(6.5, 4.0))

        ax.plot(np.linspace(0, n_epoch - 1, n_epoch), loss, linewidth=1.8)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.set_title("Optimisation loss")
        ax.set_xlim((0, n_epoch - 1))
        ax.set_ylim((0, np.max(loss) * 1.1))

        fig.tight_layout()

        plt.savefig(
            self.output_dir / f"loss_{self.title}_{int(self.pulse.T)}ms.png",
            dpi=220,
            bbox_inches="tight",
        )
        plt.close(fig)

        print("Loss plotted!")

    def plot_amp(self, B1_amp: np.ndarray, amp: np.ndarray, i: int) -> None:
        amp_dir = self.output_dir / "amp"
        amp_dir.mkdir(parents=True, exist_ok=True)

        B1_amp_tesla = B1_amp * 1_000_000 / (
            self.optimiser.GYR * self.pulse.DT / 1000
        )

        tspace = np.linspace(0, self.pulse.T, len(amp))
        indexes = np.linspace(0, self.pulse.NT - 1, len(amp)).astype(int)

        fig, ax = plt.subplots(figsize=(4.5, 3.0))

        ax.plot(self.pulse.TSEQ, B1_amp_tesla)
        ax.plot(tspace, B1_amp_tesla[indexes], "ro")
        ax.set_xlabel("Time [ms]")
        ax.set_ylabel("B1 amplitude [µT]")
        ax.set_title("Pulse amplitude")
        ax.set_xlim((0, np.max(self.pulse.TSEQ)))
        ax.set_ylim((0, 10))

        fig.tight_layout()

        plt.savefig(
            amp_dir / f"amp_{i}.png",
            dpi=220,
            bbox_inches="tight",
        )
        plt.close(fig)

    def plot_phase(self, B1_phase: np.ndarray, phi: np.ndarray, i: int) -> None:
        phase_dir = self.output_dir / "phase"
        phase_dir.mkdir(parents=True, exist_ok=True)

        tspace = np.linspace(0, self.pulse.T, len(phi))

        fig, ax = plt.subplots(figsize=(4.5, 3.0))

        ax.plot(self.pulse.TSEQ, B1_phase)
        ax.plot(tspace, phi.astype(float), "ro")
        ax.set_xlabel("Time [ms]")
        ax.set_ylabel("B1 phase [rad]")
        ax.set_title("Pulse phase")
        ax.set_xlim((0, np.max(self.pulse.TSEQ)))
        ax.set_ylim((-1.2 * np.pi, 1.2 * np.pi))

        fig.tight_layout()

        plt.savefig(
            phase_dir / f"phase_{i}.png",
            dpi=220,
            bbox_inches="tight",
        )
        plt.close(fig)

    def plot_3d_vectors(self, M: np.ndarray) -> None:
        vector_dir = self.output_dir / "vectors"
        vector_dir.mkdir(parents=True, exist_ok=True)

        for i in range(self.pulse.NT):
            if i % 8 == 0:
                d_f = 2 * self.optimiser.F / self.optimiser.NF
                fat_freq_index = int((self.optimiser.F + self.optimiser.FATFREQ) / d_f)
                fat_freq_index = self._safe_index(fat_freq_index, self.optimiser.NF)

                v_z = np.array([0.0, 0.0, 0.0])

                Mf = np.concatenate((v_z, M[int(self.optimiser.NF / 2), i, :]))
                Mw = np.concatenate((v_z, M[fat_freq_index, i, :]))

                vectors = np.stack((Mf, Mw), axis=0)

                fig = plt.figure(figsize=(5, 5))
                ax = fig.add_subplot(111, projection="3d")

                for k, vector in enumerate(vectors):
                    col = "g" if k == 0 else "r"

                    ax.quiver(
                        vector[0],
                        vector[1],
                        vector[2],
                        vector[3],
                        vector[4],
                        vector[5],
                        color=col,
                    )

                ax.set_xlim([-1, 1])
                ax.set_ylim([-1, 1])
                ax.set_zlim([0, 1])
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_zlabel("z")

                fig.tight_layout()

                plt.savefig(
                    vector_dir / f"vect_{self.title}_{i}.png",
                    dpi=220,
                    bbox_inches="tight",
                )
                plt.close(fig)

    def plot_amp_phase_points(
        self,
        B1_amp: np.ndarray,
        B1_phase: np.ndarray,
        amp: np.ndarray,
        phi: np.ndarray,
        i: int,
    ) -> None:
        ampphi_dir = self.output_dir / "ampphi"
        ampphi_dir.mkdir(parents=True, exist_ok=True)

        tspace = np.linspace(0, self.pulse.T, len(phi))
        B1_amp_tesla = B1_amp * 1_000_000 / (
            self.optimiser.GYR * self.pulse.DT / 1000
        )
        indexes = np.linspace(0, self.pulse.NT - 1, len(amp)).astype(int)

        fig, ax1 = plt.subplots(figsize=(4.5, 3.0))

        color = "tab:red"
        ax1.set_xlabel("Time [ms]")
        ax1.set_ylabel("B1 amplitude [µT]", color=color)
        ax1.plot(self.pulse.TSEQ, B1_amp_tesla, color=color)
        ax1.plot(tspace, B1_amp_tesla[indexes], "ro")
        ax1.tick_params(axis="y", labelcolor=color)
        ax1.set_xlim((0, np.max(self.pulse.TSEQ)))
        ax1.set_ylim((0, 10))

        ax2 = ax1.twinx()

        color = "tab:blue"
        ax2.set_ylabel("B1 phase [rad]", color=color)
        ax2.plot(self.pulse.TSEQ, B1_phase, color=color)
        ax2.plot(tspace, phi.astype(float), "bo")
        ax2.set_xlim((0, np.max(self.pulse.TSEQ)))
        ax2.set_ylim((-1.2 * np.pi, 1.2 * np.pi))
        ax2.tick_params(axis="y", labelcolor=color)

        fig.tight_layout()

        plt.savefig(
            ampphi_dir / f"ampphi_{i}.png",
            dpi=220,
            bbox_inches="tight",
        )
        plt.close(fig)
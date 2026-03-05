"""
HydrOptiFrame: Water Excitation RF pulse design

Numerical optimisation of an RF pulse waveform for water excitation (WE).
We optimise a set of control points (amplitude and phase), interpolate them
into a smooth waveform (B-spline), simulate magnetisation response via Bloch
equations (small-angle approx), and minimise a composed loss.

Author: Xavier Sieber
Date: 05.03.2026
"""

from __future__ import annotations

import numpy as np
import optuna

from Modules.PulsePTAGen import pta_gen
from Modules import constants, lossfunctions, pulsegen, plots, simulations


# =============================================================================
# Frequency / time axes used throughout the optimisation and simulations
# =============================================================================
TSEQ = np.linspace(0, constants.T, constants.NT)          # time axis (s)
DF = np.linspace(-constants.F, constants.F, constants.NF) # off-resonance axis (Hz)


# =============================================================================
# Optimisation configuration (tune these)
# =============================================================================
SET_EDGES_TO_ZERO = 0        # if True: enforce 0 amplitude/phase at edges
N_POINTS = 15                # number of control points to optimise
N_EPOCHS = 1000              # number of Optuna trials

PI = np.pi                   # use numpy's pi (more accurate than 3.142)
AMP_LIM_LOW = 0.01
AMP_LIM_HIGH = 1.0
PHI_LIM = 4 * PI             # +/- phase limit


def objective(trial: optuna.Trial) -> float:
    """
    Optuna objective:
    1) Suggest control points (amp, phi)
    2) Interpolate them to waveform (B1_amp, B1_phase)
    3) Simulate magnetisation response across DF
    4) Compute and return loss (to minimise)
    """
    # Suggest amplitude and phase control points
    amp = np.array(
        [trial.suggest_float(f"amp{i+1}", AMP_LIM_LOW, AMP_LIM_HIGH, step=0.001)
         for i in range(N_POINTS)],
        dtype=float,
    )
    phi = np.array(
        [trial.suggest_float(f"phi{i+1}", -PHI_LIM, PHI_LIM, step=0.001)
         for i in range(N_POINTS)],
        dtype=float,
    )

    # Optional: enforce zero edges (useful to avoid sharp jump at start/end)
    if SET_EDGES_TO_ZERO:
        amp = np.concatenate(([0.0], amp, [0.0]))
        phi = np.concatenate(([0.0], phi, [0.0]))

    # Interpolate points into a smooth RF waveform
    b1_amp, b1_phase = pulsegen.randBsplineN(amp, phi)

    # Simulate magnetisation response using Bloch equations (small-angle approx)
    M = simulations.pulse_offset_relax(b1_amp, b1_phase, DF)

    # Compute scalar loss from final magnetisation
    return float(lossfunctions.composed_loss(M))


# =============================================================================
# Run optimisation
# =============================================================================
sampler = optuna.samplers.CmaEsSampler(
    n_startup_trials=int(np.round(N_EPOCHS / 10)),
    sigma0=1 / 25,
)

study = optuna.create_study(sampler=sampler, direction="minimize")
study.optimize(objective, n_trials=N_EPOCHS)


# =============================================================================
# Post-processing: rebuild best waveform, simulate, plot, export
# =============================================================================
best_params = study.best_params

# IMPORTANT: don't rely on dict insertion order.
# Reconstruct amp/phi in the same order you defined them in the objective.
amp_best = np.array([best_params[f"amp{i+1}"] for i in range(N_POINTS)], dtype=float)
phi_best = np.array([best_params[f"phi{i+1}"] for i in range(N_POINTS)], dtype=float)

if SET_EDGES_TO_ZERO:
    amp_best = np.concatenate(([0.0], amp_best, [0.0]))
    phi_best = np.concatenate(([0.0], phi_best, [0.0]))

# Interpolate (optionally with a higher resolution parameter, like your "30")
B1_amp, B1_phase = pulsegen.randBsplineN(amp_best, phi_best, 30)

# Simulate response and compute transverse magnitude
M = simulations.pulse_offset_relax(B1_amp, B1_phase, DF)
M_xy = np.abs(M[:, :, 0] + 1j * M[:, :, 1])

# Build a descriptive title / filename tag
title = (
    f"spl_{int(constants.SYSFIELD * 10)}T_"
    f"{N_POINTS}p_{int(constants.T * 1000)}ms_"
    f"{int(constants.L2 * 10000)}L2_{constants.FATBAND}FB"
)

# Plot results
plots.pulse_amp_phase_freq_spl(M_xy, B1_amp, B1_phase, amp_best, phi_best, title)
plots.amp_phase_paramno(B1_amp, B1_phase, title)

# B1-map simulation + plot
MB1 = simulations.b1_sim_off(amp_best, phi_best, DF)
plots.b1_map(MB1)

# Export pulse file
pta_gen(
    title,
    B1_amp[:],
    -B1_phase[:],  # your sign convention
    constants.T,
    f"fatfreq = {constants.FATFREQ} l1 = {constants.L1}",
)
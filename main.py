"""
HydrOptiFrame: Water Excitation RF pulse design

Numerical optimisation of an RF pulse waveform for water excitation (WE).
We optimise a set of control points (amplitude and phase), interpolate them
into a smooth waveform (B-spline), simulate magnetisation response via Bloch
equations (small-angle approx), and minimise a composed loss.

Author: Xavier Sieber
Date: 06.03.2026
"""

from pulse import RFPulse
from optimiser import PulseOptimiser
from results import PulseResults
import numpy as np


def main() -> None:
    """
    Define the pulse template, run the RF pulse optimisation, and post-process
    the optimal solution.
    """

    # -------------------------------------------------------------------------
    # Pulse template definition
    # -------------------------------------------------------------------------
    # The pulse template specifies the waveform model and pulse timing used
    # throughout the optimisation. The amplitude and phase arrays are only
    # placeholders here; during optimisation, these control points are replaced
    # by trial values suggested by Optuna.
    pulse_template = RFPulse(
        amp=np.zeros(15),           # initial placeholder amplitude control points
        phi=np.zeros(15),           # initial placeholder phase control points
        T=2.0,                      # pulse duration [ms]
        NT=256,                     # number of temporal samples
        set_edges_to_zero=False,    # optionally constrain first/last control points to zero
        waveform_type="spline",     # waveform generation model
        spline_order=2,             # B-spline order
        flip=10,                    # target flip angle [deg]
    )

    # -------------------------------------------------------------------------
    # Optimiser definition
    # -------------------------------------------------------------------------
    # The optimiser combines the pulse template with the MR system settings
    # and the numerical optimisation parameters.
    optimiser = PulseOptimiser(
        pulse_template=pulse_template,
        n_points=15,                # number of control points to optimise
        n_epochs=100,               # number of optimisation trials
        SYSFIELD=3.0,               # main magnetic field strength [T]
    )

    # -------------------------------------------------------------------------
    # RF pulse optimisation
    # -------------------------------------------------------------------------
    # Execute the optimisation and retrieve the best pulse found over all trials.
    optimiser.optimise()
    best_pulse = optimiser.get_best_pulse()

    # -------------------------------------------------------------------------
    # Results handling
    # -------------------------------------------------------------------------
    # Create the results object for visualization and export of the optimal pulse.
    results = PulseResults(
        pulse=best_pulse,
        optimiser=optimiser,
    )

    # Generate summary figures of the optimised pulse and its simulated response.
    results.plot_all()

    # Export the final pulse in PTA format for downstream use.
    results.export_pta()


if __name__ == "__main__":
    main()
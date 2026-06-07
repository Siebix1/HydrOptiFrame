import time
import traceback
from pathlib import Path

import numpy as np
import streamlit as st

from pulse import RFPulse
from optimiser import PulseOptimiser
from results import PulseResults
from typing import Optional


st.set_page_config(
    page_title="HydrOptiFrame",
    page_icon="💧",
    layout="wide",
)


st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    div[data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 14px;
        padding: 1rem;
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        min-height: 3rem;
    }

    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 128, 128, 0.18);
    }

    .small-caption {
        color: rgba(128, 128, 128, 0.9);
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.title("HydrOptiFrame")
st.caption("Water-excitation RF pulse optimisation for MRI")

st.markdown(
    """
    HydrOptiFrame designs water-excitation RF pulses by optimising pulse
    amplitude and phase profiles, then simulating the resulting frequency
    response with Bloch-equation-based modelling.
    """
)


if "result_dir" not in st.session_state:
    st.session_state.result_dir = None

if "best_value" not in st.session_state:
    st.session_state.best_value = None

if "optimisation_duration" not in st.session_state:
    st.session_state.optimisation_duration = None

if "last_completed_trials" not in st.session_state:
    st.session_state.last_completed_trials = 0


with st.sidebar:
    st.header("About")
    st.write(
        """
        Configure the RF pulse and optimisation settings, run the optimiser,
        then inspect generated plots, reports, and PTA pulse files.
        """
    )

    st.divider()

    st.subheader("Recommended workflow")
    st.write(
        """
        1. Start with a spline pulse.
        2. Use a small number of trials for testing.
        3. Increase trials for final optimisation.
        4. Export the PTA pulse after checking the response plots.
        """
    )


with st.form("pulse_form"):
    st.subheader("Pulse parameters")

    col1, col2, col3 = st.columns(3)

    with col1:
        n_points = st.number_input(
            "Number of control points",
            min_value=3,
            max_value=100,
            value=15,
            step=1,
        )

        pulse_duration = st.number_input(
            "Pulse duration T [ms]",
            min_value=0.01,
            max_value=20.0,
            value=2.0,
            step=0.1,
        )

        nt = st.number_input(
            "Number of time samples NT",
            min_value=16,
            max_value=4096,
            value=256,
            step=1,
        )

    with col2:
        flip = st.number_input(
            "Flip angle [deg]",
            min_value=0.1,
            max_value=180.0,
            value=10.0,
            step=0.1,
        )

        spline_order = st.number_input(
            "B-spline order",
            min_value=1,
            max_value=5,
            value=2,
            step=1,
        )

        set_edges_to_zero = st.checkbox(
            "Set edges to zero",
            value=False,
        )

    with col3:
        waveform_type = st.selectbox(
            "Waveform type",
            ["spline", "rect", "binomial2", "binomial3", "double_sinc", "sinc"],
            index=0,
        )

    st.divider()

    st.subheader("Optimisation / system parameters")

    col4, col5, col6 = st.columns(3)

    with col4:
        n_epochs = st.number_input(
            "Number of optimisation trials",
            min_value=1,
            max_value=100000,
            value=100,
            step=1,
        )

        sysfield = st.number_input(
            "Field strength [T]",
            min_value=0.1,
            max_value=20.0,
            value=3.0,
            step=0.1,
        )

    with col5:
        amp_lim_low = st.number_input(
            "Amplitude lower bound",
            min_value=0.0,
            max_value=10.0,
            value=0.01,
            step=0.01,
        )

        amp_lim_high = st.number_input(
            "Amplitude upper bound",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.01,
        )

    with col6:
        phi_lim_pi = st.number_input(
            "Phase limit [×π]",
            min_value=0.1,
            max_value=20.0,
            value=4.0,
            step=0.1,
        )

        sigma0 = st.number_input(
            "CMA-ES sigma0",
            min_value=0.0001,
            max_value=10.0,
            value=1 / 25,
            step=0.001,
            format="%.4f",
        )

    submitted = st.form_submit_button(
        "Run optimisation",
        use_container_width=True,
    )


if submitted:
    try:
        if waveform_type != "spline":
            st.warning(
                "The current Optuna workflow is set up for spline pulses. "
                "Other waveform types can exist in RFPulse, but optimisation "
                "currently assumes spline control points."
            )
        else:
            pulse_template = RFPulse(
                amp=np.zeros(int(n_points)),
                phi=np.zeros(int(n_points)),
                T=float(pulse_duration),
                NT=int(nt),
                set_edges_to_zero=bool(set_edges_to_zero),
                waveform_type=waveform_type,
                spline_order=int(spline_order),
                flip=float(flip),
            )

            optimiser = PulseOptimiser(
                pulse_template=pulse_template,
                n_points=int(n_points),
                n_epochs=int(n_epochs),
                SYSFIELD=float(sysfield),
                amp_lim_low=float(amp_lim_low),
                amp_lim_high=float(amp_lim_high),
                phi_lim=float(phi_lim_pi * np.pi),
                sigma0=float(sigma0),
            )

            st.subheader("Optimisation progress")

            progress_bar = st.progress(
                0.0,
                text=f"Starting optimisation: 0/{int(n_epochs)} trials",
            )

            progress_status = st.empty()
            progress_metrics = st.empty()

            start_time = time.perf_counter()

            def update_progress(
                completed_trials: int,
                total_trials: int,
                best_value: Optional[float],
            ) -> None:
                progress = completed_trials / total_trials
                progress = min(max(progress, 0.0), 1.0)

                elapsed = time.perf_counter() - start_time

                progress_bar.progress(
                    progress,
                    text=(
                        f"Optimising pulse: "
                        f"{completed_trials}/{total_trials} trials "
                        f"({progress * 100:.1f}%)"
                    ),
                )

                if best_value is not None:
                    progress_status.caption(
                        f"Current best loss: {best_value:.6g}"
                    )

                with progress_metrics.container():
                    col_a, col_b, col_c = st.columns(3)

                    with col_a:
                        st.metric(
                            "Completed trials",
                            f"{completed_trials}/{total_trials}",
                        )

                    with col_b:
                        st.metric(
                            "Elapsed time",
                            f"{elapsed:.1f} s",
                        )

                    with col_c:
                        if best_value is not None:
                            st.metric(
                                "Current best loss",
                                f"{best_value:.6g}",
                            )
                        else:
                            st.metric(
                                "Current best loss",
                                "—",
                            )

            with st.spinner("Running optimisation..."):
                optimiser.optimise(progress_callback=update_progress)

                progress_bar.progress(
                    1.0,
                    text=f"Optimisation complete: {int(n_epochs)}/{int(n_epochs)} trials",
                )

                optimisation_duration = time.perf_counter() - start_time
                optimiser.optimisation_duration = optimisation_duration

                best_pulse = optimiser.get_best_pulse()

                results = PulseResults(
                    pulse=best_pulse,
                    optimiser=optimiser,
                )

                results.write_report()
                results.plot_all()
                results.export_pta()

                st.session_state.result_dir = str(results.output_dir)
                st.session_state.best_value = float(optimiser.study.best_value)
                st.session_state.optimisation_duration = optimisation_duration
                st.session_state.last_completed_trials = int(n_epochs)

            st.success("Optimisation finished.")

    except Exception as exc:
        st.error(f"Run failed: {exc}")
        st.code(traceback.format_exc())


if st.session_state.best_value is not None:
    st.subheader("Optimisation summary")

    metric_col1, metric_col2, metric_col3 = st.columns(3)

    with metric_col1:
        st.metric(
            label="Best pulse loss",
            value=f"{st.session_state.best_value:.6g}",
        )

    with metric_col2:
        if st.session_state.optimisation_duration is not None:
            st.metric(
                label="Optimisation duration",
                value=f"{st.session_state.optimisation_duration:.1f} s",
            )
        else:
            st.metric(
                label="Optimisation duration",
                value="—",
            )

    with metric_col3:
        st.metric(
            label="Completed trials",
            value=str(st.session_state.last_completed_trials),
        )


if st.session_state.result_dir:
    result_dir = Path(st.session_state.result_dir)

    st.subheader("Output folder")
    st.code(str(result_dir))

    report_path = result_dir / "report.txt"

    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            report_text = f.read()

        with st.expander("Report"):
            st.text(report_text)

    png_files = sorted(result_dir.glob("*.png"))

    if png_files:
        st.subheader("Generated figures")

        frequency_plots = [
            png for png in png_files if png.name.startswith("Pulse_")
        ]

        other_plots = [
            png for png in png_files if not png.name.startswith("Pulse_")
        ]
        
        other_plots = sorted(
            other_plots,
            key=lambda p: (
                not p.name.startswith("loss_"),
                p.name,
            ),
        )

        if frequency_plots:
            st.markdown("### Frequency response")

            for png in frequency_plots:
                st.image(
                    str(png),
                    caption=png.name,
                    use_container_width=True,
                )

        if other_plots:
            st.markdown("### Additional plots")

            cols = st.columns(2)

            for i, png in enumerate(other_plots):
                with cols[i % 2]:
                    st.image(
                        str(png),
                        caption=png.name,
                        use_container_width=True,
                    )

    pta_files = sorted(result_dir.glob("*.pta"))

    if pta_files:
        st.subheader("PTA files")

        for pta in pta_files:
            with open(pta, "rb") as f:
                st.download_button(
                    label=f"Download {pta.name}",
                    data=f,
                    file_name=pta.name,
                    mime="text/plain",
                )
import traceback
from pathlib import Path

import numpy as np
import streamlit as st

from pulse import RFPulse
from optimiser import PulseOptimiser
from results import PulseResults


st.set_page_config(page_title="HydrOptiFrame", layout="wide")
st.title("HydrOptiFrame")
st.caption("Water-excitation RF pulse optimisation for MRI")

# Keep the last result folder in session state so it survives reruns
if "result_dir" not in st.session_state:
    st.session_state.result_dir = None
if "best_value" not in st.session_state:
    st.session_state.best_value = None


with st.form("pulse_form"):
    st.subheader("Pulse parameters")
    col1, col2, col3 = st.columns(3)

    with col1:
        n_points = st.number_input("Number of control points", min_value=3, max_value=100, value=15, step=1)
        pulse_duration = st.number_input("Pulse duration T [ms]", min_value=0.01, max_value=20.0, value=2.0, step=0.1)
        nt = st.number_input("Number of time samples NT", min_value=16, max_value=4096, value=256, step=1)

    with col2:
        flip = st.number_input("Flip angle [deg]", min_value=0.1, max_value=180.0, value=10.0, step=0.1)
        spline_order = st.number_input("B-spline order", min_value=1, max_value=5, value=2, step=1)
        set_edges_to_zero = st.checkbox("Set edges to zero", value=False)

    with col3:
        waveform_type = st.selectbox(
            "Waveform type",
            ["spline", "rect", "binomial2", "binomial3", "double_sinc", "sinc"],
            index=0,
        )

    st.subheader("Optimisation / system parameters")
    col4, col5, col6 = st.columns(3)

    with col4:
        n_epochs = st.number_input("Number of optimisation trials", min_value=1, max_value=100000, value=100, step=1)
        sysfield = st.number_input("Field strength [T]", min_value=0.1, max_value=20.0, value=3.0, step=0.1)

    with col5:
        amp_lim_low = st.number_input("Amplitude lower bound", min_value=0.0, max_value=10.0, value=0.01, step=0.01)
        amp_lim_high = st.number_input("Amplitude upper bound", min_value=0.0, max_value=10.0, value=1.0, step=0.01)

    with col6:
        phi_lim_pi = st.number_input("Phase limit [×π]", min_value=0.1, max_value=20.0, value=4.0, step=0.1)
        sigma0 = st.number_input("CMA-ES sigma0", min_value=0.0001, max_value=10.0, value=1 / 25, step=0.001, format="%.4f")

    submitted = st.form_submit_button("Run optimisation")

if submitted:
    try:
        if waveform_type != "spline":
            st.warning(
                "The current Optuna workflow is set up for spline pulses. "
                "Other waveform types can exist in RFPulse, but optimisation currently assumes spline control points."
            )
        else:
            with st.spinner("Running optimisation..."):
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

                optimiser.optimise()
                best_pulse = optimiser.get_best_pulse()

                results = PulseResults(
                    pulse=best_pulse,
                    optimiser=optimiser,
                )

                results.plot_all()
                results.export_pta()

                st.session_state.result_dir = str(results.output_dir)
                st.session_state.best_value = float(optimiser.study.best_value)

            st.success("Optimisation finished.")

    except Exception as exc:
        st.error(f"Run failed: {exc}")
        st.code(traceback.format_exc())

if st.session_state.best_value is not None:
    st.subheader("Best objective value")
    st.write(st.session_state.best_value)

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
        for png in png_files:
            st.image(str(png), caption=png.name, use_container_width=True)

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
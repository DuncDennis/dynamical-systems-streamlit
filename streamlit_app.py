"""The dynaimcal-systems-streamlit app."""

import streamlit as st

import src.streamlit_fragments as stfrag

def main() -> None:
    with st.sidebar:
        st.header("System: ")
        system_name, system_parameters = stfrag.st_select_system()
        time_steps = stfrag.st_select_time_steps(default_time_steps=10000)

    time_series = stfrag.simulate_trajectory(system_name, system_parameters, time_steps)

    if st.checkbox("Plot time series: "):
        stfrag.st_default_simulation_plot(time_series)

    if st.checkbox("Calculate largest lyapunov exponent: "):
        stfrag.st_largest_lyapunov_exponent(system_name, system_parameters)


main()

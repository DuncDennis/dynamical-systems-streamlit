"""A file containing the streamlit app parts."""

from __future__ import annotations

from typing import Any, Callable

import streamlit as st
import numpy as np

from src import plotly_plots as plpl
import src.simulate_systems as sims
import src.measures as meas

SYSTEM_DICT = {
    "Lorenz63": sims.Lorenz63,
    "Roessler": sims.Roessler,
    "ComplexButterly": sims.ComplexButterly,
    "Chen": sims.Chen,
    "ChuaCircuit": sims.ChuaCircuit,
    "Thomas": sims.Thomas,
    "WindmiAttractor": sims.WindmiAttractor,
    "Rucklidge": sims.Rucklidge,
    "SimplestQuadraticChaotic": sims.SimplestQuadraticChaotic,
    "SimplestCubicChaotic": sims.SimplestCubicChaotic,
    "SimplestPiecewiseLinearChaotic": sims.SimplestPiecewiseLinearChaotic,
    "DoubleScroll": sims.DoubleScroll,
    "LotkaVolterra": sims.LotkaVolterra,
    "SimplestDrivenChaotic": sims.SimplestDrivenChaotic,
    "UedaOscillator": sims.UedaOscillator,
    "Henon": sims.Henon,
    "Logistic": sims.Logistic,
    "KuramotoSivashinsky": sims.KuramotoSivashinsky,
    "Lorenz96": sims.Lorenz96,
}


def st_select_system(systems_sub_section: tuple[str, ...] | None = None,
                     default_parameters: dict[str, dict[str, Any]] | None = None
                     ) -> tuple[str, dict[str, Any]]:
    """ Select the system to simulate and specify the parameters.
    # TODO: update docstring
    Args:
        systems_sub_section: If None, take all in SYSTEM_DICT, else take only subsection.

    Returns:
        tuple with system name and parameters dict

    """
    if systems_sub_section is None:
        system_dict = SYSTEM_DICT
    else:
        system_dict = {system_name: system_class for system_name, system_class in
                       SYSTEM_DICT.items()
                       if system_name in systems_sub_section}
        if len(system_dict) == 0:  # TODO: proper error
            raise Exception(f"The systems in {systems_sub_section} are not accounted for.")

    system_name = st.selectbox('System', system_dict.keys())

    sim_class = system_dict[system_name]

    if default_parameters is None:
        system_parameters = sim_class.default_parameters
        # fullargspec = inspect.getfullargspec(sim_class.__init__)
        # system_parameters = {x: y for x, y in zip(fullargspec[0][1:], fullargspec[3])}
    else:
        if system_name in default_parameters.keys():
            system_parameters = default_parameters[system_name]
        else:  # TODO: proper error
            raise Exception(f"The system specified in default_parameters is not accounted for.")

    with st.expander("Parameters: "):
        for key, val in system_parameters.items():

            val_type = type(val)
            if val_type == float:
                system_parameters[key] = st.number_input(key, value=float(val), step=0.01,
                                                         format="%f")
            elif val_type == int:
                system_parameters[key] = int(st.number_input(key, value=int(val)))
            else:
                raise TypeError("Other default keyword arguments than float and int are currently"
                                "not supported.")

    return system_name, system_parameters


def st_get_model_system(system_name: str, system_parameters: dict[str, Any]
                        ) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, Any]]:
    """Get a app-section to modify the system parameters of the system given by system_name.

    Args:
        system_name: The name of the system. has to be in SYSTEM_DICT.
        system_parameters: The original system_parameters to be modified.

    Returns: The iterator function of the modified model.

    """

    modified_system_parameters = system_parameters.copy()

    relative_change = st.checkbox("Relative change")

    for i, (key, val) in enumerate(modified_system_parameters.items()):
        print(i, (key, val))
        val_type = type(val)
        if relative_change:
            left, right = st.columns(2)

            with right:
                eps = st.number_input("Relative change", value=0.0, step=0.01, format="%f",
                                      key=f"releps{i}")
            if val_type == float:
                new_val = system_parameters[key] * (1 + eps)
            elif val_type == int:
                new_val = int(system_parameters[key] * (1 + eps))
            else:
                raise TypeError(
                    "Other default keyword arguments than float and int are currently"
                    "not supported.")
            with left:
                st.number_input(key, value=new_val, disabled=True, key=f"relative{i}")

        else:
            left, right = st.columns(2)
            with left:
                if val_type == float:
                    new_val = st.number_input(key, value=float(val), step=0.01, format="%f",
                                              key=f"absfloat{i}")
                elif val_type == int:
                    new_val = st.number_input(key, value=int(val), key=f"absint{i}", step=1)
                else:
                    raise TypeError(
                        "Other default keyword arguments than float and int are currently"
                        "not supported.")
            with right:
                if system_parameters[key] == 0:
                    eps = np.nan
                else:
                    eps = new_val / system_parameters[key] - 1
                st.number_input("Relative change", value=eps, step=0.01, format="%f",
                                disabled=True, key=f"abseps{i}")

        modified_system_parameters[key] = new_val

    model_func = SYSTEM_DICT[system_name](**modified_system_parameters).iterate

    return model_func, modified_system_parameters


def st_select_time_steps(default_time_steps: int = 10000) -> int:
    """
    Add number input for time steps.
    """
    return int(st.number_input('time_steps', value=default_time_steps, step=1))


def st_select_time_steps_split_up(default_t_train_disc: int = 1000,
                                  default_t_train_sync: int = 300,
                                  default_t_train: int = 2000,
                                  default_t_pred_disc: int = 1000,
                                  default_t_pred_sync: int = 300,
                                  default_t_pred: int = 5000,
                                  ) -> tuple[int, int, int, int, int, int]:
    """Add number inputs for train discard, train sync, train, pred discard, pred sync and pred t.

    """
    with st.expander("Time steps: "):
        t_train_disc = st.number_input('t_train_disc', value=default_t_train_disc, step=1)
        t_train_sync = st.number_input('t_train_sync', value=default_t_train_sync, step=1)
        t_train = st.number_input('t_train', value=default_t_train, step=1)
        t_pred_disc = st.number_input('t_pred_disc', value=default_t_pred_disc, step=1)
        t_pred_sync = st.number_input('t_pred_sync', value=default_t_pred_sync, step=1)
        t_pred = st.number_input('t_pred', value=default_t_pred, step=1)
        return int(t_train_disc), int(t_train_sync), int(t_train), int(t_pred_disc), \
               int(t_pred_sync), int(t_pred)


@st.experimental_memo
def simulate_trajectory(system_name: str, system_parameters: dict[str, Any],
                        time_steps: int) -> np.ndarray:
    return SYSTEM_DICT[system_name](**system_parameters).simulate(time_steps=time_steps)


@st.experimental_memo
def get_largest_lyapunov_exponent(system_name: str, system_parameters: dict[str, Any],
                                  steps: int = int(1e3), deviation_scale: float = 1e-10,
                                  part_time_steps: int = 15, steps_skip: int = 50) -> np.ndarray:
    sim_instance = SYSTEM_DICT[system_name](**system_parameters)
    starting_point = sim_instance.default_starting_point

    if hasattr(sim_instance, "dt"):
        dt = sim_instance.dt
    else:
        dt = 1.0

    iterator_func = sim_instance.iterate
    lle_conv = meas.largest_lyapunov_exponent(iterator_func, starting_point=starting_point, dt=dt,
                                              steps=steps, part_time_steps=part_time_steps,
                                              deviation_scale=deviation_scale,
                                              steps_skip=steps_skip,
                                              return_convergence=True)
    return lle_conv


def st_largest_lyapunov_exponent(system_name: str, system_parameters: dict[str, Any]):
    left, right = st.columns(2)
    with left:
        steps = int(st.number_input("steps", value=int(1e3)))
    with right:
        part_time_steps = int(st.number_input("time steps of each part", value=15))
    left, right = st.columns(2)
    with left:
        steps_skip = int(st.number_input("steps to skip", value=50, min_value=0))
    with right:
        deviation_scale = 10 ** (float(st.number_input("log (deviation_scale)", value=-10.0)))

    lle_conv = get_largest_lyapunov_exponent(system_name, system_parameters, steps=steps,
                                             part_time_steps=part_time_steps,
                                             deviation_scale=deviation_scale,
                                             steps_skip=steps_skip)
    figs = plpl.multiple_1d_time_series({"LLE convergence": lle_conv}, x_label="N",
                                        y_label="running avg of LLE")
    plpl.multiple_figs(figs)

    largest_lle = lle_conv[-1]
    st.write(f"Largest Lyapunov Exponent = {np.round(largest_lle, 5)}")


def st_default_simulation_plot(time_series):
    x_dim = time_series.shape[1]
    if x_dim == 1:

        figs = plpl.multiple_1d_time_series({"simulated timeseries": time_series, },
                                            x_label="time step", )
        plpl.multiple_figs(figs)

    elif x_dim == 2:
        fig = plpl.multiple_2d_time_series({"simulated timeseries": time_series}, mode="scatter")
        st.plotly_chart(fig)

    elif x_dim == 3:
        fig = plpl.multiple_3d_time_series({"simulated timeseries": time_series}, )
        st.plotly_chart(fig)

    elif x_dim > 3:
        figs = plpl.multiple_time_series_image({"simulated timeseries": time_series},
                                               x_label="time steps",
                                               y_label="dimensions"
                                               )
        plpl.multiple_figs(figs)
    else:
        raise ValueError("x_dim < 1 not supported.")

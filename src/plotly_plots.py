""" A collection of utility plotting functions using plotly"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

DEFAULT_TWO_D_FIGSIZE = (650, 350)
DEFAULT_THREE_D_FIGSIZE = (650, 500)


def multiple_figs(figs: list[go.Figure, ...]) -> None:
    """Utility function to plot multiple figs in streamlit.
    Args:
        figs: List of plotly figures.

    """
    for fig in figs:
        st.plotly_chart(fig)


@st.experimental_memo
def matrix_as_barchart(data_matrix: np.ndarray, x_axis: str = "x_dim", y_axis: str = "y_dim",
                       value_name: str = "value", title: str = "",
                       fig_size: tuple[int, int] = DEFAULT_TWO_D_FIGSIZE,
                       log_y: bool = False, abs_bool: bool = True, barmode: str = "relative"
                       ) -> go.Figure:
    """Plot the absolut values of a matrix as a relative/grouped/subplotted barchart.


    Args:
        data_matrix: 2 dimensional numpy array to visualize.
        x_axis: Name of the x-axis index of the data_matrix. Will be displayed as the x-axis of the
                bar-plot.
        y_axis: Name of the y-axis index of the data_matrix. Will be displayed above the colorsbar.
        value_name: Name of the values within the data_matrix.
        title: Title of the plot.
        fig_size: The size of the figure in (width, height).
        log_y: If true the y axis of the plot will be displayed logarithmically.
        abs_bool: If true the absolute value of data_matrix entries is used.
        barmode: If "relative" the values corresponding to the different y_axis_indices are plotted
                in one bar chart and are stacked on top of each other. If "grouped" they are
                plotted next to each other. If "subplot" there is a new subplot for every y_axis
                index.

    Returns:
        plotly figure.
    """

    x_dim, y_dim = data_matrix.shape

    data_dict = {x_axis: [], y_axis: [], value_name: []}
    for i_x in range(x_dim):
        for i_y in range(y_dim):
            value = data_matrix[i_x, i_y]
            data_dict[x_axis].append(i_x)
            data_dict[value_name].append(value)
            data_dict[y_axis].append(i_y)

    df = pd.DataFrame.from_dict(data_dict)

    if abs_bool:
        abs_value_name = f"absolute of {value_name}"
        df[abs_value_name] = np.abs(df[value_name])
        value_col_to_plot = abs_value_name
    else:
        value_col_to_plot = value_name

    if barmode in ["relative", "grouped"]:
        fig = px.bar(df, x=x_axis, y=value_col_to_plot, color=y_axis,
                     title=title, width=fig_size[0],
                     height=fig_size[1], barmode=barmode)

    elif barmode == "subplot":
        subplot_titles = [f"{title} - {y_axis}: {i_y}" for i_y in range(y_dim)]
        fig = make_subplots(rows=y_dim, cols=1, subplot_titles=subplot_titles)
        for i_y in range(y_dim):
            sub_df = df[df[y_axis] == i_y]
            sub_fig = px.bar(sub_df, x=x_axis, y=value_col_to_plot)

            fig.add_trace(sub_fig["data"][0], row=i_y + 1, col=1)
        fig.update_layout(height=fig_size[1] * y_dim, width=fig_size[0])
        fig.update_yaxes(title=value_col_to_plot)
        fig.update_xaxes(title=x_axis)

    else:
        raise ValueError(f"Value of keyword argument barmode = {barmode} is not supported.")

    if log_y:
        fig.update_yaxes(type="log", exponentformat="E")

    return fig


@st.experimental_memo
def multiple_1d_time_series(time_series_dict: dict[str, np.ndarray], mode: str = "line",
                            line_size: float | None = 1, scatter_size: float | None = 1,
                            title: str | None = None,
                            fig_size: tuple[int, int] = DEFAULT_TWO_D_FIGSIZE,
                            x_scale: float | None = None, x_label: str = "x",
                            y_label: str = "y", dimensions: tuple[int, ...] | None = None
                            ) -> list[go.Figure, ...]:
    """ Plot multiple 1d time_series as a line or scatter plot.
    # TODO: add possibility for vertical lines seperators

    Args:
        time_series_dict: Dict of the form {"timeseries_name_1": time_series_1, ...}.
        mode: "line" or "scatter".
        line_size: If mode = "line" size of lines.
        scatter_size: If mode = "scatter", size of markers.
        title: Title of figure
        fig_size: The size of the figure in (width, height).
        x_scale: Scale the x axis. (Probably time axis).
        x_label: x_label for xaxis.
        y_label: y_label for yaxis.
        dimensions: If the timeseries is multidimensional specify the dimensions to plot. If None
                    All dimensions are plotted beneath each other.

    Returns: plotly figure.

    """
    figs = []
    shape = list(time_series_dict.values())[0].shape
    if len(shape) == 1:
        x_steps, y_dim = shape[0], 1
    else:
        x_steps, y_dim = shape

    if x_scale is None:
        x_array = np.arange(x_steps)
    else:
        x_array = np.arange(x_steps) * x_scale

    if dimensions is None:
        dimensions = tuple(np.arange(y_dim))
    for i in dimensions:
        to_plot_dict = {x_label: [], y_label: [], "label": []}
        for label, time_series in time_series_dict.items():
            if len(shape) == 1:
                time_series = time_series[:, np.newaxis]
            to_plot_dict[x_label].extend(x_array)
            to_plot_dict[y_label].extend(time_series[:, i])
            to_plot_dict["label"].extend([label, ] * time_series.shape[0])

        if len(dimensions) > 1:
            if title is not None:
                title_i = f"{title}: dimension: {i}"
            else:
                title_i = f"Dimension: {i}"
        else:
            title_i = title

        if mode == "line":
            fig = px.line(to_plot_dict, x=x_label, y=y_label, color="label", title=title_i)
            if line_size is not None:
                fig.update_traces(line={"width": line_size})
        elif mode == "scatter":
            fig = px.scatter(to_plot_dict, x=x_label, y=y_label, color="label",
                             title=title_i)
            if scatter_size is not None:
                fig.update_traces(marker={'size': scatter_size})
        else:
            raise Exception(f"mode = {mode} not accounted for.")  # TODO: proper error

        fig.update_layout(height=fig_size[1], width=fig_size[0])
        figs.append(fig)

    return figs


@st.experimental_memo
def multiple_2d_time_series(time_series_dict: dict[str, np.ndarray], mode: str = "line",
                            line_size: float | None = 1, scatter_size: float | None = 1,
                            title: str | None = None,
                            fig_size: tuple[int, int] = DEFAULT_TWO_D_FIGSIZE,
                            x_label: str = "x", y_label: str = "y",
                            ) -> go.Figure:
    """ Plot multiple 2d time_series as a line or scatter plot.

    Args:
        time_series_dict: Dict of the form {"timeseries_name_1": time_series_1, ...}.
        mode: "line" or "scatter".
        line_size: If mode = "line" size of lines.
        scatter_size: If mode = "scatter", size of markers.
        title: Title of figure
        fig_size: The size of the figure in (width, height).
        x_label: x_label for xaxis.
        y_label: y_label for yaxis.

    Returns: plotly figure.

    """

    to_plot_dict = {x_label: [], y_label: [], "label": []}
    for label, time_series in time_series_dict.items():
        to_plot_dict[x_label].extend(time_series[:, 0])
        to_plot_dict[y_label].extend(time_series[:, 1])
        to_plot_dict["label"].extend([label, ] * time_series.shape[0])

    if mode == "line":
        fig = px.line(to_plot_dict, x=x_label, y=y_label, color="label", title=title)
        if line_size is not None:
            fig.update_traces(line={"width": line_size})
    elif mode == "scatter":
        fig = px.scatter(to_plot_dict, x=x_label, y=y_label, color="label", title=title)
        if scatter_size is not None:
            fig.update_traces(marker={'size': scatter_size})
    else:
        raise Exception(f"mode = {mode} not accounted for.")  # TODO: proper error

    fig.update_layout(height=fig_size[1], width=fig_size[0])
    return fig


@st.experimental_memo
def multiple_3d_time_series(time_series_dict: dict[str, np.ndarray], mode: str = "line",
                            line_size: float | None = 1, scatter_size: float | None = 1,
                            title: str | None = None,
                            fig_size: tuple[int, int] = DEFAULT_THREE_D_FIGSIZE,
                            x_label: str = "x", y_label: str = "y", z_label: str = "z"
                            ) -> go.Figure:
    """ Plot multiple 3d time_series as a line or scatter plot.

    Args:
        time_series_dict: Dict of the form {"timeseries_name_1": time_series_1, ...}.
        mode: "line" or "scatter".
        line_size: If mode = "line" size of lines.
        scatter_size: If mode = "scatter", size of markers.
        title: Title of figure
        fig_size: The size of the figure in (width, height).
        x_label: x_label for xaxis.
        y_label: y_label for yaxis.
        z_label: z_label for zaxis.

    Returns: plotly figure.

    """
    to_plot_dict = {x_label: [], y_label: [], z_label: [], "label": []}
    for label, time_series in time_series_dict.items():
        to_plot_dict[x_label].extend(time_series[:, 0])
        to_plot_dict[y_label].extend(time_series[:, 1])
        to_plot_dict[z_label].extend(time_series[:, 2])
        to_plot_dict["label"].extend([label, ] * time_series.shape[0])

    if mode == "line":
        fig = px.line_3d(to_plot_dict, x=x_label, y=y_label, z=z_label, color="label", title=title)
        if line_size is not None:
            fig.update_traces(line={"width": line_size})
    elif mode == "scatter":
        fig = px.scatter_3d(to_plot_dict, x=x_label, y=y_label, z=z_label, color="label",
                            title=title)
        if scatter_size is not None:
            fig.update_traces(marker={'size': scatter_size})
    else:
        raise Exception(f"mode = {mode} not accounted for.")  # TODO: proper error

    fig.update_layout(height=fig_size[1], width=fig_size[0])
    return fig


@st.experimental_memo
def multiple_time_series_image(time_series_dict: dict[str, np.ndarray],
                               fig_size: tuple[int, int] = DEFAULT_TWO_D_FIGSIZE,
                               x_label: str = "x",
                               y_label: str = "y",
                               x_scale: float | None = None,
                               ) -> list[go.Figure, ...]:
    # TODO: add docstring
    # TODO: add possibility for vertical lines seperators
    figs = []
    labels = {"x": x_label, "y": y_label}

    if x_scale is None:
        x_array = None
    else:
        time_steps = list(time_series_dict.values())[0].shape[0]
        x_array = np.arange(time_steps) * x_scale

    for i, (key, val) in enumerate(time_series_dict.items()):
        figs.append(
            px.imshow(val.T, aspect="auto", title=key, width=fig_size[0], height=fig_size[1],
                      labels=labels, x=x_array)
        )
    return figs

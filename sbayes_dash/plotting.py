from __future__ import annotations

import numpy as np
import pandas as pd
from plotly import express as px, graph_objects as go

from sbayes_dash.app_state import AppState
from sbayes_dash.util import min_and_max_with_padding, compute_delaunay, gabriel_graph_from_delaunay, COLOR_0


blank_axis = {
    "showgrid": False,  # thin lines in the background
    "zeroline": False,  # thick line at x=0
    "visible": False,  # numbers below
}


blank_layout = {
    "xaxis_title": "",
    "yaxis_title": "",
    "xaxis": blank_axis,
    "yaxis": blank_axis,
    "plot_bgcolor": "rgba(0, 0, 0, 0)",
    "paper_bgcolor": "rgba(0, 0, 0, 0)",
}


def initialize_results_map(state: AppState):
    """Initialize the map to show sBayes results on the dashboard."""
    fig = px.scatter_geo(
        state.object_data,
        lat="y", lon="x", color="cluster",
        hover_data=["name", "family", "posterior_support", "cluster"],
        projection="natural earth",
        color_discrete_sequence=state.cluster_colors,
    )

    for i in range(state.n_clusters):
        fig_lines = px.line_geo(lat=[None], lon=[None])
        fig = go.Figure(fig.data + fig_lines.data)

    fig.update_layout(
        height=600,
        margin=dict(l=0, r=0, b=0, t=0, pad=0),
        geo=dict(
            lonaxis=dict(
                showgrid=True,
                gridwidth=0.5,
                range=[*min_and_max_with_padding(state.locations[:, 0])],
                dtick=5,
            ),
            lataxis=dict(
                showgrid=True,
                gridwidth=0.5,
                range=[*min_and_max_with_padding(state.locations[:, 1])],
                dtick=5,
            ),
        ),
    )

    # Store figure, lines and scatter in the state object
    state.fig = fig
    state.lines = fig.data[1:]
    state.scatter = fig.data[0]

    # Fix z-order so that lines are behind scatter:
    fig.data = fig.data[::-1]

    return fig


def create_data_figure(state: AppState):
    """Initialize the figure to show sBayes data on the dashboard."""
    fig = px.scatter_geo(
        state.object_data,
        lat="y", lon="x",
        hover_data=["name", "family"],
        projection="natural earth",
        size_max=0.1,
    )
    fig.update_traces(marker=dict(size=4, color=COLOR_0))
    fig.update_layout(
        height=600,
        margin=dict(l=0, r=0, b=0, t=0, pad=0),
        geo=dict(
            lonaxis=dict(
                showgrid=True,
                gridwidth=0.5,
                range=[*min_and_max_with_padding(state.locations[:, 0])],
                dtick=5,
            ),
            lataxis=dict(
                showgrid=True,
                gridwidth=0.5,
                range=[*min_and_max_with_padding(state.locations[:, 1])],
                dtick=5,
            ),
        ),
    )
    return fig


def plot_trace(state: AppState):
    interval = 2
    sizes_np = np.sum(state.clusters[:, ::interval, :], axis=2)
    sizes_df = pd.DataFrame(
        [(i, interval*j, s) for (i, j), s in np.ndenumerate(sizes_np)],
        columns=["cluster", "sample", "size"]
    )
    fig = px.line(sizes_df, x="sample", y="size", color="cluster",
                  color_discrete_sequence=state.cluster_colors)
    fig.update_traces(line={'width': 1.0})
    fig.update_layout(
        height=160,
        margin=dict(l=20, r=20, t=0, b=10),
        **blank_layout,
    )

    return fig


def plot_summary_map(state: AppState, sample_range: list[int], posterior_threshold: float = 0.5):
    """Plot a summary map where an object is assigned to a cluster if its posterior
    frequency is above `posterior_threshold`."""
    i_start, i_end = sample_range
    # colors = np.full(state.objects.n_objects, "lightgrey", dtype=object)
    cluster_posterior = np.mean(state.clusters[:, i_start:i_end, :], axis=1)
    summary_clusters = (cluster_posterior > posterior_threshold)
    state.object_data.posterior_support = 1 - np.sum(cluster_posterior, axis=0)

    any_cluster = np.any(summary_clusters, axis=0)
    state.object_data.loc[any_cluster, "cluster"] = np.nonzero(summary_clusters.T)[1]
    state.object_data.loc[~any_cluster, "cluster"] = -1

    for i, c in enumerate(summary_clusters):
        state.lines[i].lon, state.lines[i].lat = cluster_to_graph(state.locations[c])
        # colors[c] = state.cluster_colors[i]
        state.lines[i].line.color = state.cluster_colors[i]
        state.scatter.customdata[c, 2] = i
        state.scatter.customdata[c, 3] = cluster_posterior[i, c]
    state.scatter.hovertemplate = "y=%{lat}<br>x=%{lon}<br>name=%{customdata[0]}<br>family=%{customdata[1]}<br>cluster=%{customdata[2]}<br>posterior_support=%{customdata[3]:.2f}"

    # state.scatter.marker.color = state.cluster_colors[state.object_data["cluster"].to_numpy()]
    # state.fig.update_traces(marker={
    #     "size": np.full(state.n_objects, 4),
    #     "color": state.cluster_colors[state.object_data["cluster"].to_numpy()],
    # })
    marker_style = {"color": state.cluster_colors[state.object_data["cluster"].to_numpy()]}

    if state.highlighted_cluster is None:
        marker_style["size"] = np.full(state.n_objects, 4)
    else:
        in_cluster = (state.object_data.cluster == state.highlighted_cluster)
        marker_style["size"] = np.where(in_cluster, 10, 4)

    state.fig.update_traces(marker=marker_style)
    return state.fig


def plot_sample_map(state: AppState, i_sample: int):
    """Plot a map of the clusters in a single posterior sample."""
    # colors = np.full(state.objects.n_objects, "lightgrey", dtype=object)

    any_cluster = np.any(state.clusters[:, i_sample], axis=0)
    state.object_data.loc[any_cluster, "cluster"] = np.nonzero(state.clusters[:, i_sample].T)[1]
    state.object_data.loc[~any_cluster, "cluster"] = -1
    state.scatter.customdata[~any_cluster, 2] = ""

    for i, c in enumerate(state.clusters[:, i_sample, :]):
        state.lines[i].lon, state.lines[i].lat = cluster_to_graph(state.locations[c])
        state.lines[i].line.color = state.cluster_colors[i]
        state.scatter.customdata[c, 2] = i

    state.scatter.hovertemplate = "y=%{lat}<br>x=%{lon}<br>name=%{customdata[0]}<br>family=%{customdata[1]}<br>cluster=%{customdata[2]}"

    # state.scatter.marker.color = state.cluster_colors[state.object_data["cluster"].to_numpy()]

    marker_style = {"color": state.cluster_colors[state.object_data["cluster"].to_numpy()]}

    if state.highlighted_cluster is None:
        marker_style["size"] = np.full(state.n_objects, 4)
    else:
        in_cluster = (state.object_data.cluster == state.highlighted_cluster)
        marker_style["size"] = np.where(in_cluster, 10, 4)

    state.fig.update_traces(marker=marker_style)
    return state.fig


def cluster_to_graph(locations):
    if len(locations) < 2:
        return [], []
    delaunay = compute_delaunay(locations)
    graph_connections = gabriel_graph_from_delaunay(delaunay, locations)

    x, y = [], []
    for i1, i2 in graph_connections:
        x += [locations[i1, 0], locations[i2, 0], None]
        y += [locations[i1, 1], locations[i2, 1], None]
    return x, y

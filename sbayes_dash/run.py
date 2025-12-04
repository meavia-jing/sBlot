from __future__ import annotations
import os;os.environ['USE_PYGEOS'] = '0'  # Fix for Pandas deprecation warning

from dash import dcc, no_update
from dash.html import Figure

from io import StringIO
import base64
from pathlib import Path

# from jupyter_dash import JupyterDash
# from dash import Input, Output, State
from dash import html
from dash_extensions.enrich import Output, DashProxy, Input, MultiplexerTransform, State
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from sbayes_dash.app_state import AppState
from sbayes_dash import dash_components as components
from sbayes_dash.load_data import Confounder, Objects
from sbayes_dash.util import read_data_csv, reproject_locations, COLOR_0, COLOR_1, activate_verbose_warnings
from sbayes_dash.plotting import plot_summary_map, plot_sample_map, create_data_figure


def find_biggest_angle_gap(degrees: NDArray[float]) -> float:
    degrees = np.sort(degrees)
    degrees = np.append(degrees, degrees[0] + 360)
    i = np.argmax(np.diff(degrees))
    return (degrees[i + 1] + degrees[i]) / 2


def parse_content(content: str) -> str:
    if isinstance(content, str):
        _, content = content.split(',')
    content = str(base64.b64decode(content))[2:-1]
    return content.replace(r"\t", "\t").replace(r"\n", "\n")


def encode_content(content: bytes) -> bytes:
    return base64.b64encode(content)


def parse_clusters_samples(clusters_samples: str) -> NDArray[bool]:  # shape: (n_clusters, n_samples, n_sites)
    samples_list = [
        [list(c) for c in line.split('\t')]
        for line in clusters_samples.split("\n")
        if line.strip()
    ]
    return np.array(samples_list, dtype=int).astype(bool).transpose((1, 0, 2))


# Initialized app
app = DashProxy(prevent_initial_callbacks='initial_duplicate', transforms=[MultiplexerTransform()], suppress_callback_exceptions=True)
# app = DashProxy(prevent_initial_callbacks=False, transforms=[MultiplexerTransform()], suppress_callback_exceptions=True)
# app = JupyterDash(__name__, suppress_callback_exceptions=True)
server = app.server
state = AppState()



@app.callback(
    Output('dashboard', 'children'),
    Output('upload-clusters', 'disabled'),
    Output('upload-data', 'children'),
    Input('upload-data', 'contents'),
    Input('upload-data', 'filename'),
    prevent_initial_call=True
)
def update_data(content, filename):
    if content is None:
        if state.data is None:
            return
        else:
            return components.build_tabs(state), False, html.Div([state.data_filename])

    # Load data
    data_str = parse_content(content)
    data_file = StringIO(data_str)
    state.data_filename = filename
    state.data = data = read_data_csv(data_file)
    state.objects = objects = Objects.from_dataframe(data)
    state.families = families = Confounder.from_dataframe(data, confounder_name="family")
    state.locations = locations = reproject_locations(objects.locations, state.data_crs, "epsg:4326")
    cut_longitude = find_biggest_angle_gap(locations[:, 0])
    locations[:, 0] = (locations[:, 0] - cut_longitude) % 360 + cut_longitude
    n_objects = len(locations)

    family_names = np.array(families.group_names + [""])
    family_ids = []
    for i, lang in enumerate(objects.names):
        i_fam = np.flatnonzero(families.group_assignment[:, i])
        i_fam = i_fam[0] if len(i_fam) > 0 else families.n_groups
        family_ids.append(i_fam)
    family_ids = np.array(family_ids)

    state.object_data = pd.DataFrame({
        "x": locations[:, 0],
        "y": locations[:, 1],
        "name": objects.names,
        "family": family_names[family_ids],
        "cluster": np.zeros(n_objects, dtype=int),
        "posterior_support": np.zeros(n_objects),
    })

    # Figure for the data plot:
    state.data_fig = create_data_figure(state)

    return components.build_tabs(state), False, html.Div([filename])


@app.callback(
    Output('results-tab', 'children'),
    Output('results-tab', 'disabled'),
    Output('tabs', 'value'),
    Output('upload-clusters', 'children'),
    Input('upload-clusters', 'contents'),
    Input('upload-clusters', 'filename'),
    prevent_initial_call=True
)
def update_clusters(content, filename):
    if content is None:
        return

    state.clusters_path = Path(filename)
    clusters_str = parse_content(content)
    state.clusters = parse_clusters_samples(clusters_str)

    results_components = components.build_results_components(state)
    return results_components, False, "results-tab", html.Div([filename])


@app.callback(
    Output("data_map", "figure"),
    Input("family-sizes", "hoverData"),
    prevent_initial_call=True
)
def hover_family(hover_data: dict):
    family = hover_data["points"][0]["x"]
    in_family = (state.object_data.family == family)
    state.data_fig.update_traces(marker={
        "color": np.where(in_family, COLOR_1, COLOR_0)
    })
    return state.data_fig


@app.callback(
    Output('data_map', 'figure'),
    Input('feature-selector', 'value'),
    prevent_initial_call=True
)
def select_feature(feature: str):
    # Which states can this feature take
    states = state.data[feature].unique()
    states = states[~pd.isna(states)]

    # Create color palette for these states
    color_seq = state.get_cluster_colors(len(states))
    color_map = {s: color_seq[i] for i, s in enumerate(states)}
    color_map[np.nan] = COLOR_0

    # Color the map accordingly
    colors = [color_map[x] for x in state.data[feature]]
    state.data_fig.update_traces(marker={"color": colors})
    return state.data_fig


@app.callback(
    Output("map", "figure"),
    Input("i_sample", "value"),
    prevent_initial_call=True
)
def update_sample_map(i_sample: int):
    if state.clusters is None:
        return None

    state.i_sample = i_sample
    return plot_sample_map(state, i_sample)


@app.callback(
    Output("map", "figure"),
    Output("i_sample", "value"),
    Input("trace", "hoverData"),
    State("summarize_switch", "on"),
    prevent_initial_call=True
)
def hover_tracer(hover_data: dict, summarize: bool):
    cluster = hover_data["points"][0]["curveNumber"]
    state.highlighted_cluster = cluster
    if summarize:
        in_cluster = (state.object_data.cluster == cluster)
        state.fig.update_traces(marker={"size": np.where(in_cluster, 10, 5)})
        return state.fig, no_update
    else:
        sample = hover_data["points"][0]["x"]
        return state.fig, sample


@app.callback(
    Output("map", "figure"),
    Input("sample_range", "value"),
    prevent_initial_call=True
)
def update_summary_map(sample_range: list[int]):
    if state.clusters is None:
        return None

    state.i_start, state.i_end = sample_range
    return plot_summary_map(state, sample_range)


@app.callback(
    Output("map", "figure"),
    Output("slider_div", "style"),
    Output("range_slider_div", "style"),
    Input("summarize_switch", "on"),
    prevent_initial_call=True
)
def switch_summarization(summarize: bool) -> (Figure, dcc.Slider):
    state.highlighted_cluster = None

    sample_slider_style = {"width": "90%", "display": "none"}
    range_slider_style = {"width": "90%", "display": "none"}
    if summarize:
        map_figure = plot_summary_map(state, [0, state.n_samples])
        range_slider_style["display"] = "inline-block"

    else:
        map_figure = plot_sample_map(state, state.i_sample)
        sample_slider_style["display"] = "inline-block"

    return map_figure, sample_slider_style, range_slider_style


@app.callback(
    Output("download-map", "data"),
    Input("download-map-button", "n_clicks"),
    prevent_initial_call=True
)
def download_figure(n_clicks):
    map_html = state.serialize_results_map(filename="sBayes_map.html")
    return map_html


def main(port=8050, crs="epsg:4326", data_path: Path = None):
    state.data_crs = crs

    if data_path:
        data_path = Path(data_path)
        with open(data_path, 'rb') as data_str:
            data_encoded = encode_content(data_str.read())
            update_data(data_encoded, data_path.name)

    # Set up the layout
    app.layout = components.get_base_layout()
    app.run(debug=True, port=port)


def cli():
    import argparse
    parser = argparse.ArgumentParser(description="Interactive visualisation of sBayes results.")
    parser.add_argument("-p", "--port", type=int, nargs="?", default=8050,
                        help="The port  used to serve the application.")
    parser.add_argument("-c", "--crs", type=str, nargs="?", default="epsg:4326",
                        help="The coordinate reference system (CRS) of the provided coordinates in the data file.")
    parser.add_argument("-d", "--data", type=Path, nargs="?",
                        help="Path to the data file")
    cli_args = parser.parse_args()

    if __debug__:
        activate_verbose_warnings()

    main(port=cli_args.port, crs=cli_args.crs, data_path=cli_args.data)

    # Data CRS for South America case study
    # "+proj=eqdc +lat_0=-32 +lon_0=-60 +lat_1=-5 +lat_2=-42 +x_0=0 +y_0=0 +ellps=aust_SA +units=m +no_defs "


if __name__ == '__main__':
    cli()

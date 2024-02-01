import dash
import pandas as pd
from dash import html, dcc
import dash_daq as daq
import plotly.express as px

from sbayes_dash.app_state import AppState
from sbayes_dash.plotting import initialize_results_map, plot_sample_map, plot_summary_map, plot_trace
from sbayes_dash.util import COLOR_0, COLOR_1


upload_box_style = {
    "width": "98%",
    "height": "40px",
    "lineHeight": "40px",
    "borderWidth": "1px",
    "borderStyle": "dashed",
    "borderRadius": "6px",
    "textAlign": "center",
    "margin": "4px",
    "font-variant": "small-caps",
    "fontSize": "11pt",
}

tabs_styles = {
    "height": "36px",
    "margin-top": "8px",
    "margin-bottom": "8px",
}
tab_style = {
    "padding": "8px",
    "borderBottom": "1px solid #d6d6d6",
    "fontSize": "11pt",
}
tab_selected_style = {
    "padding": "8px",
    "fontWeight": "bold",
    "fontSize": "11pt",
}
all_tab_styles = {
    "style": tab_style | {"color": "rgb(70,70,70)"},
    "selected_style": tab_selected_style,
    "disabled_style": tab_style,
}

def get_base_layout() -> html.Div:
    return html.Div(
        children=[
            # html.Img(src='assets/sbayes_logo.png', style={"width": "10%"}),
            html.Div([dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'drag and drop or select the ', html.B('data file'), ' (.csv)'
                ]),
                style=upload_box_style,
                disabled=False,
                style_disabled={"opacity": 0.3},
            )], style={"width": "50%", "display": "inline-block"}),
            html.Div([dcc.Upload(
                id='upload-clusters',
                children=html.Div([
                    'drag and drop or select the ', html.B('clusters file'), ' (clusters_*.txt)'
                ]),
                style=upload_box_style,
                disabled=True,
                style_disabled={"opacity": 0.3},
            )], style={"width": "50%", "display": "inline-block"}),
            html.Div(id='dashboard', children=[])
        ], style={"font-family": "sans-serif"}
    )


def get_sample_slider(state: AppState) -> dcc.Slider:
    return dcc.Slider(
            id="i_sample", value=1, step=1, min=0, max=state.n_samples-1,
            marks={i: str(i) for i in range(0, state.n_samples, max(1, state.n_samples//10))},
    )


def get_summary_range_slider(state: AppState) -> dcc.RangeSlider:
    return dcc.RangeSlider(
        id="sample_range", value=[0, state.n_samples], step=1, min=0, max=state.n_samples,
        marks={i: str(i) for i in range(0, state.n_samples, max(1, state.n_samples//10))},
    )


def build_tabs(state: AppState) -> dcc.Tabs:
    return dcc.Tabs(id="tabs", value='data-tab', style=tabs_styles, children=[
        dcc.Tab(value="data-tab", id="data-tab", label="Data", **all_tab_styles,
                children=[build_data_component(state)]),
        dcc.Tab(value="results-tab", id="results-tab", label="Results", **all_tab_styles,
                children=[], disabled=True),
    ])


def column_style(width: int):
    return {"width": f"{width}%", "display": "inline-block", "margin-left": 10, "verticalAlign": "middle"}


def get_family_sizes(object_data: pd.DataFrame) -> pd.DataFrame:
    family_sizes = object_data \
        .groupby("family") \
        .size() \
        .reset_index(name="size") \
        .sort_values("size", ascending=False)

    family_sizes["is_isolate"] = family_sizes.family == ""
    family_sizes.loc[family_sizes.is_isolate, "family"] = "Isolates and singleton families"
    return family_sizes


def build_data_component(state: AppState) -> html.Div:
    family_sizes = get_family_sizes(state.object_data)
    family_size_graph = px.bar(family_sizes, x='family', y='size', orientation='v',
                               color="is_isolate", color_discrete_sequence=[COLOR_0, COLOR_1])
    family_size_graph.update_layout(showlegend=False, font={"size": 10})

    data_component = html.Div([
        dcc.Graph(id="data_map", figure=state.data_fig),
        # html.Div(dcc.RadioItems(['Family sizes', 'Features'], 'Family sizes', id='data-plot-selector'),
        #          style=column_style(13)),
        # html.Div(dcc.Graph(id='family-sizes', figure=family_size_graph),
        #          style=column_style(83), id="data-plot"),
        dcc.Tabs(id="data-subtabs", value='family-tab', style=tabs_styles, children=[
            dcc.Tab(value="family-tab", id="family-tab", label="Families", **all_tab_styles,
                    children=dcc.Graph(id='family-sizes', figure=family_size_graph)),
            dcc.Tab(value="feature-tab", id="feature-tab", label="Features", **all_tab_styles,
                    children=dcc.RadioItems(state.data.columns[5:], id='feature-selector', style={"font-size": 12})),
        ])
    ])

    return data_component


def build_results_components(state: AppState, start_with_summarize: bool = True) -> html.Div:
    initialize_results_map(state)

    sample_slider = get_sample_slider(state)
    range_slider = get_summary_range_slider(state)
    trace_fig = plot_trace(state)

    if start_with_summarize:
        results_map = plot_summary_map(state, range_slider.value)
    else:
        results_map = plot_sample_map(state, sample_slider.value)

    return html.Div([
        html.Div([
                html.P(id="sample", children="Sample number", style={"font-size": 14, "text-indent": "10px"}),
                sample_slider,
            ],
            style={"width": "90%", "display": "none" if start_with_summarize else "inline-block"},
            id="slider_div",
        ),
        html.Div([
                html.P(id="sample-range", children="Sample range", style={"font-size": 14, "text-indent": "10px"}),
                range_slider,
            ],
            style={"width": "90%", "display": "inline-block" if start_with_summarize else "none"},
            id="range_slider_div",
        ),
        html.Div([
            daq.BooleanSwitch(id="summarize_switch", label={"label": "Summarize samples", "style": {"font-size": 14}},
                              labelPosition="top", on=start_with_summarize)
        ], style={"width": "9%", "display": "inline-block"}),
        dcc.Graph(id="trace", figure=trace_fig, style={"width": "93vw", "height": "160px"}),
        dcc.Graph(id="map", figure=results_map, style={"width": "95vw", "margin-left": "1.8vw"}),
        html.Div([
            html.Button("Download map as HTML", id="download-map-button"),
            dcc.Download(id="download-map"),

        ]),
    ], style={'textAlign': 'center'})

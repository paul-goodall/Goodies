from pathlib import Path
import dotenv
import os
import sys
import numpy as np

import pandas as pd
import boto3
import s3fs
s3 = s3fs.S3FileSystem(anon=False)

import dash
import dash_bootstrap_components as dbc
from dash import Dash, html, dcc, callback, Output, Input, clientside_callback, ctx
import plotly.express as px



# =========================================================



external_scripts = ['https://ajax.googleapis.com/ajax/libs/jquery/3.7.1/jquery.min.js']
app = dash.Dash(
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    external_scripts=external_scripts,
    # only for aws sagemaker:
    #requests_pathname_prefix='/jupyter/default/proxy/8050/'
)



plot_menu1_options = html.Div(
    [
        dbc.Label("Enter a value for sample_id:"),
        dbc.Input(id="input_id", type="text", value=chosen_id),
        html.Hr(),
        dbc.Label("Choose a dataset:"),
        dbc.Checklist(
            options=dataset_opts,
            value=dataset_opts,
            id='choose_dataset',
        ),
        html.Hr(),
        dbc.Label("Image opacity:"),
        dcc.Slider(0.0, 1.0, 0.1,
               value=1.0,
               id='image_opacity_slider'
        ),
        html.Hr(),
        dbc.Label("Image display: (currently broken)"),
        dbc.RadioItems(
            options=[
                {"label": "Colour", "value": 1},
                {"label": "Greyscale", "value": 2},
            ],
            value=1,
            id="image_palette",
        ),
        html.Hr(),
        dbc.Input(id="info", type="text", value="nada", style={'visibility':'hidden'}),

    ]
)

plot_menu2_options = html.Div(
    [
        dbc.Label("Choose a layer:"),
        dbc.Checklist(
            options=layer_opts,
            value=layer_opts,
            id='choose_layer',
        ),
        html.Hr(),
    ]
)



plot_menu_panel_lhs = html.Div(
    [
        dbc.Form([plot_menu1_options]),
    ]
)


plot_menu_panel_rhs = html.Div(
    [
        dbc.Form([plot_menu2_options]),
    ]
)


app.layout = dbc.Container(
    [
        html.H1("Generic Python Dashboard"),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(plot_menu_panel_lhs, md=2),
                dbc.Col(dcc.Graph(id="geo_layers_graph"), md=8),
                dbc.Col(plot_menu_panel_rhs, md=2),
            ],
            align="center",
        ),
    ],
    fluid=True,
)



# ==============================================
# Callbacks



@app.callback(
    [
        Output(component_id='geo_layers_graph', component_property='figure'),
        Output(component_id='choose_dataset', component_property='options'),
        Output(component_id='choose_dataset', component_property='value'),
        Output(component_id='choose_layer', component_property='options'),
        Output(component_id='choose_layer', component_property='value'),
    ],
    [
        Input(component_id='input_id', component_property='value'),
        Input(component_id='choose_dataset', component_property='options'),
        Input(component_id='choose_dataset', component_property='value'),
        Input(component_id='choose_layer', component_property='options'),
        Input(component_id='choose_layer', component_property='value'),
    ],
)
def update_graph(selected_id,options_dataset,selected_dataset,options_layer,selected_layer):
    trigger_id = ctx.triggered_id
    selected_id = int(selected_id)

    global image_obj
    global gdf_combined

    reload_data = False
    if trigger_id == 'input_id' or trigger_id is None:
        reload_data = True

    #reload_data = True
    if reload_data:
        fn = f'{s3_base}/{selected_id:09}.pkl'
        #print(fn)
        dset = pg.rpkl(fn)
        image_obj = dset['image_obj']
        gdf_combined = pg.create_plot_data(dset)
        dataset_opts = list(gdf_combined.dataset.unique())
        layer_opts   = list(gdf_combined.layer_name.unique())
        options_dataset = list(gdf_combined.dataset.unique())
        selected_dataset = options_dataset
        options_layer = list(gdf_combined.layer_name.unique())
        selected_layer = options_layer

    ii  = gdf_combined.dataset.isin(selected_dataset)
    dff = gdf_combined[ii]
    ii  = dff.layer_name.isin(selected_layer)
    dff = dff[ii]

    #options_layer = list(dff.layer_name.unique())

    fig = pg.dash_tall_gdf(dff, image_obj, pairs_df, labelcol='geom_description', opacitycol='plot_opacity')
    fig.update_layout(
        margin={'l': 50, 'b': 50, 't': 50, 'r': 50},
        hovermode='closest',
        height=900
    )
    return fig, options_dataset,selected_dataset,options_layer,selected_layer




clientside_callback(
    """
    function(pallete_val,opacity_val) {

        var elementExists = !!document.getElementsByClassName("js-plotly-plot");
        var g_im = null;

        if (elementExists) {
            g_im = $('.js-plotly-plot image');

            if (pallete_val == 1){
                g_im[0].style.filter = "grayscale(0)";
            }
            if (pallete_val == 2){
                g_im[0].style.filter = "grayscale(100%)";
            }

            g_im[0].style.opacity = opacity_val;
        }
        return pallete_val;
    }
    """,
    Output('info', 'value'),
    Input(component_id='image_palette', component_property='value'),
    Input(component_id='image_opacity_slider', component_property='value'),
    prevent_initial_call=True,
)




if __name__ == "__main__":
    app.run_server(debug=True, port=8050, host='0.0.0.0')

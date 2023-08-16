import dash
from dash import Dash, dcc, html, Input, Output, callback, State
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import base64
import dash_bootstrap_components as dbc
import plotly.express as px

# Load numpy arrays
embeddings = np.load('embeddings.npy')
labels = np.load('labels.npy')
filenames = np.load('filenames.npy')

# Create a DataFrame from the numpy arrays
df = pd.DataFrame({
    'UMAP_1': embeddings[:, 0],
    'UMAP_2': embeddings[:, 1],
    'label': labels,
    'filename': filenames,
})

# Generate the Inferno colormap sequence based on the number of unique labels
num_labels = len(df['label'].unique())
color_idx = np.linspace(0, len(px.colors.sequential.Viridis)-1, num_labels).astype(int)
colors = [px.colors.sequential.Viridis[i] for i in color_idx]

classes_map = {
    'Abstract_Expressionism': 0,
    'Action_painting': 1,
    'Analytical_Cubism': 2,
    'Art_Nouveau_Modern': 3,
    'Baroque': 4,
    'Color_Field_Painting': 5,
    'Contemporary_Realism': 6,
    'Cubism': 7,
    'Early_Renaissance': 8,
    'Expressionism': 9,
    'Fauvism': 10,
    'High_Renaissance': 11,
    'Impressionism': 12,
    'Mannerism_Late_Renaissance': 13,
    'Minimalism': 14,
    'Naive_Art_Primitivism': 15,
    'New_Realism': 16,
    'Northern_Renaissance': 17,
    'Pointillism': 18,
    'Pop_Art': 19,
    'Post_Impressionism': 20,
    'Realism': 21,
    'Rococo': 22,
    'Romanticism': 23,
    'Symbolism': 24,
    'Synthetic_Cubism': 25,
    'Ukiyo_e': 26
}

# Reverse the dictionary so that the indices are keys
classes_map = {v: k for k, v in classes_map.items()}

# Create a new column for the actual class names
df['class_name'] = df['label'].map(classes_map)

# Create the Dash app
external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css", dbc.themes.LUX]
app = Dash(__name__, external_stylesheets=external_stylesheets)
app.title = 'Art Embedder - Aniket Pant'

app.layout = html.Div([
    html.Div([
        dbc.Container(
            [
                html.H1("WikiArt Embedder", className="display-3"),
                html.P(
                    "Using a ConvVAE + UMAP to embed art in low dimensional spaces. 20,000 images are embedded in RGB space.",
                    className="lead",
                ),
                html.P(
                    "Author: Aniket Pant", style  = {"font-size": "14px"}
                ),
            ]
    ),], style = {'margin-top': "4rem"}),
    html.Div([
        html.Div(
            [
                html.Div([
                    dcc.Graph(id='umap-plot', config={'displayModeBar': False}, style = {"height": "70vh"}),
                ]),
                html.Div([
                        html.Img(id='image-display', src='', height='auto', width='300px'),
                        html.P(id='image-filename', children='No image selected'),
                ], style = {"display": "flex", "flex-direction": "column", "align-items": "center", "margin-right": "1rem"}),
            ], style = {"display": "flex", "flex-direction": "row", 'align-items': 'center', 'justify-content': 'space-around'}
        ),
    ]),
], style={'fontFamily': 'Helvetica', 'padding': '20'})

# function to make UMAP plot
@app.callback(
    Output('umap-plot', 'figure'),
    [Input('umap-plot', 'id')]
)
def update_graph(id):

    traces = []
    for idx, (label, group) in enumerate(df.groupby('label')):
        trace = go.Scatter(
            x=group['UMAP_1'],
            y=group['UMAP_2'],
            mode='markers',
             marker=dict(
                    size=3,
                    color=colors[idx],
                    showscale=False,
                    opacity=0.8,
                ),
            hovertext=[x.split("/")[-1] for x in group['filename']],
            customdata=group['filename'],
            name=classes_map[label]  # Label by the actual class name
        )
        traces.append(trace)

    figure = go.Figure(
        data=traces,
        layout=go.Layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title = "UMAP1"),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title = "UMAP2"),
            margin=dict(l=40, r=40, b=80, t=40),
            hovermode='closest',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(
                x=0.5,
                y=-0.2,  # Adjusting y to position the legend below the plot
                xanchor='center',  # Ensures the center of the legend is at x=0.5
                orientation='h',  # Keeps the legend horizontal
                font=dict(
                    size=10,
                    color="black"
                )
            )
        )
    )
    return figure

# Callback to update the image
@app.callback(
    Output('image-display', 'src'),
    [Input('umap-plot', 'clickData')],
    [State('image-display', 'src')]
)
def update_image(clickData, src):
    if clickData is None:
        return src
    else:
        # Open and base64 encode the image file
        filename = clickData['points'][0]['customdata']
        encoded_image = base64.b64encode(open(filename, 'rb').read()).decode('ascii')
        src = "data:image/png;base64,{}".format(encoded_image)
        return src

@app.callback(
    Output('image-filename', 'children'),
    [Input('umap-plot', 'clickData')]
)
def update_filename(clickData):
    if clickData is None:
        return 'No image selected'
    else:
        filename = clickData['points'][0]['customdata'].split("/")[-1]
        return 'Selected image: {}'.format(filename)

if __name__ == '__main__':
    app.run(debug=False, host= '0.0.0.0', port = 8050)

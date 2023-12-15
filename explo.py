
import webbrowser
import dash
from dash.exceptions import PreventUpdate
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import json

app = dash.Dash(__name__)
df = pd.read_csv('Data/doc_centroid.csv')
df["Cluster"] = df["Cluster"].astype(str)
# Specify the order of colors for each category (Cluster)
cluster_colors = px.colors.qualitative.Set3
category_order = sorted(df['Cluster'].unique())  # Assuming 'Cluster' is your category column

fig = px.scatter(df, x="X", y="Y", color="Cluster", custom_data=["DOI"],
                 color_discrete_map=dict(zip(category_order, cluster_colors)),
                 opacity=0.80 ,
                 category_orders={"Cluster": category_order},
                 hover_data={"DOI": True, "title": True})

fig.update_layout(clickmode='event+select',
                  height=1200,            # Set the height of the plot
                  width=1200)
fig.update_traces(marker_size=20)


app.layout = html.Div(
   [
      dcc.Graph(
         id="graph_interaction",
         figure=fig,
      ),
      html.Pre(id='data')
   ]
)

@app.callback(
   Output('data', 'children'),
   Input('graph_interaction', 'clickData'))
def open_url(clickData):
   if clickData:
       webbrowser.open(clickData["points"][0]["customdata"][0])
   else:
      raise PreventUpdate
      # return json.dumps(clickData, indent=2)
      
if __name__ == '__main__':
    app.run_server(debug=True)
import os
import math
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from dotenv import load_dotenv
from pylab import *
from plotly.offline import (init_notebook_mode, iplot, plot)


mpl.rcParams["front.sans-serif"] = ["SimHei"]
_path = os.path.dirname(__file__)
load_dotenv(dotenv_path=os.path.join(_path, "python_adv.env"))
mapbox_access_token = os.getenv("MAPBOX_TOKEN")

def poi_maply(df_master, df_slave, df_noise):
    # trace
    list_trace = []

    trace_master = go.Scattermapbox(
        lon=df_master.lon,
        lat=df_master.lat,
        text=df_master.label,
        mode="makers",
        marker=dict(size=5, color="red", opacity=0.8)
    )
    list_trace.append(trace_master)

    trace_location = go.Scattermapbox(
        lon=[df_master.lon.mean()],
        lat=[df_master.lat.mean()],
        text=[...],
        mode="makers",
        marker=dict(size=10, color="red", opacity=1, symbol="star")
    )
    list_trace.append(trace_location)

    trace_noise = go.Scattermapbox(
        lon=df_noise.lon,
        lat=df_noise.lat,
        text=df_noise.label,
        mode="makers",
        marker=dict(size=3, color="black", opacity=0.8)
    )
    list_trace.append(trace_noise) 

    for label in set(df_slave.label):
        trace_slave = go.Scattermapbox(
            lon=df_slave[df_slave.label == label, "lon"],
            lat=df_slave[df_slave.label == label, "lat"],
            text=df_slave[df_slave.label == label, "label"],
            mode="markers",
            marker=dict(size=5, color=int(label), opacity=0.8)
        )
        list_trace.append(trace_slave)
    
    lon_avg = df_master.lon.mean()
    lat_avg = df_master.lat.mean()

    # plot
    layout = go.Layout(
        title="Your Title",
        xaxis=dict(title="Longitude"),
        yaxis=dict(title="Latitude"),
        mapbox=dict(accesstoken=mapbox_access_token,
                    center=dict(lon=lon_avg, lat=lat_avg),
                    zoom=5),
        width=1000,
        height=1000
    )
    fig = go.Figure(data=list_trace, layout=layout)
    plot(fig, filename="Your File Name.html")
    return
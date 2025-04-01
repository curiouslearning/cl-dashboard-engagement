import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Linear histogram of total engagement time (in minutes), with optional percent mode


def engagement_histogram_below_threshhold(df, percentile, lower_limit=60, key="key", as_percent=False):

    # Filter out very short sessions
    df = df[df["total_time_seconds"] > lower_limit].copy()

    # Trim to users within the given percentile threshold
    threshold_df = df["total_time_seconds"].quantile(percentile)
    df_trimmed = df[df["total_time_seconds"] <= threshold_df].copy()

    # Convert seconds to minutes for x-axis readability
    df_trimmed["total_time_minutes"] = df_trimmed["total_time_seconds"] / 60

    histnorm = "percent" if as_percent else None

    # Plot histogram
    fig = px.histogram(
        df_trimmed,
        x="total_time_minutes",
        nbins=50,
        histnorm=histnorm,
        title=f"Engagement Time (>{lower_limit}s, ≤ {int(percentile * 100)}th Percentile)",
        labels={"total_time_minutes": "Total Time (minutes)"}
    )

    # Update y-axis label based on percent toggle
    fig.update_layout(bargap=0.1)
    fig.update_yaxes(title_text="% of Users" if as_percent else "User Count")

    # Custom hover formatting
    if as_percent:
        fig.update_traces(
            hovertemplate="%{y:.2f}% of users<br>%{x:.2f} min<extra></extra>")
    else:
        fig.update_traces(
            hovertemplate="%{y} users<br>%{x:.2f} min<extra></extra>")

    # Display in Streamlit
    st.plotly_chart(fig, use_container_width=True, key=f"{key}-2")


# Log-scale histogram of total engagement time with manual binning and custom hover text
def engagement_histogram_log(df, key, percentile=1.00, nbins=30, lower_limit=60, as_percent=False):
    # Filter out very short sessions
    df_log = df[df["total_time_seconds"] > lower_limit].copy()

    # Trim to users within percentile cap
    upper_limit = df_log["total_time_seconds"].quantile(percentile)
    df_log = df_log[df_log["total_time_seconds"] <= upper_limit]

    # Apply log10 transform to total time
    df_log["log_total_time"] = np.log10(df_log["total_time_seconds"])

    # Define log bin edges
    min_log = np.floor(df_log["log_total_time"].min())
    max_log = np.ceil(df_log["log_total_time"].max())
    bin_edges = np.linspace(min_log, max_log, nbins + 1)

    # Bin each row into a log interval
    df_log["log_bin"] = pd.cut(df_log["log_total_time"], bins=bin_edges)

    # Count users per bin
    bin_counts = df_log.groupby("log_bin").size().reset_index(name="count")
    bin_counts["bin_left"] = bin_counts["log_bin"].apply(lambda x: x.left)
    bin_counts["bin_right"] = bin_counts["log_bin"].apply(lambda x: x.right)

    # Convert bin edges back to seconds
    bin_counts["sec_left"] = (
        10 ** bin_counts["bin_left"].astype(float)).round(2)
    bin_counts["sec_right"] = (
        10 ** bin_counts["bin_right"].astype(float)).round(2)

    # Format hover labels
    if as_percent:
        total_count = bin_counts["count"].sum()
        bin_counts["percent"] = 100 * bin_counts["count"] / total_count
        bin_counts["hover"] = bin_counts.apply(
            lambda row: f"{row['percent']:.2f}% of users<br>{row['sec_left']}s – {row['sec_right']}s", axis=1)
        y_vals = bin_counts["percent"]
    else:
        bin_counts["hover"] = bin_counts.apply(
            lambda row: f"{row['count']} users<br>{row['sec_left']}s – {row['sec_right']}s", axis=1)
        y_vals = bin_counts["count"]

    # Create bar chart manually for full control
    fig = go.Figure(data=[go.Bar(
        x=bin_counts["bin_left"],
        y=y_vals,
        hovertext=bin_counts["hover"],
        hovertemplate="%{hovertext}<extra></extra>",
        width=np.diff(bin_edges)
    )])

    # Add log-scale tick labels (e.g., 10s, 1m, etc.)
    tickvals_raw = [1, 3, 10, 30, 100, 300, 1000, 3000, 10000]
    tickvals = [np.log10(val)
                for val in tickvals_raw if min_log <= np.log10(val) <= max_log]
    tick_seconds = [10 ** float(val) for val in tickvals]
    ticktext = [f"{int(s)}s" if s <
                60 else f"{int(s // 60)}m" for s in tick_seconds]

    fig.update_xaxes(
        tickvals=tickvals,
        ticktext=ticktext,
        range=[min_log, max_log],
        title_text="Total Time (seconds, log scale)"
    )

    # Update y-axis label
    fig.update_yaxes(title_text="% of Users" if as_percent else "User Count")
    fig.update_layout(
        title=f"Log10 Histogram of Engagement Time (≤ {int(percentile*100)}th Percentile)")

    # Display in Streamlit
    st.plotly_chart(fig, use_container_width=True, key=f"{key}-3")


# Caches a CSV version of the dataframe for downloads
@st.cache_data
def convert_for_download(df):
    return df.to_csv().encode("utf-8")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Caches a CSV version of the dataframe for downloads
@st.cache_data
def convert_for_download(df):
    return df.to_csv().encode("utf-8")

# Linear histogram of total engagement time (in minutes), with optional percent mode
def engagement_histogram(df,min_seconds=60, key="key", as_percent=False):

    # Filter out very short sessions
    df_trimmed = df[df["total_time_seconds"] > min_seconds].copy()

    # Convert seconds to minutes for x-axis readability
    df_trimmed["total_time_minutes"] = df_trimmed["total_time_seconds"] / 60

    histnorm = "percent" if as_percent else None

    # Plot histogram
    fig = px.histogram(
        df_trimmed,
        x="total_time_minutes",
        nbins=50,
        histnorm=histnorm,
        title=f"Engagement Time (>{min_seconds/60}m",
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
    st.plotly_chart(fig, use_container_width=True, key=key)


# Log-scale histogram of total engagement time with manual binning and custom hover text
def engagement_histogram_log(df, key, nbins=30, min_seconds=60, as_percent=False):
    # Filter out very short sessions
    df_log = df[df["total_time_seconds"] > min_seconds].copy()

    # Apply log10 transform to total time (in seconds)
    df_log["log_total_time"] = np.log10(df_log["total_time_seconds"])

    # Define log bin edges
    min_log = np.floor(df_log["log_total_time"].min())
    max_log = np.ceil(df_log["log_total_time"].max())
    bin_edges = np.linspace(min_log, max_log, nbins + 1)

    # Bin each row into a log interval
    df_log["log_bin"] = pd.cut(df_log["log_total_time"], bins=bin_edges)

    # Count users per bin
    bin_counts = df_log.groupby(
        "log_bin", observed=True).size().reset_index(name="count")

    # Extract float bin edges from Interval objects and cast explicitly to float
    bin_counts["bin_left"] = bin_counts["log_bin"].apply(
        lambda x: x.left).astype(float)
    bin_counts["bin_right"] = bin_counts["log_bin"].apply(
        lambda x: x.right).astype(float)

    # Convert bin edges from log10(seconds) → seconds → minutes
    bin_counts["sec_left"] = 10 ** bin_counts["bin_left"]
    bin_counts["sec_right"] = 10 ** bin_counts["bin_right"]
    bin_counts["min_left"] = (bin_counts["sec_left"] / 60).round(2)
    bin_counts["min_right"] = (bin_counts["sec_right"] / 60).round(2)

    # Format hover labels
    if as_percent:
        total_count = bin_counts["count"].sum()
        bin_counts["percent"] = 100 * bin_counts["count"] / total_count
        bin_counts["hover"] = bin_counts.apply(
            lambda row: f"{row['percent']:.2f}% of users<br>{row['min_left']} – {row['min_right']} min", axis=1
        )
        y_vals = bin_counts["percent"]
    else:
        bin_counts["hover"] = bin_counts.apply(
            lambda row: f"{row['count']} users<br>{row['min_left']} – {row['min_right']} min", axis=1
        )
        y_vals = bin_counts["count"]

    # Create bar chart
    fig = go.Figure(data=[go.Bar(
        x=bin_counts["bin_left"],
        y=y_vals,
        hovertext=bin_counts["hover"],
        hovertemplate="%{hovertext}<extra></extra>",
        width=np.diff(bin_edges)
    )])

    # Custom ticks on x-axis (convert seconds to minutes)
    tickvals_raw = [1, 3, 10, 30, 60, 120, 300, 600, 1800, 3600, 7200, 10000]
    tickvals = [np.log10(val)
                for val in tickvals_raw if min_log <= np.log10(val) <= max_log]
    tick_minutes = [
        val / 60 for val in tickvals_raw if min_log <= np.log10(val) <= max_log]
    ticktext = [f"{m:.0f}m" if m >=
                1 else f"{int(m * 60)}s" for m in tick_minutes]

    fig.update_xaxes(
        tickvals=tickvals,
        ticktext=ticktext,
        range=[min_log, max_log],
        title_text="Total Time (log scale, in minutes)"
    )

    fig.update_yaxes(title_text="% of Users" if as_percent else "User Count")
    fig.update_layout(
        title="Log10 Histogram of Engagement Time (Minutes)"
    )

    # Display in Streamlit
    st.plotly_chart(fig, use_container_width=True, key=key)


def engagement_scatter_plot(df, min_seconds=60, key="key"):
    df_trimmed = df[df["total_time_seconds"] > min_seconds].copy()
    df_trimmed["total_time_minutes"] = df_trimmed["total_time_seconds"] / 60

    fig = px.scatter(df_trimmed,
                     x="engagement_event_count",
                     y="total_time_minutes",
                     title="User Engagement: Events vs. Time",
                     color="country",
                     labels={
                         "engagement_event_count": "Engagement Event Count",
                         "total_time_minutes": "Total Time (min)"
                     },
                     hover_data=["country"])

    fig.update_traces(
        hovertemplate="Country: %{customdata[0]}<br>Events: %{x}<br>%{y:.2f} minutes")
    st.plotly_chart(fig, use_container_width=True, key=key)


def engagement_pareto_chart(df, min_seconds=60, key="key"):
    df_trimmed = df[df["total_time_seconds"] > min_seconds].copy()

    top_countries = df_trimmed["country"].value_counts()
    valid_countries = top_countries[top_countries >= 100].index
    df_trimmed = df_trimmed[df_trimmed["country"].isin(valid_countries)]

    df_trimmed["total_time_minutes"] = df_trimmed["total_time_seconds"] / 60

    df_sorted = df_trimmed.sort_values(
        by="total_time_minutes", ascending=False).reset_index(drop=True)
    df_sorted["cumulative_time"] = df_sorted["total_time_minutes"].cumsum()
    df_sorted["cumulative_percent"] = 100 * \
        df_sorted["cumulative_time"] / df_sorted["total_time_minutes"].sum()
    df_sorted["user_rank"] = range(1, len(df_sorted) + 1)
    df_sorted["user_percent"] = 100 * df_sorted["user_rank"] / len(df_sorted)

    fig = px.line(df_sorted, x="user_percent", y="cumulative_percent",
                  color="country",
                  title="Cumulative Engagement by Country (Pareto)",
                  labels={"user_percent": "% of Users", "cumulative_percent": "Cumulative % of Time (min)"})

    st.plotly_chart(fig, use_container_width=True, key=key)



def engagement_box_plot(df, key="key", min_seconds=60):
    # Filter users with very short sessions
    df_filtered = df[df["total_time_seconds"] > min_seconds].copy()

    # Add minutes column
    df_filtered["total_time_minutes"] = df_filtered["total_time_seconds"] / 60

    fig = px.violin(
        df_filtered,
        y="total_time_minutes",
        box=True,         # Show box plot inside the violin
        points="all",     # Show all user data points
        title="Violin Plot of Total Engagement Time (Minutes)",
        labels={"total_time_minutes": "Total Time (min)"}
    )

    fig.update_traces(hovertemplate="%{y:.2f} minutes")
    st.plotly_chart(fig, use_container_width=True, key=f"{key}-violin")


    # Total Time by Country (in minutes)
    fig = px.box(
        df_filtered,
        x="country",
        y="total_time_minutes",
        title="Engagement Time Distribution by Country (Minutes)",
        labels={
            "total_time_minutes": "Total Time (min)", "country": "Country"},
        points="all"
    )
    fig.update_traces(hovertemplate="Country: %{x}<br>%{y:.2f} minutes")
    st.plotly_chart(fig, use_container_width=True, key=f"{key}-3")
    
def most_engaged_users_chart(df,key="key"):   
    top_n = 20

    df_top = df.sort_values("total_time_seconds", ascending=False).head(top_n)
    
    df_top["total_time_minutes"] = df_top["total_time_seconds"] / 60


    # Then plot using the minutes column
    fig = px.bar(
        df_top,
        x="user_pseudo_id",
        y="total_time_minutes",
        title=f"Top {top_n} Most Engaged Users (by Time)",
        labels={
            "user_pseudo_id": "User ID",
            "total_time_minutes": "Total Time (min)"
        },
        text_auto=".2f"
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True, key=key)

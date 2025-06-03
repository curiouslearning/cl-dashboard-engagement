import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import ui_widgets as ui

# Caches a CSV version of the dataframe for downloads
@st.cache_data
def convert_for_download(df):
    return df.to_csv().encode("utf-8")


# Linear histogram of total engagement time (in minutes), with optional percent mode
def engagement_histogram(df, min_minutes=1, key="key", as_percent=False, percent_cutoff=0.5):

    # Toggle to determine percent base
    show_percent_of_all = st.toggle(
        "Show % of all users", value=False, key=f"{key}-toggle")

    # Step 1: Filter users
    df_trimmed = df[df["total_time_minutes"] > min_minutes].copy()

    # Step 2: Define bins starting from min threshold
    bin_start = min_minutes 
    bin_end = df_trimmed["total_time_minutes"].max() + 1
    bin_width = 11
    bin_edges = np.arange(bin_start, bin_end, bin_width)
    if len(bin_edges) < 2:
        bin_edges = np.array([bin_start, bin_start + bin_width])

    bin_labels = [
        f"{int(bin_edges[i])}–{int(bin_edges[i + 1])} min"
        for i in range(len(bin_edges) - 1)
    ]

    # Step 3: Assign bins and count
    df_trimmed["bin"] = pd.cut(
        df_trimmed["total_time_minutes"],
        bins=bin_edges,
        labels=bin_labels,
        include_lowest=True
    )

    bin_counts = df_trimmed["bin"].value_counts(sort=False).reset_index()
    bin_counts.columns = ["bin", "count"]
    bin_counts["bin"] = bin_counts["bin"].astype(str)

    # Step 4: Calculate values & hovertext
    if as_percent:
        if show_percent_of_all:
            total = df["cr_user_id"].nunique()
        else:
            total = df_trimmed["cr_user_id"].nunique()

        bin_counts["percent"] = 100 * bin_counts["count"] / total
        bin_counts = bin_counts[bin_counts["percent"] >= percent_cutoff]

        y_vals = bin_counts["percent"]
        hovertext = bin_counts.apply(
            lambda row: f"{row['percent']:.2f}% of users<br>{row['bin']}", axis=1)
    else:
        bin_counts = bin_counts[bin_counts["count"] >= 50]
        y_vals = bin_counts["count"]
        hovertext = bin_counts.apply(
            lambda row: f"{row['count']} users<br>{row['bin']}", axis=1)

    # Step 5: Plot
    fig = go.Figure(data=[
        go.Bar(
            x=bin_counts["bin"],
            y=y_vals,
            hovertext=hovertext,
            hovertemplate="%{hovertext}<extra></extra>"
        )
    ])

    fig.update_layout(
        title=f"Engagement Time (users > {min_minutes:.1f} min playing time)",
        xaxis_title="Total Time (minutes)",
        yaxis_title="% of Users" if as_percent else "User Count",
        bargap=0.1
    )

    st.plotly_chart(fig, use_container_width=True, key=f"{key}-{min_minutes}")


# Log-scale histogram of total engagement time with manual binning and custom hover text
def engagement_histogram_log(df, key, nbins=30, min_minutes=1, as_percent=False):
    # Filter out very short sessions
    df_log = df[df["total_time_minutes"] > min_minutes].copy()


    # Apply log10 transform to total time (in seconds)
    # Filter rows based on total time in minutes
    df_log = df[df["total_time_minutes"] > min_minutes].copy()

    # Apply log10 transform to total time (in minutes)
    df_log["log_total_time"] = np.log10(df_log["total_time_minutes"])

    # Define log bin edges
    min_log = np.floor(df_log["log_total_time"].min())
    max_log = np.ceil(df_log["log_total_time"].max())
    bin_edges = np.linspace(min_log, max_log, nbins + 1)

    # Bin each row into a log interval
    df_log["log_bin"] = pd.cut(df_log["log_total_time"], bins=bin_edges)

    # Count users per bin
    bin_counts = df_log.groupby(
        "log_bin", observed=True).size().reset_index(name="count")

    # Extract float bin edges from Interval objects
    bin_counts["bin_left"] = bin_counts["log_bin"].apply(
        lambda x: x.left).astype(float)
    bin_counts["bin_right"] = bin_counts["log_bin"].apply(
        lambda x: x.right).astype(float)

    # Convert bin edges from log10(minutes) → minutes
    bin_counts["min_left"] = (10 ** bin_counts["bin_left"]).round(2)
    bin_counts["min_right"] = (10 ** bin_counts["bin_right"]).round(2)


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
        title_text="Total Play Time Distribution (Log Scale, Minutes) – Users > {min_minutes:.1f} min"
    )

    fig.update_yaxes(title_text="% of Users" if as_percent else "User Count")
    fig.update_layout(
        title="Log10 Histogram of Engagement Time (Minutes)"
    )

    # Display in Streamlit
    st.plotly_chart(fig, use_container_width=True, key=f"{key}-{min_minutes}")


def engagement_scatter_plot(df, min_minutes=1, key="key"):
    df_trimmed = df[df["total_time_minutes"] > min_minutes].copy()

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
    st.plotly_chart(fig, use_container_width=True, key=f"{key}-{min_minutes}")


def engagement_pareto_chart(df, min_minutes=1, key="key"):
    df_trimmed = df[df["total_time_minutes"] > min_minutes].copy()

    top_countries = df_trimmed["country"].value_counts()
    valid_countries = top_countries[top_countries >= 100].index
    df_trimmed = df_trimmed[df_trimmed["country"].isin(valid_countries)]

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

    st.plotly_chart(fig, use_container_width=True, key=f"{key}-{min_minutes}")


def engagement_box_plot(df, key="key", min_minutes=1):
    # Filter users with very short sessions
    df_filtered = df[df["total_time_minutes"] > min_minutes].copy()


    fig = px.violin(
        df_filtered,
        y="total_time_minutes",
        box=True,         # Show box plot inside the violin
        points="all",     # Show all user data points
        title="Violin Plot of Total Engagement Time (Minutes)",
        labels={"total_time_minutes": "Total Time (min)"}
    )

    fig.update_traces(hovertemplate="%{y:.2f} minutes")
    st.plotly_chart(fig, use_container_width=True, key=f"{key}-{min_minutes}")


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
    st.plotly_chart(fig, use_container_width=True, key=f"{key}-{min_minutes}-2")
    
def most_engaged_users_chart(df,key="key"):   
    top_n = 20

    df_top = df.sort_values("total_time_minutes", ascending=False).head(top_n)

    # Then plot using the minutes column
    fig = px.bar(
        df_top,
        x="country",
        y="total_time_minutes",
        title=f"Top {top_n} Most Engaged Users (by Time)",
        labels={
            "country": "Country of user",
            "total_time_minutes": "Total Time (min)"
        },
        text_auto=".2f"
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True, key=key)
    

def first_open_vs_total_time(df, min_minutes=1, key="key"):
    df = df[df["total_time_minutes"] > min_minutes].copy()

    # Create weekly cohort from first_open
    df["first_open_date"] = pd.to_datetime(
        df["first_open"]).dt.to_period("W").dt.to_timestamp()

    # Aggregate by weekly cohort
    df_weekly = df.groupby("first_open_date", observed=True).agg(
        avg_time=("total_time_minutes", "mean"),
        user_count=("user_pseudo_id", "count")
    ).reset_index()

    # Add rolling average
    df_weekly["avg_time_smoothed"] = df_weekly["avg_time"].rolling(
        window=3, min_periods=1).mean()

    # Round values for readability
    df_weekly["avg_time"] = df_weekly["avg_time"].round(2)
    df_weekly["avg_time_smoothed"] = df_weekly["avg_time_smoothed"].round(2)

    # Plot the main line
    fig = px.line(
        df_weekly,
        x="first_open_date",
        y="avg_time",
        title="Avg Total Time by First Open Week",
        labels={
            "first_open_date": "First Open (Week)",
            "avg_time": "Avg Time (min)",
            "user_count": "Users"
        },
        hover_data={"user_count": True, "avg_time": True}
    )

    # Add rolling average line
    fig.add_scatter(
        x=df_weekly["first_open_date"],
        y=df_weekly["avg_time_smoothed"],
        mode="lines",
        name="3-Week Rolling Avg",
        line=dict(dash="dash", color="orange"),
        hovertemplate="%{y:.2f} min<extra>3-Week Rolling Avg</extra>"
    )

    st.plotly_chart(fig, use_container_width=True, key=f"{key}-{min_minutes}")
    
def engagement_by_country_bar_chart(df,key="key"):

    # Count users per country
    user_counts = df.groupby("country")[
        "user_pseudo_id"].nunique().reset_index(name="user_count")

    # Filter to countries with more than 100 users
    eligible_countries = user_counts[user_counts["user_count"] > 100]["country"]
    df_filtered = df[df["country"].isin(eligible_countries)]

    # Aggregate metrics
    agg_df = df_filtered.groupby("country").agg(
        user_count=("user_pseudo_id", "nunique"),
        total_time_minutes_sum=("total_time_minutes", "sum"),
        engagement_event_count_sum=("engagement_event_count", "sum"),
        avg_time_per_user=("total_time_minutes", "mean"),
        avg_events_per_user=("engagement_event_count", "mean")
    ).reset_index()

    # Sort for each chart
    top_time = agg_df.sort_values(
        "total_time_minutes_sum", ascending=False).head(20)
    top_events = agg_df.sort_values(
        "engagement_event_count_sum", ascending=False).head(20)
    avg_time = agg_df.sort_values("avg_time_per_user", ascending=False).head(20)
    avg_events = agg_df.sort_values(
        "avg_events_per_user", ascending=False).head(20)

    # Plot
    fig = go.Figure()


    # Trace for Total Time (minutes)
    fig.add_trace(go.Bar(
        x=top_time["country"],
        y=top_time["total_time_minutes_sum"],
        name="Total Time (minutes)",
        visible=True,
        hovertemplate=(
            "<b>%{x}</b><br>" +
            "Total Minutes: %{y:,.0f}<extra></extra>"
        )
    ))

    # Trace for Total Events
    fig.add_trace(go.Bar(
        x=top_events["country"],
        y=top_events["engagement_event_count_sum"],
        name="Total Engagement Events",
        visible=False,
        hovertemplate=(
            "<b>%{x}</b><br>" +
            "Total Events: %{y:,}<extra></extra>"
        )
    ))

    # Trace for Avg Time per User
    fig.add_trace(go.Bar(
        x=avg_time["country"],
        y=avg_time["avg_time_per_user"],
        name="Avg Time per User (minutes)",
        visible=False,
        hovertemplate=(
            "<b>%{x}</b><br>" +
            "Avg Minutes per User: %{y:.2f}<br>" +
            "Total Minutes: %{customdata[0]:,.0f}<br>" +
            "Users: %{customdata[1]:,}<extra></extra>"
        ),
        customdata=avg_time[["total_time_minutes_sum", "user_count"]].values
    ))

    # Trace for Avg Events per User
    fig.add_trace(go.Bar(
        x=avg_events["country"],
        y=avg_events["avg_events_per_user"],
        name="Avg Events per User",
        visible=False,
        hovertemplate=(
            "<b>%{x}</b><br>" +
            "Avg Events per User: %{y:.2f}<br>" +
            "Total Events: %{customdata[0]:,.0f}<br>" +
            "Users: %{customdata[1]:,}<extra></extra>"
        ),
        customdata=avg_events[["engagement_event_count_sum", "user_count"]].values
    ))


    # Toggle buttons
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                showactive=True,
                x=0.5,
                y=1.2,
                xanchor="center",
                yanchor="top",
                bgcolor="#d9d9d9",
                bordercolor="#2c2c2c",
                borderwidth=2,
                font=dict(color="#000", size=13),
                pad={"r": 10, "t": 10},
                buttons=[
                    dict(label="Total Time (minutes)",
                        method="update",
                        args=[{"visible": [True, False, False, False]},
                            {"title": "Top Countries by Total Time (minutes)",
                                "yaxis": {"title": "Total Time (minutes)"}}]),
                    dict(label="Total Events",
                        method="update",
                        args=[{"visible": [False, True, False, False]},
                            {"title": "Top Countries by Engagement Event Count",
                                "yaxis": {"title": "Event Count"}}]),
                    dict(label="Avg Time per User",
                        method="update",
                        args=[{"visible": [False, False, True, False]},
                            {"title": "Top Countries by Avg Time per User",
                                "yaxis": {"title": "Avg Minutes per User"}}]),
                    dict(label="Avg Events per User",
                        method="update",
                        args=[{"visible": [False, False, False, True]},
                            {"title": "Top Countries by Avg Events per User",
                                "yaxis": {"title": "Avg Events per User"}}])
                ]
            )
        ],
        title="Top Countries by Total Time (minutes)",
        xaxis_title="Country",
        yaxis_title="Total Time (minutes)",
        barmode='group',
        height=600
    )

    st.plotly_chart(fig, use_container_width=True, key=f"{key}")
    

def compute_cdf(df, col, step):
    max_val = df[col].max()
    thresholds = np.arange(0, max_val + step, step)

    data = []
    for t in thresholds:
        proportion = (df[col] > t).mean()
        percent = proportion * 100
        if percent < 1:  # Stop when less than 1% of users exceed threshold
            break
        data.append((t, percent))

    return pd.DataFrame(data, columns=["threshold", "user_percent"])


def cumulative_distribution_chart(df, key="key"):

    c1, c2 = st.columns([1, 4])
    with c1:
        option = st.selectbox(
            "Select metric to display:",
            options=["Users Over Time", "Users Over Events"],
            key=f"select_{key}"
        )

    if option == "Users Over Time":
        step = 1
        cdf_df = compute_cdf(df, "total_time_minutes", step)
        x_label = "Time Threshold (minutes)"
        y_label = "Percent of Users with More Time"
        hover_template = (
            "Time Threshold: %{x} minutes<br>" +
            "Users Above Threshold: %{y:.2f}%<extra></extra>"
        )
    else:
        step = 1
        cdf_df = compute_cdf(df, "engagement_event_count", step)
        x_label = "Event Count Threshold"
        y_label = "Percent of Users with More Events"
        hover_template = (
            "Event Threshold: %{x} events<br>" +
            "Users Above Threshold: %{y:.2f}%<extra></extra>"
        )

    if not cdf_df.empty:
        fig = px.line(
            cdf_df,
            x="threshold",
            y="user_percent",
            labels={"threshold": x_label, "user_percent": y_label},
            title=option
        )

        fig.update_traces(mode="lines+markers", hovertemplate=hover_template)
        fig.update_layout(yaxis_ticksuffix="%", xaxis_range=[
                          0, cdf_df["threshold"].max()])

        st.plotly_chart(fig, use_container_width=True, key=f"{key}_plot")
    else:
        st.warning(
            "No data to plot — fewer than 1% of users exceed any threshold.")
        

def engagement_device_analysis(df, key="key-eda", min_users=50, max_devices=30):
    df = df.copy()

    # Normalize "empty-ish" values to "(unknown)"
    for col in ["device_category", "device_mobile_brand_name", "device_mobile_model_name", "device_mobile_marketing_name"]:
        if col in df.columns:
            df[col] = df[col].replace("", None).fillna("(unknown)")

    # Streamlit layout: 2 columns
    col1, col2 = st.columns(2)

    with col1:
        groupby_option = st.radio(
            "Group by device field:",
            options=[
                "device_category",
                "device_mobile_brand_name",
                "device_mobile_model_name",
                "device_mobile_marketing_name"
            ],
            index=0,
            key=f"{key}-groupby-radio"
        )

    with col2:
        metric_option = st.radio(
            "Metric to focus on:",
            options=["Total Engagement Minutes", "Engagement Event Count"],
            index=0,
            key=f"{key}-metric-radio"
        )

        view_option = st.radio(
            "View Type:",
            options=["Top Devices by Total", "Top Devices by Average Per User",
                     "Scatter Plot: Engagement Analysis"],
            index=0,
            key=f"{key}-view-radio"
        )

    # Aggregate
    df_summary = (
        df.groupby(groupby_option, dropna=False)
        .agg(
            total_minutes=("total_time_minutes", "sum"),
            total_events=("engagement_event_count", "sum"),
            user_count=("user_pseudo_id", "nunique")
        )
        .reset_index()
    )

    # Calculate averages
    df_summary["avg_minutes_per_user"] = df_summary["total_minutes"] / \
        df_summary["user_count"]
    df_summary["avg_events_per_user"] = df_summary["total_events"] / \
        df_summary["user_count"]

    # Filter based on min_users
    df_summary = df_summary[df_summary["user_count"] >= min_users]

    # Choose metric
    if metric_option == "Total Engagement Minutes":
        total_col = "total_minutes"
        avg_col = "avg_minutes_per_user"
        y_label_total = "Total Engagement Time (minutes)"
        y_label_avg = "Average Engagement Time per User (minutes)"
    else:
        total_col = "total_events"
        avg_col = "avg_events_per_user"
        y_label_total = "Total Engagement Event Count"
        y_label_avg = "Average Engagement Events per User"

    # Format for hover
    df_summary["user_count_fmt"] = df_summary["user_count"].apply(
        lambda x: f"{x:,}")
    df_summary["total_fmt"] = df_summary[total_col].apply(
        lambda x: f"{x:,.0f}" if x >= 100 else f"{x:.2f}")
    df_summary["avg_fmt"] = df_summary[avg_col].apply(lambda x: f"{x:.2f}")

    if view_option == "Top Devices by Total":
        df_plot = df_summary.sort_values(
            total_col, ascending=False).head(max_devices)

        fig = px.bar(
            df_plot,
            x=groupby_option,
            y=total_col,
            labels={groupby_option: groupby_option.replace("_", " ").title()},
            title=f"Top {max_devices} {groupby_option.replace('_', ' ').title()} by {y_label_total}",
            custom_data=[groupby_option,
                         "user_count_fmt", "total_fmt", "avg_fmt"]
        )

        fig.update_traces(
            hovertemplate=(
                "<b>%{customdata[0]}</b><br><br>"
                "Users: %{customdata[1]}<br>"
                "Total: %{customdata[2]}<br>"
                "Average per User: %{customdata[3]}<extra></extra>"
            )
        )

        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    elif view_option == "Top Devices by Average Per User":
        df_plot = df_summary.sort_values(
            avg_col, ascending=False).head(max_devices)

        fig = px.bar(
            df_plot,
            x=groupby_option,
            y=avg_col,
            labels={groupby_option: groupby_option.replace("_", " ").title()},
            title=f"Top {max_devices} {groupby_option.replace('_', ' ').title()} by {y_label_avg}",
            custom_data=[groupby_option,
                         "user_count_fmt", "total_fmt", "avg_fmt"]
        )

        fig.update_traces(
            hovertemplate=(
                "<b>%{customdata[0]}</b><br><br>"
                "Users: %{customdata[1]}<br>"
                "Total: %{customdata[2]}<br>"
                "Average per User: %{customdata[3]}<extra></extra>"
            )
        )

        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    else:  # Scatter plot
        scatter_x_option = st.radio(
            "Scatter X-axis:",
            options=["Number of Users", "Average per User"],
            index=0,
            key=f"{key}-scatter-x-radio"
        )

        if scatter_x_option == "Number of Users":
            x_axis = "user_count"
            size_axis = avg_col
            x_axis_label = "Number of Users"
        else:
            x_axis = avg_col
            size_axis = "user_count"
            x_axis_label = y_label_avg

        fig = px.scatter(
            df_summary,
            x=x_axis,
            y=total_col,
            size=size_axis,
            color=groupby_option,
            labels={
                x_axis: x_axis_label,
                total_col: y_label_total,
                size_axis: "Bubble Size"
            },
            title=f"{x_axis_label} vs {y_label_total} by {groupby_option.replace('_', ' ').title()}",
            custom_data=[groupby_option,
                         "user_count_fmt", "total_fmt", "avg_fmt"]
        )

        fig.update_traces(
            hovertemplate=(
                "<b>%{customdata[0]}</b><br><br>"
                "Users: %{customdata[1]}<br>"
                "Total: %{customdata[2]}<br>"
                "Average per User: %{customdata[3]}<extra></extra>"
            )
        )

        fig.update_layout(
            xaxis_title=x_axis_label,
            yaxis_title=y_label_total,
            hovermode="closest"
        )
        st.plotly_chart(fig, use_container_width=True)


    # CSV export remains at the bottom
    csv = ui.convert_for_download(df)
    st.download_button(label="Download all data", data=csv, file_name="df.csv",
                       key="d1", icon=":material/download:", mime="text/csv")
    csv = ui.convert_for_download(df_summary)
    st.download_button(label="Download summary data", data=csv, file_name="df_summary.csv",
                       key="d2", icon=":material/download:", mime="text/csv")


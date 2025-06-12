import streamlit as st
from rich import print
import pandas as pd
import numpy as np
import datetime as dt
import users


from config import default_daterange


@st.cache_data(ttl="1d", show_spinner=True)
def build_engagement_cost_table(df_campaigns, df_users):
    # Remove Unity campaigns (e.g., FTM)
    df_campaigns = df_campaigns[~df_campaigns["campaign_name"].str.contains(
        "FTM", na=False)]

    # Aggregate total cost by country
    df_campaigns = (
        df_campaigns.groupby("country", as_index=False)
        .agg({"cost": "sum"})
        .round(2)
    )

    # === Compute Literacy Rate (LR) per country (number of unique users) ===
    df_lr = (
        df_users.groupby("country", as_index=False)
        .agg({"cr_user_id": "nunique"})
        .rename(columns={"cr_user_id": "LR"})
    )

    # Merge LR and compute LRC (cost per user)
    df_campaigns = df_campaigns.merge(df_lr, on="country", how="left")
    df_campaigns["LRC"] = np.where(
        df_campaigns["LR"] > 0,
        (df_campaigns["cost"] / df_campaigns["LR"]).round(2),
        0
    )

    # Merge in external literacy_rate
    countries_dataframe = users.get_country_literacy_dataframe()
    df = df_campaigns.merge(
        countries_dataframe[["country", "literacy_rate"]],
        on="country", how="left"
    )

    # Compute average time per user in minutes per country
    df_users_avg = (
        df_users.groupby("country", as_index=False)
        .agg({"total_time_minutes": "mean"})
        .rename(columns={"total_time_minutes": "avg_time_minutes"})
        .round(2)
    )
    df = df.merge(df_users_avg, on="country", how="left")

    # Compute total engagement minutes per country
    df_total_minutes = (
        df_users.groupby("country", as_index=False)
        .agg({"total_time_minutes": "sum"})
        .rename(columns={"total_time_minutes": "total_minutes"})
        
    )
    df = df.merge(df_total_minutes, on="country", how="left")

    # Compute Cost Per Engagement Minute (CPEM)
    df["CPEM"] = np.where(
        df["total_minutes"] > 0,
        (df["cost"] / df["total_minutes"]).round(2),
        0
    )

    return df


# Get the campaign data and filter by date, language, and country selections
@st.cache_data(ttl="1d", show_spinner=False)
def filter_campaigns(df_campaigns_all, daterange,  countries_list):

    # Drop the campaigns that don't meet the naming convention
    condition = (df_campaigns_all["app_language"].isna()) | (
        df_campaigns_all["country"].isna())
    df_campaigns = df_campaigns_all[~condition]

    mask = (df_campaigns['segment_date'].dt.date >= daterange[0]) & (
        df_campaigns['segment_date'].dt.date <= daterange[1])
    df_campaigns = df_campaigns.loc[mask]
    # Apply country filter if not "All"

    if countries_list[0] != "All":
      mask &= df_campaigns['country'].isin(set(countries_list))


    df_campaigns = df_campaigns.loc[mask]

    col = df_campaigns.pop("country")
    df_campaigns.insert(2, col.name, col)
    df_campaigns.reset_index(drop=True, inplace=True)

    col = df_campaigns.pop("app_language")
    df_campaigns.insert(3, col.name, col)
    df_campaigns.reset_index(drop=True, inplace=True)

    return df_campaigns

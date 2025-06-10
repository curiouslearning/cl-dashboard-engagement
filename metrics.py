import streamlit as st
from rich import print
import pandas as pd
import numpy as np
import datetime as dt
import users


from config import default_daterange

def get_LR(
    daterange=default_daterange,
    countries_list=[],

):

    # if no list passed in then get the full list
    if len(countries_list) == 0:
        countries_list = users.get_country_list()

    df_user_list = filter_user_data(
        daterange=daterange, countries_list=countries_list
    )

    return len(df_user_list) #All LR 


# Takes the complete user lists (cr_user_id) and filters based on input data, and returns
# a new filtered dataset
def filter_user_data(
    daterange=default_daterange,
    countries_list=["All"],

):

    # Check if necessary dataframes are available
    if not all(key in st.session_state for key in [ "df_cr_app_launch"]):
        print("PROBLEM!")
        return pd.DataFrame()


    df = st.session_state.df_cr_app_launch

    # Initialize a boolean mask
    mask = (df['first_open'] >= daterange[0]) & (df['first_open'] <= daterange[1])

    if countries_list[0] != "All":
        mask &= df['country'].isin(set(countries_list))
    
    # Filter the dataframe with the combined mask
    df = df.loc[mask]

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

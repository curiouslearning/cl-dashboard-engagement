import streamlit as st
import pandas as pd
from rich import print as print
import numpy as np
from pyinstrument import Profiler
import logging
import asyncio

# Event data started getting the campaign metadata in production on this date
start_date = '2024-11-08'

# Firebase returns two different formats of user_pseudo_id between
# web app events and android events, so we have to run multiple queries
# instead of a join because we don't have a unique key for both
# This would all be unncessery if dev had included the app user id per the spec.


import logging
import streamlit as st

async def get_users_list():
    p = Profiler(async_mode="disabled")
    with p:

        bq_client = st.session_state.bq_client

        # Helper function to run BigQuery in a thread
        async def run_query(query):
            return await asyncio.to_thread(bq_client.query(query).to_dataframe)

        # Define the queries
        sql_cr_app_launch = f"""
            SELECT *
            FROM `dataexploration-193817.user_data.cr_app_launch`
            WHERE first_open BETWEEN PARSE_DATE('%Y-%m-%d','{start_date}') AND CURRENT_DATE()
        """
        
        sql_day1_app_remove = f"""
            SELECT *
            FROM `dataexploration-193817.user_data.day1_app_remove_users`
         """
         
        google_ads_query = f"""
            SELECT
                distinct metrics.campaign_id,
                metrics.segments_date as segment_date,
                campaigns.campaign_name,
                metrics_cost_micros as cost
            FROM dataexploration-193817.marketing_data.p_ads_CampaignStats_6687569935 as metrics
            INNER JOIN dataexploration-193817.marketing_data.ads_Campaign_6687569935 as campaigns
            ON metrics.campaign_id = campaigns.campaign_id
            AND metrics.segments_date >= '{start_date}'

        """

        # Facebook Ads Query
        facebook_ads_query = f"""
            SELECT 
                d.campaign_id,
                d.data_date_start as segment_date,
                d.campaign_name,
                d.spend as cost
            FROM dataexploration-193817.marketing_data.facebook_ads_data as d
            WHERE d.data_date_start >= '{start_date}'
            ORDER BY d.data_date_start DESC;
        """

        # Run all the queries asynchronously
        df_day1_app_remove, df_cr_app_launch, df_google_ads_data, df_facebook_ads_data = await asyncio.gather(
            run_query(sql_day1_app_remove),
            run_query(sql_cr_app_launch),
            run_query(google_ads_query),
            run_query(facebook_ads_query)
        )

        # Process Google Ads Data
        df_google_ads_data["campaign_id"] = df_google_ads_data["campaign_id"].astype(
            str).str.replace(",", "")
        df_google_ads_data["cost"] = df_google_ads_data["cost"].divide(
            1000000).round(2)
        df_google_ads_data["segment_date"] = pd.to_datetime(
            df_google_ads_data["segment_date"])

    p.print(color="red")
    
    return df_day1_app_remove, df_cr_app_launch, df_google_ads_data, df_facebook_ads_data

    


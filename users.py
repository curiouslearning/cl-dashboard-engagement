import streamlit as st
import pandas as pd
from rich import print as print
import numpy as np
from pyinstrument import Profiler
import asyncio

start_date = '2024-05-01'
# Starting 05/01/2024, campaign names were changed to support an indication of
# both language and country through a naming convention.  So we are only collecting
# and reporting on daily campaign segment data from that day forward.

# Firebase returns two different formats of user_pseudo_id between
# web app events and android events, so we have to run multiple queries
# instead of a join because we don't have a unique key for both
# This would all be unncessery if dev had included the app user id per the spec.


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
         


        # Run all the queries asynchronously
        df_day1_app_remove, df_cr_app_launch,  = await asyncio.gather(
            run_query(sql_day1_app_remove),
            run_query(sql_cr_app_launch),

        )

    p.print(color="red")
    
    return df_day1_app_remove, df_cr_app_launch

    
@st.cache_data(ttl="1d", show_spinner=False)
def get_country_list():
    countries_list = []
    if "bq_client" in st.session_state:
        bq_client = st.session_state.bq_client
        sql_query = f"""
                    SELECT country
                    FROM `dataexploration-193817.user_data.active_countries`
                    order by country asc
                    ;
                    """
        rows_raw = bq_client.query(sql_query)
        rows = [dict(row) for row in rows_raw]
        if len(rows) == 0:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        countries_list = np.array(df.values).flatten().tolist()
    return countries_list


@st.cache_data(ttl="1d", show_spinner=False)
def get_country_literacy_dataframe():
    countries_dataframe = []
    if "bq_client" in st.session_state:
        bq_client = st.session_state.bq_client
        sql_query = f"""
                    SELECT *
                    FROM `dataexploration-193817.user_data.active_countries`
                    order by country asc
                    ;
                    """
        rows_raw = bq_client.query(sql_query)
        rows = [dict(row) for row in rows_raw]
        if len(rows) == 0:
            return pd.DataFrame()

        countries_dataframe = pd.DataFrame(rows)

    return countries_dataframe

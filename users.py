import streamlit as st
import pandas as pd
from rich import print as print
import numpy as np
from pyinstrument import Profiler
import logging
import asyncio

# How far back to obtain user data.  Currently the queries pull back to 01/01/2021
start_date = "2023/01/01"

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
        sql_cr_engagement = f"""
            SELECT *
            FROM `dataexploration-193817.user_data.cr_user_engagement`
            WHERE first_open BETWEEN PARSE_DATE('%Y/%m/%d','{start_date}') AND CURRENT_DATE()
        """

        # Run all the queries asynchronously
        df_cr_engagement = await asyncio.gather(
            run_query(sql_cr_engagement),

        )

    p.print(color="red")
    
    return df_cr_engagement


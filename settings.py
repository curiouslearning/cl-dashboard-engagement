import logging
import streamlit as st
from google.oauth2 import service_account
from google.cloud import bigquery
from rich import print
import pandas as pd
import datetime as dt
from google.cloud import secretmanager
import json
import asyncio

# Starting 05/01/2024, campaign names were changed to support an indication of
# both language and country through a naming convention.  So we are only collecting
# and reporting on daily campaign segment data from that day forward.
default_daterange = [dt.datetime(2024, 5, 1).date(), dt.date.today()]
start_date = '2024-05-01'

@st.cache_resource(ttl="1d")
def get_logger(name="dashboard_logger"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        # Console handler or file handler
        handler = logging.StreamHandler()  # or logging.FileHandler("logs/app.log")

        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        logger.addHandler(handler)
        logger.propagate = False  # Prevent double logging
    return logger


@st.cache_resource(ttl="1d")
def get_gcp_credentials():
    client = secretmanager.SecretManagerServiceClient()
    name = "projects/405806232197/secrets/service_account_json/versions/latest"
    response = client.access_secret_version(name=name)
    key = response.payload.data.decode("UTF-8")

    service_account_info = json.loads(key)
    gcp_credentials = service_account.Credentials.from_service_account_info(
        service_account_info,
        scopes=[
            "https://www.googleapis.com/auth/cloud-platform",
            "https://www.googleapis.com/auth/devstorage.read_only",
            "https://www.googleapis.com/auth/bigquery",
            "https://www.googleapis.com/auth/drive",
        ],
    )

    bq_client = bigquery.Client(
        credentials=gcp_credentials, project="dataexploration-193817"
    )

    return gcp_credentials, bq_client


def initialize():
    pd.options.mode.copy_on_write = True
    pd.set_option("display.max_columns", 20)


# Get the campaign data from BigQuery, roll it up per campaign
def init_data():
    from campaigns import add_country_and_language

    # Call the combined asynchronous campaign data function
    df_google_ads_data, df_facebook_ads_data = cache_marketing_data()

    # Get all campaign data by segment_date
    df_campaigns_all = pd.concat([df_google_ads_data, df_facebook_ads_data])
    df_campaigns_all = add_country_and_language(df_campaigns_all)
    df_campaigns_all = df_campaigns_all.reset_index(drop=True)


    if "df_campaigns_all" not in st.session_state:
        st.session_state["df_campaigns_all"] = df_campaigns_all

    from users import ensure_user_data_initialized
    ensure_user_data_initialized()


@st.cache_data(ttl="1d", show_spinner="Loading Data")
def cache_marketing_data():
    from campaigns import get_campaign_data
    # Execute the async function and return its result synchronously
    return asyncio.run(get_campaign_data())

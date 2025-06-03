import streamlit as st
from google.oauth2 import service_account
from google.cloud import bigquery
from rich import print
import pandas as pd
import users
import datetime as dt
from google.cloud import secretmanager
import json
import asyncio

default_daterange = [dt.datetime(2021, 1, 1).date(), dt.date.today()]


def get_gcp_credentials():
    # first get credentials to secret manager
    client = secretmanager.SecretManagerServiceClient()

    # get the secret that holds the service account key
    name = "projects/405806232197/secrets/service_account_json/versions/latest"
    response = client.access_secret_version(name=name)
    key = response.payload.data.decode("UTF-8")

    # use the key to get service account credentials
    service_account_info = json.loads(key)
    # Create BigQuery API client.
    gcp_credentials = service_account.Credentials.from_service_account_info(
        service_account_info,
        scopes=[
            "https://www.googleapis.com/auth/cloud-platform",
            "https://www.googleapis.com/auth/drive",
            "https://www.googleapis.com/auth/bigquery",
        ],
    )

    bq_client = bigquery.Client(
        credentials=gcp_credentials, project="dataexploration-193817"
    )
    return bq_client


def initialize():
    pd.options.mode.copy_on_write = True
    pd.set_option("display.max_columns", 20)

    bq_client = get_gcp_credentials()

    if "bq_client" not in st.session_state:
        st.session_state["bq_client"] = bq_client

def init_user_list():

    df_day1_app_remove, df_cr_app_launch, google_ads_data, facebook_ads_data = cache_users_list()

    if "df_cr_app_launch" not in st.session_state:
        st.session_state["df_cr_app_launch"] = df_cr_app_launch

    if "df_google_ads_data" not in st.session_state:
        st.session_state["df_google_ads_data"] = google_ads_data

    if "df_facebook_ads_data" not in st.session_state:
        st.session_state["df_facebook_ads_data"] = facebook_ads_data

        # only keep one cr_user_id row per user
        df_cr_app_launch = df_cr_app_launch.drop_duplicates(
            subset='cr_user_id')

        # remove the users who uninstalled on day 1
        df_cr_app_launch = df_cr_app_launch[~df_cr_app_launch["user_pseudo_id"].isin(
            df_day1_app_remove["user_pseudo_id"])]



@st.cache_data(ttl="1d", show_spinner="Gathering User List")
def cache_users_list():
    # Execute the async function and return its result synchronously
    return asyncio.run(users.get_users_list())



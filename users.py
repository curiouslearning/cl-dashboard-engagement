import streamlit as st
import pandas as pd
from rich import print as print
import numpy as np

import gcsfs



@st.cache_data(ttl="1d", show_spinner=False)
def load_parquet_from_gcs(file_pattern: str) -> pd.DataFrame:
    from settings import get_gcp_credentials
    credentials, _ = get_gcp_credentials()
    fs = gcsfs.GCSFileSystem(
        project="dataexploration-193817", token=credentials)
    files = fs.glob(file_pattern)
    if not files:
        raise FileNotFoundError(f"No files matching pattern: {file_pattern}")
    df = pd.read_parquet(files, filesystem=fs).copy()

    return df


def load_unity_user_progress_from_gcs():
    return load_parquet_from_gcs("user_data_parquet_cache/unity_user_progress_*.parquet")


def load_day1_app_remove_from_gcs():
    return load_parquet_from_gcs("user_data_parquet_cache/day1_app_remove_from_gcs_*.parquet")


def load_cr_app_launch_from_gcs():
    return load_parquet_from_gcs("user_data_parquet_cache/cr_app_launch_device_data_*.parquet")


def ensure_user_data_initialized():
    import traceback
    """Run init_user_data once per session, with error handling."""
    if "user_data_initialized" not in st.session_state:
        try:
            init_user_data()
            st.session_state["user_data_initialized"] = True
        except Exception as e:
            st.error(f"❌ Failed to initialize user data: {e}")
            st.text(traceback.format_exc())
            st.stop()


def init_user_data():
    if st.session_state.get("user_data_initialized"):
        return  # already initialized this session
    with st.spinner("Loading User Data", show_time=True):
        from pyinstrument import Profiler
        from pyinstrument.renderers.console import ConsoleRenderer
        import settings

        profiler = Profiler(async_mode="disabled")
        with profiler:
            # Cached fast parquet loads
            df_day1_app_remove = load_day1_app_remove_from_gcs()
            df_unity_users = load_unity_user_progress_from_gcs()
            df_cr_app_launch = load_cr_app_launch_from_gcs()

            # Validation
            if df_day1_app_remove.empty or df_unity_users.empty or df_cr_app_launch.empty:
                raise ValueError(
                    "❌ One or more dataframes were empty after loading.")


            df_unity_users = fix_date_columns(
                df_unity_users, ["first_open", "la_date", "last_event_date"])
            df_cr_app_launch = fix_date_columns(
                df_cr_app_launch, ["first_open"])
            
            from settings import start_date
            df_unity_users = df_unity_users[
                df_unity_users["first_open"] >= start_date]

            df_cr_app_launch = df_cr_app_launch[
                df_cr_app_launch["first_open"] >= start_date]

            max_level_indices = df_unity_users.groupby(
                "user_pseudo_id")["max_user_level"].idxmax()
            df_unity_users = df_unity_users.loc[max_level_indices].reset_index(
                drop=True)

            df_cr_app_launch["app_language"] = clean_language_column(
                df_cr_app_launch)

            # only keep one cr_user_id row per user
            df_cr_app_launch = df_cr_app_launch.drop_duplicates(
                subset='cr_user_id')

            # remove the users who uninstalled on day 1
            df_cr_app_launch = df_cr_app_launch[~df_cr_app_launch["user_pseudo_id"].isin(
                df_day1_app_remove["user_pseudo_id"])]

            # Assign to session state

            st.session_state["df_unity_users"] = df_unity_users
            st.session_state["df_cr_app_launch"] = df_cr_app_launch
            st.session_state["user_data_initialized"] = True

        # Log the profile only once
        settings.get_logger().debug(
            profiler.output(ConsoleRenderer(
                show_all=False, timeline=True, color=True, unicode=True, short_mode=False))
        )

# Language cleanup
def clean_language_column(df):
    return df["app_language"].replace({
        "ukranian": "ukrainian",
        "malgache": "malagasy",
        "arabictest": "arabic",
        "farsitest": "farsi"
    })


@st.cache_data(ttl="1d", show_spinner=False)
def get_language_list():
    lang_list = ["All"]
    from settings import get_gcp_credentials
    _, bq_client = get_gcp_credentials()

    sql_query = f"""
                SELECT display_language
                FROM `dataexploration-193817.user_data.language_max_level`
                ;
                """
    rows_raw = bq_client.query(sql_query)
    rows = [dict(row) for row in rows_raw]
    if len(rows) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.drop_duplicates(inplace=True)
    lang_list = np.array(df.values).flatten().tolist()
    lang_list = [x.strip(" ") for x in lang_list]
    return lang_list


@st.cache_data(ttl="1d", show_spinner=False)
def get_country_list():
    countries_list = []
    from settings import get_gcp_credentials
    _, bq_client = get_gcp_credentials()

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



def fix_date_columns(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


@st.cache_data(ttl="1d", show_spinner=False)
def get_country_literacy_dataframe():
    countries_dataframe = []
    
    from settings import get_gcp_credentials
    _, bq_client = get_gcp_credentials()
    
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

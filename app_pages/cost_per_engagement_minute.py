import streamlit as st
from rich import print as rprint
from millify import prettify
import metrics
import users
import ui_components as uic

from settings import default_daterange
from settings import initialize
from settings import init_data

initialize()
init_data()

c1, c2 = st.columns([1, 8])

df_campaigns_all = st.session_state["df_campaigns_all"]
df_cr_app_launch = st.session_state["df_cr_app_launch"]

countries_list = countries_list = users.get_country_list()
df_campaigns = metrics.filter_campaigns(
    df_campaigns_all,default_daterange,countries_list)

df = metrics.build_engagement_cost_table(
    df_campaigns, df_cr_app_launch)

uic.cpem_scatter_plot(df)
st.divider()
st.subheader("CPEM Data")

df_display = df.copy()
df_display["cost"] = df_display["cost"].map(lambda x: f"{x:,.2f}")
df_display["LR"] = df_display["LR"].map(lambda x: f"{x:,.0f}")
df_display["LRC"] = df_display["LRC"].map(lambda x: f"{x:.2f}")
df_display["total_minutes"] = df_display["total_minutes"].round(
    0).astype("Int64")
df_display["avg_time_minutes"] = df_display["avg_time_minutes"].map(
    lambda x: f"{x:.2f}")
st.table(df_display)  


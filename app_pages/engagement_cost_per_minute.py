import streamlit as st
from rich import print as rprint
from millify import prettify
import ui_widgets as ui
import settings
import metrics
import users
import campaigns

settings.initialize()
settings.init_user_list()
settings.init_campaign_data()

from config import default_daterange

c1, c2 = st.columns([1, 8])

df_campaigns_all = st.session_state["df_campaigns_all"]
countries_list = countries_list = users.get_country_list()
df_campaigns = metrics.filter_campaigns(
    df_campaigns_all,default_daterange,countries_list)

df = campaigns.build_campaign_table(df_campaigns, daterange=default_daterange)
df_display = df.copy()
df_display["cost"] = df_display["cost"].map(lambda x: f"{x:,.2f}")
df_display["LR"] = df_display["LR"].map(lambda x: f"{x:,.0f}")
df_display["LRC"] = df_display["LRC"].map(lambda x: f"{x:.2f}")

st.table(df_display)  # or st.dataframe(df_display)

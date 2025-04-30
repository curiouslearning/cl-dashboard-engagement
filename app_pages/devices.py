import streamlit as st
from rich import print as rprint
from millify import prettify
import ui_widgets as ui
import ui_components as uic
import settings
import plotly.express as px

settings.initialize()
settings.init_user_list()

c1, c2 = st.columns([1, 8])

df_cr_engagement = st.session_state["df_cr_engagement"]
    
uic.engagement_device_analysis(df=df_cr_engagement, min_users=15,max_devices=25, key="dev-2")


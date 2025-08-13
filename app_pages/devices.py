import streamlit as st
from rich import print as rprint
from millify import prettify
import ui_components as uic

from settings import initialize
from settings import init_data

initialize()
init_data()

c1, c2 = st.columns([1, 8])

df_cr_app_launch = st.session_state["df_cr_app_launch"]
    
uic.engagement_device_analysis(df=df_cr_app_launch, min_users=15,max_devices=25, key="dev-2")


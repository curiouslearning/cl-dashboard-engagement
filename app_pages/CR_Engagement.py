import streamlit as st
from rich import print as rprint
from millify import prettify
import ui_widgets as ui
import settings



settings.initialize()
settings.init_user_list()


df_cr_engagement = st.session_state["df_cr_engagement"]
df_cr_engagement
import streamlit as st
from rich import print as rprint
from millify import prettify
import ui_widgets as ui
import ui_components as uic
import settings



settings.initialize()
settings.init_user_list()

c1, c2 = st.columns([1, 8])
with c1:
    min_seconds = st.number_input("Min Minutes Played", value=1)
    min_seconds = min_seconds * 60
    
df_cr_engagement = st.session_state["df_cr_engagement"]
# 1. Create the tabs first
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Histograms", "Scatter Plot", "Pareto", "Box Plot", "Most Engaged"])


with tab1:
    by_percent = st.toggle("By Percent", value=True)
    uic.engagement_histogram(
        df_cr_engagement, key="CRE-1", as_percent=by_percent, min_seconds=min_seconds)
    uic.engagement_histogram_log(
        df_cr_engagement, nbins=50, key="CRE-2", as_percent=by_percent, min_seconds=min_seconds)

with tab2:
    uic.engagement_scatter_plot(
        df_cr_engagement, key="CRE-3", min_seconds=min_seconds)

with tab3:
    uic.engagement_pareto_chart(
        df_cr_engagement, key="CRE-4", min_seconds=min_seconds)

with tab4:
    uic.engagement_box_plot(
        df_cr_engagement, key="CRE-5", min_seconds=min_seconds)

with tab5: 
    uic.most_engaged_users_chart(df_cr_engagement,key="CRE=6")

    

csv = ui.convert_for_download(df_cr_engagement)
st.download_button(label="Download CSV", data=csv, file_name="df_cr_engagement.csv",
                   key="fc-10", icon=":material/download:", mime="text/csv")

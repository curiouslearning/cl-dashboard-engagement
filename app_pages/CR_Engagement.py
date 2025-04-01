import streamlit as st
from rich import print as rprint
from millify import prettify
import ui_widgets as ui
import ui_components as uic
import settings



settings.initialize()
settings.init_user_list()


df_cr_engagement = st.session_state["df_cr_engagement"]
tab1, tab2, tab3 = st.tabs(["Histograms", "blank", "blank"])

with tab1:
    by_percent = st.toggle("By Percent",value=True)
    uic.engagement_histogram_below_threshhold(
        df_cr_engagement, percentile=.95, key="CRE-1", as_percent=by_percent)
    uic.engagement_histogram_log(
        df_cr_engagement, percentile=1.00, nbins=50, key="CRE-2", as_percent=by_percent)
    
csv = ui.convert_for_download(df_cr_engagement)    
st.download_button(label="Download CSV", data=csv, file_name="df_cr_engagement.csv",
                   key="fc-10", icon=":material/download:", mime="text/csv")

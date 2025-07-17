import streamlit as st
from rich import print as rprint
from millify import prettify
import ui_widgets as ui
import ui_components as uic
import settings


settings.initialize()
settings.init_user_list()

c1, c2 = st.columns([1, 3])
with c1:
    min_minutes = st.number_input("Min Minutes Played", value=1)


app = ui.app_selector()
if (app == "CR"):
    df = st.session_state["df_cr_app_launch"]
    df["user_id"] = df["cr_user_id"]
else:
    df = st.session_state["df_unity_users"]
    df["user_id"] = df["user_pseudo_id"]


# Use selectbox instead of st.tabs for lazy evaluation
tab_options = [
    "Histograms", "By Country", "Scatter Plot",
    "Over Time", "Box Plot", "Pareto"
]
with c2:
    selected_tab = st.selectbox("Select Analysis View", tab_options)

# Conditional rendering
if selected_tab == "Histograms":
    by_percent = st.toggle("By Percent", value=True)
    uic.engagement_histogram(
        df, key="CRE-1", as_percent=by_percent, min_minutes=min_minutes)
    uic.cumulative_distribution_chart(df, key="CRE-1a")
    uic.engagement_histogram_log(
        df, nbins=50, key="CRE-2", as_percent=by_percent, min_minutes=min_minutes)

elif selected_tab == "By Country":
    uic.engagement_by_country_bar_chart(df, "CRE-6a")
    st.divider()
    uic.most_engaged_users_chart(df, key="CRE-6")

elif selected_tab == "Scatter Plot":
    uic.engagement_scatter_plot(
        df, key="CRE-3", min_minutes=min_minutes)

elif selected_tab == "Over Time":
    uic.first_open_vs_total_time(df, key="CRE-7")

elif selected_tab == "Box Plot":
    uic.engagement_box_plot(
        df, key="CRE-5", min_minutes=min_minutes)

elif selected_tab == "Pareto":
    uic.engagement_pareto_chart(
        df, key="CRE-4", min_minutes=min_minutes)

# CSV export remains at the bottom
csv = ui.convert_for_download(df)
st.download_button(label="Download CSV", data=csv, file_name="users.csv",
                   key="fc-10", icon=":material/download:", mime="text/csv")

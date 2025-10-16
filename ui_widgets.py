import streamlit as st

# Restricts selection to a single country


def single_selector(selections,  title="", key="key", include_All=True, index=0):
    options = list(selections)  # Defensive copy

    if include_All:
        if "All" not in options:
            options = ["All"] + [s for s in options if s != "All"]
    else:
        options = [s for s in options if s != "All"]

    selection = st.selectbox(
        index=index,
        label=title,
        options=options,
        key=key,
    )

    return [selection]

@st.cache_data
def convert_for_download(df):
    return df.to_csv().encode("utf-8")


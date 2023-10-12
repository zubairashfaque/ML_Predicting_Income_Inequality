import altair as alt
import streamlit as st

from data import get_data, get_github_scale

_df = get_data()
n = len(_df)  # FIXME: can use n in badge, without generating local file
st.subheader('Dataframe:')
n, m = _df.shape
st.write(f'<p style="font-size:130%">Dataset contains {n} rows and {m} columns.</p>', unsafe_allow_html=True)
st.dataframe(_df)


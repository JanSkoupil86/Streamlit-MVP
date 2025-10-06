import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Football Scouting App", page_icon="⚽", layout="wide")

st.title("⚽ Football Scouting Dashboard")
st.write("Upload your Wyscout dataset (CSV) to explore player statistics interactively.")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns.")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        metric = st.selectbox("Choose a metric to rank players:", numeric_cols)
        top_n = st.slider("Show top N players:", 5, 50, 10)
        player_col = "Player" if "Player" in df.columns else df.columns[0]

        top_players = df.sort_values(metric, ascending=False).head(top_n)
        st.subheader(f"Top {top_n} players by {metric}")
        st.dataframe(top_players[[player_col, metric]])

        fig = px.bar(top_players, x=metric, y=player_col, orientation="h", title=f"Top {metric} performers")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No numeric columns found for visualization.")
else:
    st.info("Please upload a CSV file to start.")

# app.py — Football Scouting App v2 (Multipage)
import streamlit as st
from lib.utils import init_state, load_csv_and_normalize, show_env_box

st.set_page_config(page_title="Scouting App v2", page_icon="⚽", layout="wide")
st.title("⚽ Scouting App v2 — Home / Data Manager")
show_env_box()

init_state()

st.markdown("Upload your Wyscout-style CSV here. The dataset is cached for other pages.")

file = st.file_uploader("Upload CSV", type=["csv"])
if file is not None:
    try:
        df = load_csv_and_normalize(file)
        st.session_state['df'] = df
        st.success(f"Loaded {df.shape[0]:,} rows × {df.shape[1]:,} columns")
        st.dataframe(df.head(20), use_container_width=True)
        st.caption("Columns have been lightly normalized (e.g., 'player' → 'Player').")
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")

st.info("Navigate using the **Pages** sidebar: Player Finder, Comparison, Similarity, Team Analytics, Shortlist & Export.")

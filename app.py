# app.py — Scouting App: Comparison & Similarity (safe build)
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Scouting App — Comparison & Similarity", page_icon="⚽", layout="wide")
st.title("⚽ Football Scouting App")
st.caption("Upload your Wyscout-style CSV and compare players or find similar profiles.")

# 1) Upload
file = st.file_uploader("Upload CSV", type=["csv"])
if file is None:
    st.info("Upload a CSV to begin.")
    st.stop()

try:
    df = pd.read_csv(file)
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

st.success(f"Loaded {df.shape[0]:,} rows × {df.shape[1]:,} columns")

# 2) Column detection
string_cols = df.select_dtypes(include=["object"]).columns.tolist()
player_col = next((c for c in ["Player", "player", "Name", "Full name"] if c in df.columns), None)
if player_col is None and string_cols:
    player_col = string_cols[0]
if player_col is None:
    st.error("Couldn't find a text-like column for player names.")
    st.stop()

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if not num_cols:
    st.error("No numeric columns found to compare.")
    st.stop()

# 3) Sidebar filters
with st.sidebar:
    st.header("Filters")
    minutes_col = next((c for c in ["Minutes", "minutes", "Minutes played", "Time played"] if c in df.columns), None)
    if minutes_col:
        min_minutes = st.number_input("Minimum minutes", min_value=0, value=300, step=30)
        df = df[df[minutes_col].fillna(0) >= min_minutes]

    pos_col = next((c for c in ["Main Position", "Position", "Role"] if c in df.columns), None)
    if pos_col:
        pos_vals = sorted([p for p in df[pos_col].dropna().unique().tolist() if isinstance(p, str)])
        chosen_pos = st.multiselect("Positions", pos_vals)
        if chosen_pos:
            df = df[df[pos_col].isin(chosen_pos)]

st.divider()
tab1, tab2 = st.tabs(["🔁 Player comparison", "🧭 Similarity finder"])

# 4) Player comparison
with tab1:
    st.subheader("Select 2–3 players and compare")
    left, right = st.columns([2,3])
    with left:
        all_players = sorted([p for p in df[player_col].dropna().unique().tolist() if isinstance(p, str)])
        players = st.multiselect("Players", all_players)  # keep simple & compatible
        metric_candidates = [c for c in num_cols if "per 90" in c.lower() or c.lower() in ("xg", "xa", "goals", "assists")] or num_cols[:20]
        default_pick = metric_candidates[:6] if len(metric_candidates) >= 6 else metric_candidates
        metrics = st.multiselect("Metrics (3–8 recommended)", metric_candidates, default=default_pick)
        chart_type = st.radio("Chart type", ["Radar (spider)", "Bar comparison"], horizontal=True)

    if len(players) < 2:
        st.info("Pick at least 2 players.")
    elif len(players) > 3:
        st.warning("Please keep it to 2 or 3 players for readability.")
    elif len(metrics) < 1:
        st.info("Choose at least one metric.")
    else:
        comp = df[df[player_col].isin(players)][[player_col] + metrics].copy()
        for m in metrics:
            comp[m] = pd.to_numeric(comp[m], errors="coerce")
        comp = comp.fillna(0)

        if chart_type.startswith("Radar") and len(metrics) >= 3:
            # Normalize vs entire filtered dataset (z-score -> squash 0..1)
            norm = {}
            for m in metrics:
                x = pd.to_numeric(df[m], errors="coerce")
                mu, sd = x.mean(), x.std(ddof=0)
                sd = sd if pd.notna(sd) and sd != 0 else 1.0
                norm[m] = (pd.to_numeric(comp[m], errors="coerce") - mu) / sd

            fig = go.Figure()
            for _, row in comp.iterrows():
                vals = [float(norm[m].loc[row.name]) for m in metrics]
                vals = [1/(1+np.exp(-v)) for v in vals]
                fig.add_trace(go.Scatterpolar(r=vals, theta=metrics, fill="toself", name=row[player_col]))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), showlegend=True, height=560)
            st.plotly_chart(fig, use_container_width=True)
        else:
            long = comp.melt(id_vars=[player_col], value_vars=metrics, var_name="Metric", value_name="Value")
            fig = px.bar(long, x="Metric", y="Value", color=player_col, barmode="group")
            fig.update_layout(height=560, xaxis_tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(comp.set_index(player_col), use_container_width=True, height=300)

# 5) Similarity finder (cosine)
with tab2:
    st.subheader("Find players with similar profiles (cosine similarity)")
    left, right = st.columns([2,3])
    with left:
        all_players = sorted([p for p in df[player_col].dropna().unique().tolist() if isinstance(p, str)])
        target = st.selectbox("Target player", all_players)
        defaults = [c for c in num_cols if "per 90" in c.lower()][:12] or num_cols[:12]
        sim_metrics = st.multiselect("Metrics for similarity", defaults, default=defaults[:8] or defaults)
        top_k = st.slider("Top K similar", 5, 50, 15)
        team_col = next((c for c in ["Team", "team", "Club"] if c in df.columns), None)
        exclude_same_team = st.checkbox("Exclude players from the same team", value=False) if team_col else False

    if not sim_metrics:
        st.info("Choose at least one metric to compute similarity.")
    else:
        M = df[[player_col] + sim_metrics].copy()
        for m in sim_metrics:
            M[m] = pd.to_numeric(M[m], errors="coerce")
        M = M.fillna(0)

        X = M[sim_metrics].to_numpy(dtype=float)
        mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, keepdims=True)
        sd[sd == 0] = 1.0
        Z = (X - mu) / sd

        norms = np.linalg.norm(Z, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        Z_norm = Z / norms
        S = Z_norm @ Z_norm.T

        idx_list = list(M[player_col])
        if target not in idx_list:
            st.error("Target player not found after filtering/NA handling.")
        else:
            idx = idx_list.index(target)
            M["Similarity"] = S[idx]
            res = M[M[player_col] != target].copy()

            if exclude_same_team and team_col:
                tteam = df.loc[df[player_col] == target, team_col].iloc[0] if len(df.loc[df[player_col] == target]) else None
                res = res.merge(df[[player_col, team_col]], on=player_col, how="left")
                res = res[res[team_col] != tteam]

            out_cols = [player_col, "Similarity"]
            if team_col:
                res = res.merge(df[[player_col, team_col]], on=player_col, how="left")
                out_cols.append(team_col)

            res = res.sort_values("Similarity", ascending=False).head(top_k)
            st.dataframe(res[out_cols], use_container_width=True, hide_index=True, height=420)
            fig = px.bar(res, x="Similarity", y=player_col, orientation="h", color=player_col)
            fig.update_layout(height=560, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

st.caption("If you see an error, try fewer metrics, ensure they are numeric, and check that you picked at least 2 players.")

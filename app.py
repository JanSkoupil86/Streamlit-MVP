
# app.py — Scouting App: Comparison & Similarity
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Scouting App — Comparison & Similarity", page_icon="⚽", layout="wide")

st.title("⚽ Football Scouting App")
st.caption("Upload your Wyscout-style CSV and compare players or find similar profiles.")

# --------------------
# Load data
# --------------------
file = st.file_uploader("Upload CSV", type=["csv"])
if file is None:
    st.info("Upload a CSV to begin.")
    st.stop()

df = pd.read_csv(file)
st.success(f"Loaded {df.shape[0]:,} rows × {df.shape[1]:,} columns")

# Identify a likely 'Player' column
string_cols = df.select_dtypes(include=["object"]).columns.tolist()
player_col = "Player" if "Player" in df.columns else ( "player" if "player" in df.columns else (string_cols[0] if string_cols else None))

if player_col is None:
    st.error("Couldn't find a text-like column for player names.")
    st.stop()

# Prepare numeric metrics
num_cols = df.select_dtypes(include=np.number).columns.tolist()
if not num_cols:
    st.error("No numeric columns found to compare.")
    st.stop()

# Sidebar filters (optional)
with st.sidebar:
    st.header("Filters")
    # Minutes threshold if present
    minutes_col = None
    for cand in ["Minutes", "minutes", "Minutes played", "Time played"]:
        if cand in df.columns:
            minutes_col = cand
            break
    if minutes_col:
        min_minutes = st.number_input("Minimum minutes", min_value=0, value=300, step=30)
        df = df[df[minutes_col].fillna(0) >= min_minutes]
    # Position filter
    pos_col = None
    for cand in ["Main Position", "Position", "Role"]:
        if cand in df.columns:
            pos_col = cand
            break
    if pos_col:
        positions = st.multiselect("Positions", sorted(df[pos_col].dropna().unique().tolist())) 
        if positions:
            df = df[df[pos_col].isin(positions)]

st.divider()

tab1, tab2 = st.tabs(["🔁 Player comparison", "🧭 Similarity finder"])

# --------------------
# Tab 1: Player comparison
# --------------------
with tab1:
    st.subheader("Select 2–3 players and compare")
    col_left, col_right = st.columns([2,3])
    with col_left:
        players = st.multiselect("Players", sorted(df[player_col].dropna().unique().tolist()), max_selections=3)
        metric_candidates = [c for c in num_cols if "per 90" in c.lower() or c.lower() in ("xg", "xa", "goals", "assists")]
        if not metric_candidates:
            metric_candidates = num_cols[:20]
        metrics = st.multiselect("Metrics (3–8 recommended)", metric_candidates, default=metric_candidates[:6] if len(metric_candidates)>=6 else metric_candidates)
        chart_type = st.radio("Chart type", ["Radar (spider)", "Bar comparison"], horizontal=True)
    if not players or len(metrics) < 1:
        st.info("Choose 2–3 players and at least one metric.")
    else:
        # Build comparison frame
        comp = df[df[player_col].isin(players)][[player_col] + metrics].copy()
        # Ensure numeric + fill
        for m in metrics:
            comp[m] = pd.to_numeric(comp[m], errors="coerce")
        comp = comp.dropna(subset=metrics, how="all").fillna(0)

        if chart_type.startswith("Radar") and len(metrics) >= 3:
            # Normalize each metric across the selection for a clean radar
            # Use z-score across the WHOLE filtered dataset to be fair
            norm = {}
            for m in metrics:
                x = pd.to_numeric(df[m], errors="coerce")
                mu, sd = x.mean(), x.std(ddof=0)
                sd = sd if pd.notna(sd) and sd != 0 else 1.0
                norm[m] = (pd.to_numeric(comp[m], errors="coerce") - mu) / sd

            fig = go.Figure()
            for _, row in comp.iterrows():
                values = [float(norm[m].loc[row.name]) for m in metrics]
                # squash to 0..1 for readability
                values = [1/(1+np.exp(-v)) for v in values]
                fig.add_trace(go.Scatterpolar(r=values, theta=metrics, fill='toself', name=row[player_col]))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0,1])),
                showlegend=True, height=560, margin=dict(l=10,r=10,t=30,b=10)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Long format for bar comparison
            long = comp.melt(id_vars=[player_col], value_vars=metrics, var_name="Metric", value_name="Value")
            fig = px.bar(long, x="Metric", y="Value", color=player_col, barmode="group")
            fig.update_layout(height=560, margin=dict(l=10,r=10,t=30,b=10), xaxis_tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(comp.set_index(player_col), use_container_width=True, height=300)

# --------------------
# Tab 2: Similarity finder
# --------------------
with tab2:
    st.subheader("Find players with similar profiles (cosine similarity)")
    col1, col2 = st.columns([2,3])
    with col1:
        target = st.selectbox("Target player", sorted(df[player_col].dropna().unique().tolist()))
        # Pick metrics
        default_metrics = [c for c in num_cols if "per 90" in c.lower()][:12]
        if not default_metrics:
            default_metrics = num_cols[:12]
        sim_metrics = st.multiselect("Metrics for similarity (normalize & compare)", default_metrics, default=default_metrics[:8] if len(default_metrics)>=8 else default_metrics)
        top_k = st.slider("Show top K similar", 5, 50, 15)
        exclude_same_team = False
        team_col = None
        for cand in ["Team", "team", "Club"]:
            if cand in df.columns:
                team_col = cand
                break
        if team_col:
            exclude_same_team = st.checkbox("Exclude players from the same team", value=False)
    if not sim_metrics:
        st.info("Choose at least one metric to compute similarity.")
    else:
        # Build matrix
        M = df[[player_col] + sim_metrics].copy()
        for m in sim_metrics:
            M[m] = pd.to_numeric(M[m], errors="coerce")
        M = M.dropna(subset=sim_metrics, how="all").fillna(0)

        # Standardize each metric (z-score)
        X = M[sim_metrics].to_numpy(dtype=float)
        mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, keepdims=True)
        sd[sd == 0] = 1.0
        Z = (X - mu) / sd

        # Cosine similarity
        def cos_sim_matrix(A):
            norms = np.linalg.norm(A, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            A_norm = A / norms
            return A_norm @ A_norm.T

        S = cos_sim_matrix(Z)

        # Find index of target
        idx_list = list(M[player_col])
        if target not in idx_list:
            st.error("Target player not found after filtering/NA handling.")
        else:
            idx = idx_list.index(target)
            sims = S[idx]
            M["Similarity"] = sims
            res = M[M[player_col] != target].copy()

            # Optional: exclude same team
            if exclude_same_team and team_col and team_col in df.columns:
                tteam = df.loc[df[player_col] == target, team_col].iloc[0] if len(df.loc[df[player_col] == target]) else None
                res = res.merge(df[[player_col, team_col]], on=player_col, how="left")
                res = res[res[team_col] != tteam]

            out_cols = [player_col, "Similarity"]
            if team_col:
                res = res.merge(df[[player_col, team_col]], on=player_col, how="left")
                out_cols.append(team_col)
            res = res.sort_values("Similarity", ascending=False).head(top_k)

            st.write(f"Players similar to **{target}** (by cosine similarity over {len(sim_metrics)} metrics):")
            st.dataframe(res[out_cols], use_container_width=True, hide_index=True, height=420)

            fig = px.bar(res, x="Similarity", y=player_col, orientation="h", color=player_col)
            fig.update_layout(height=560, margin=dict(l=10,r=10,t=30,b=10), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

# Footer
st.caption("Tips: choose per-90 metrics for fairer comparisons; use minutes/position filters in the sidebar.")

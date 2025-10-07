# app.py — Football Scouting App (Comparison + Similarity, safe build)
# Works on Streamlit Cloud with Python 3.12 + pinned deps (see note at bottom).

import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------
# Page setup
# ---------------------------
st.set_page_config(page_title="Scouting App — Comparison & Similarity", page_icon="⚽", layout="wide")
st.title("⚽ Football Scouting App")
st.caption("Upload your Wyscout-style CSV and compare players or find similar profiles.")

# ---------------------------
# Diagnostics (shows versions on screen to catch env issues)
# ---------------------------
with st.expander("Environment info (for debugging)", expanded=False):
    try:
        st.write({
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "plotly": px.__version__,
            "streamlit": st.__version__,
        })
    except Exception as e:
        st.error(f"Import/version check failed: {e}")

# ---------------------------
# Upload
# ---------------------------
file = st.file_uploader("Upload CSV", type=["csv"])
if file is None:
    st.info("Upload a CSV to begin.")
    st.stop()

# Read CSV safely
try:
    df = pd.read_csv(file, low_memory=False)
except Exception as e:
    # Try a more forgiving read if the first fails
    try:
        df = pd.read_csv(file, low_memory=False, engine="python")
    except Exception as e2:
        st.error(f"Failed to read CSV: {e}\nFallback error: {e2}")
        st.stop()

st.success(f"Loaded {df.shape[0]:,} rows × {df.shape[1]:,} columns")
st.caption("Tip: metrics that end with 'per 90' are best for fair comparisons.")

# ---------------------------
# Column detection
# ---------------------------
string_cols = df.select_dtypes(include=["object"]).columns.tolist()
player_col = next((c for c in ["Player", "player", "Name", "Full name"] if c in df.columns), None)
if player_col is None and string_cols:
    player_col = string_cols[0]  # fallback
if player_col is None:
    st.error("Couldn't find a text-like column for player names.")
    st.stop()

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if not num_cols:
    st.error("No numeric columns found to compare.")
    st.stop()

# ---------------------------
# Sidebar filters
# ---------------------------
with st.sidebar:
    st.header("Filters")

    minutes_col = next((c for c in ["Minutes", "minutes", "Minutes played", "Time played"] if c in df.columns), None)
    if minutes_col:
        try:
            min_minutes = st.number_input("Minimum minutes", min_value=0, value=300, step=30)
            df = df[pd.to_numeric(df[minutes_col], errors="coerce").fillna(0) >= min_minutes]
        except Exception:
            st.warning("Minutes filter skipped due to invalid column values.")

    pos_col = next((c for c in ["Main Position", "Position", "Role"] if c in df.columns), None)
    if pos_col:
        try:
            pos_vals = sorted([p for p in df[pos_col].dropna().unique().tolist() if isinstance(p, str)])
            chosen_pos = st.multiselect("Positions", pos_vals)
            if chosen_pos:
                df = df[df[pos_col].isin(chosen_pos)]
        except Exception:
            st.warning("Position filter skipped due to invalid values.")

st.divider()
tab1, tab2 = st.tabs(["🔁 Player comparison", "🧭 Similarity finder"])

# ---------------------------
# Helpers
# ---------------------------
def coerce_numeric_cols(frame: pd.DataFrame, cols):
    f = frame.copy()
    for c in cols:
        f[c] = pd.to_numeric(f[c], errors="coerce")
    return f.fillna(0)

def zscore_columns(frame: pd.DataFrame, cols):
    X = frame[cols].to_numpy(dtype=float)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return (X - mu) / sd

def cosine_similarity_matrix(A: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    A_norm = A / norms
    return A_norm @ A_norm.T

# ---------------------------
# Tab 1: Player comparison
# ---------------------------
with tab1:
    st.subheader("Select 2–3 players and compare")

    left, right = st.columns([2, 3])
    with left:
        all_players = sorted([p for p in df[player_col].dropna().unique().tolist() if isinstance(p, str)])
        players = st.multiselect("Players", all_players)  # keep API simple & compatible

        # Prefer "per 90" columns, then common attacking stats, else first 20 numerics
        metric_candidates = [c for c in num_cols if "per 90" in c.lower() or c.lower() in ("xg", "xa", "goals", "assists")]
        if not metric_candidates:
            metric_candidates = num_cols[:20]
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
        comp = coerce_numeric_cols(comp, metrics)

        if chart_type.startswith("Radar") and len(metrics) >= 3:
            # Normalize against the filtered dataset for fairness
            ref = coerce_numeric_cols(df[[m for m in metrics]].copy(), metrics)
            Z = zscore_columns(ref, metrics)
            ref_means = ref[metrics].mean().to_dict()
            ref_stds = ref[metrics].std(ddof=0).replace(0, 1.0).to_dict()

            # Build z-scores per selected player
            fig = go.Figure()
            for _, row in comp.iterrows():
                zs = []
                for m in metrics:
                    v = float(row[m])
                    z = (v - ref_means[m]) / (ref_stds[m] if ref_stds[m] != 0 else 1.0)
                    zs.append(1 / (1 + np.exp(-z)))  # squash to 0..1
                fig.add_trace(go.Scatterpolar(r=zs, theta=metrics, fill="toself", name=row[player_col]))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True, height=560)
            st.plotly_chart(fig, use_container_width=True)
        else:
            long = comp.melt(id_vars=[player_col], value_vars=metrics, var_name="Metric", value_name="Value")
            fig = px.bar(long, x="Metric", y="Value", color=player_col, barmode="group")
            fig.update_layout(height=560, xaxis_tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(comp.set_index(player_col), use_container_width=True, height=300)

# ---------------------------
# Tab 2: Similarity finder
# ---------------------------
with tab2:
    st.subheader("Find players with similar profiles (cosine similarity)")

    left, right = st.columns([2, 3])
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
        # Matrix with numeric metrics only
        M = df[[player_col] + sim_metrics].copy()
        M = coerce_numeric_cols(M, sim_metrics)

        # Standardize (z-score) across the whole filtered dataset
        Z = zscore_columns(M, sim_metrics)

        # Cosine similarity
        S = cosine_similarity_matrix(Z)

        names = list(M[player_col])
        if target not in names:
            st.error("Target player not found after filtering/NA handling.")
        else:
            idx = names.index(target)
            M["Similarity"] = S[idx]
            res = M[M[player_col] != target].copy()

            if exclude_same_team and team_col:
                try:
                    tteam = df.loc[df[player_col] == target, team_col].iloc[0]
                    res = res.merge(df[[player_col, team_col]], on=player_col, how="left")
                    res = res[res[team_col] != tteam]
                except Exception:
                    st.warning("Team-based exclusion skipped due to missing/invalid team data.")

            out_cols = [player_col, "Similarity"]
            if team_col:
                res = res.merge(df[[player_col, team_col]], on=player_col, how="left")
                out_cols.append(team_col)

            res = res.sort_values("Similarity", ascending=False).head(top_k)
            st.dataframe(res[out_cols], use_container_width=True, hide_index=True, height=420)

            fig = px.bar(res, x="Similarity", y=player_col, orientation="h", color=player_col)
            fig.update_layout(height=560, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

st.caption("If anything fails, try fewer metrics, ensure they are numeric, and verify at least 2 players are selected.")

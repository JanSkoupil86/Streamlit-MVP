import io
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Scouting App", layout="wide", page_icon="🧭")

# =========================================================
# Utility helpers
# =========================================================
def _dedupe_columns(cols):
    seen = {}
    out = []
    for c in cols:
        name = str(c).strip()
        if name not in seen:
            seen[name] = 0
            out.append(name)
        else:
            seen[name] += 1
            out.append(f"{name}.{seen[name]}")
    return out

def _ensure_series(s):
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return s.squeeze()

# =========================================================
# Data loading
# =========================================================
@st.cache_data(show_spinner=False)
def load_csv_from_bytes(data_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(data_bytes), low_memory=False)
    df.columns = _dedupe_columns(df.columns)
    return df

def upload_csv_ui() -> pd.DataFrame | None:
    st.markdown("#### Upload your Football Data CSV")
    uploaded = st.file_uploader(
        "Drag and drop file here",
        type=["csv"],
        accept_multiple_files=False,
        help="Limit 200MB per file • CSV",
        label_visibility="visible",
    )
    if uploaded is not None:
        try:
            df = load_csv_from_bytes(uploaded.getvalue())
            st.success(f"Loaded **{uploaded.name}** – {len(df):,} rows × {len(df.columns)} cols")
            return df
        except Exception as e:
            st.error(f"❌ Failed to read CSV: {e}")
    return None

# =========================================================
# Schema detection
# =========================================================
@st.cache_data(show_spinner=False)
def detect_schema(df: pd.DataFrame) -> Dict[str, List[str]]:
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    obj_cols = [c for c in df.columns if pd.api.types.is_object_dtype(df[c])]
    cat_cols = [c for c in obj_cols if df[c].nunique(dropna=True) <= max(50, len(df)//100)]
    name_like = [c for c in df.columns if any(k in c.lower() for k in ["player", "name"])]
    team_like = [c for c in df.columns if any(k in c.lower() for k in ["team", "club"])]
    league_like = [c for c in df.columns if any(k in c.lower() for k in ["league", "competition"])]
    position_like = [c for c in df.columns if "position" in c.lower()]
    minutes_like = [c for c in df.columns if any(k in c.lower() for k in ["minutes", "mins", "90s", "time played"])]
    return {
        "numeric": list(dict.fromkeys(num_cols)),
        "categorical": cat_cols,
        "player_cols": name_like or obj_cols[:1],
        "team_cols": team_like,
        "league_cols": league_like,
        "position_cols": position_like,
        "minutes_cols": minutes_like,
    }

def get_primary_col(df: pd.DataFrame, candidates: List[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

# =========================================================
# Numeric transforms
# =========================================================
def per90_transform(dff: pd.DataFrame, metrics: List[str], minutes_col: str | None) -> pd.DataFrame:
    if not minutes_col or minutes_col not in dff.columns:
        return dff
    out = dff.copy()
    m = pd.to_numeric(_ensure_series(out[minutes_col]), errors="coerce").replace(0, np.nan)
    for c in metrics:
        if c not in out.columns:
            continue
        col = pd.to_numeric(_ensure_series(out[c]), errors="coerce")
        out[c] = (col / m) * 90.0
    return out

def normalize_0_100(dff: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    out = dff.copy()
    for c in metrics:
        col = pd.to_numeric(_ensure_series(out[c]), errors="coerce")
        lo, hi = np.nanmin(col), np.nanmax(col)
        if not np.isfinite(lo) or not np.isfinite(hi):
            out[c] = np.nan
        elif hi - lo == 0:
            out[c] = 50.0
        else:
            out[c] = (col - lo) / (hi - lo) * 100.0
    return out

# =========================================================
# Radar chart builder
# =========================================================
def radar_figure(series_dict: Dict[str, pd.Series], title: str, fill: bool, colors: Dict[str, str] | None = None) -> go.Figure:
    metrics = series_dict[next(iter(series_dict))].index.tolist()
    fig = go.Figure()
    for player, s in series_dict.items():
        vals = s.fillna(0).tolist()
        r = vals + [vals[0]]
        theta = metrics + [metrics[0]]
        color = (colors or {}).get(player, None)
        fig.add_trace(
            go.Scatterpolar(
                r=r, theta=theta, name=player,
                mode="lines+markers",
                fill="toself" if fill else None,
                line=dict(width=2, color=color) if color else dict(width=2),
            )
        )
    fig.update_layout(
        title=title,
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        margin=dict(l=20, r=20, t=50, b=10),
        height=650,
    )
    return fig

# =========================================================
# Sidebar navigation
# =========================================================
def sidebar_nav() -> str:
    st.sidebar.markdown("## ☰ Menu")
    return st.sidebar.radio(
        "Navigate",
        options=[
            "Home",
            "Player Comparison",
            "Player Scout Report",
            "Player Clone",
            "Player Profiler",
            "Player Performance",
            "Player Screener",
            "Help",
        ],
        index=1,
        label_visibility="collapsed",
    )

# =========================================================
# Stat category builder
# =========================================================
def stat_categories_from_columns(num_cols: List[str]) -> Dict[str, List[str]]:
    def pick(*keys):
        kl = [k.lower() for k in keys]
        return [c for c in num_cols if any(k in c.lower() for k in kl)]
    return {
        "Comprehensive": num_cols[:20],
        "Shooting": pick("shot", "goal", "xg", "sca", "gca"),
        "Passing": pick("pass", "assist", "cross"),
        "Possession": pick("dribble", "carry", "progress"),
        "Defending": pick("tackle", "interception", "clearance", "block", "duel"),
        "Goalkeeping": pick("save", "keeper", "psxg"),
    }

# =========================================================
# Pages
# =========================================================
def page_home(df, schema):
    st.title("🏠 Home")
    st.write(f"Loaded dataset: **{len(df):,} rows × {len(df.columns)} cols**")
    with st.expander("Detected schema"):
        st.json(schema)

def page_player_comparison(df: pd.DataFrame, schema: Dict[str, List[str]]):
    st.title("⚖️ Player Comparison")
    st.caption("Compare players visually with radar charts.")

    player_col = get_primary_col(df, schema["player_cols"])
    if not player_col:
        st.warning("Couldn't find a player name column.")
        return

    all_players = sorted([p for p in df[player_col].dropna().unique().tolist() if str(p).strip() != ""])
    selected_players = st.multiselect("🌍 Select Players:", all_players, default=all_players[:2])

    with st.expander("🎨 Customize Player Colors"):
        colors = {p: st.color_picker(f"Color for {p}", "#1f77b4") for p in selected_players}

    num_cols = schema["numeric"]
    categories = stat_categories_from_columns(num_cols)
    cat_name = st.selectbox("📂 Select a Stat Category:", list(categories.keys()), index=0)
    metrics_default = categories[cat_name][:12]
    with st.expander("🔎 Filter Stats"):
        metrics = st.multiselect("Pick metrics to plot:", options=num_cols, default=metrics_default)

    c1, c2, c3 = st.columns(3)
    with c1:
        do_norm = st.checkbox("Normalize Values", True)
    with c2:
        do_per90 = st.checkbox("Per 90 Minutes", False)
    with c3:
        do_fill = st.checkbox("Enable Fill Color", True)

    if not selected_players or len(metrics) < 3:
        st.info("Select at least one player and 3+ metrics.")
        return

    keep_cols = [player_col] + metrics + schema.get("minutes_cols", [])
    dff = df[[c for c in keep_cols if c in df.columns]].copy()
    dff = dff[dff[player_col].isin(selected_players)]

    min_col = get_primary_col(dff, schema.get("minutes_cols", []))
    if do_per90:
        dff = per90_transform(dff, metrics, min_col)
    if do_norm:
        dff = normalize_0_100(dff, metrics)
    else:
        for c in metrics:
            dff[c] = pd.to_numeric(_ensure_series(dff[c]), errors="coerce")

    agg = dff.groupby(player_col, dropna=True)[metrics].mean(numeric_only=True).fillna(0)
    series_dict = {p: agg.loc[p] for p in agg.index if p in selected_players}
    fig = radar_figure(series_dict, f"Player Comparison – {cat_name}", do_fill, colors)
    st.plotly_chart(fig, use_container_width=True)

    try:
        import plotly.io as pio
        png_bytes = pio.to_image(fig, format="png", scale=2)
        st.download_button("⬇️ Download Chart", png_bytes, file_name="player_comparison.png", mime="image/png")
    except Exception:
        st.info("Install `kaleido` for PNG download (`pip install -U kaleido`).")

def placeholder(title):
    st.title(title)
    st.info("This page is a placeholder. Coming soon!")

# =========================================================
# App entry
# =========================================================
def main():
    df = upload_csv_ui()
    if df is None:
        st.stop()

    schema = detect_schema(df)
    page = sidebar_nav()

    if page == "Home":
        page_home(df, schema)
    elif page == "Player Comparison":
        page_player_comparison(df, schema)
    else:
        placeholder(page)

    st.markdown("---")
    st.caption("🧭 Scouting App • Upload CSV → Compare Players → Export Radar Chart")

if __name__ == "__main__":
    main()

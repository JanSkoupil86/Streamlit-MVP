import io
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Scouting App", layout="wide", page_icon="🧭")

# =========================================================
# Data loading & schema helpers
# =========================================================
@st.cache_data(show_spinner=False)
def load_csv_from_bytes(data_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(data_bytes), low_memory=False)

@st.cache_data(show_spinner=False)
def load_csv_from_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)

def upload_csv_ui() -> pd.DataFrame | None:
    st.markdown("#### Upload your Football Data CSV")
    up = st.file_uploader(
        "Drag and drop file here",
        type=["csv"],
        accept_multiple_files=False,
        help="Limit 200MB per file • CSV",
        label_visibility="visible",
    )
    if up is not None:
        data_bytes = up.getvalue()
        try:
            df = load_csv_from_bytes(data_bytes)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            return None
        df.columns = [c.strip() for c in df.columns]
        st.success(f"Loaded **{up.name}** – {len(df):,} rows × {len(df.columns)} cols")
        return df
    return None

@st.cache_data(show_spinner=False)
def detect_schema(df: pd.DataFrame) -> Dict[str, List[str]]:
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    obj_cols = [c for c in df.columns if pd.api.types.is_object_dtype(df[c])]
    cat_cols = [c for c in obj_cols if df[c].nunique(dropna=True) <= max(50, len(df)//100)]

    # Name-ish / entity-ish
    name_like = [c for c in df.columns if any(k in c.lower() for k in ["player", "name"])]
    team_like = [c for c in df.columns if any(k in c.lower() for k in ["team", "club"])]
    league_like = [c for c in df.columns if any(k in c.lower() for k in ["league", "competition"])]
    position_like = [c for c in df.columns if "position" in c.lower()]
    minutes_like = [c for c in df.columns if any(k in c.lower() for k in ["minutes", "mins", "min played", "90s"])]

    return {
        "numeric": num_cols,
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
# Radar helpers
# =========================================================
def per90_transform(dff: pd.DataFrame, metrics: List[str], minutes_col: str | None) -> pd.DataFrame:
    """Convert selected metrics to per90 using minutes_col when available."""
    if not minutes_col or minutes_col not in dff.columns:
        return dff
    out = dff.copy()
    m = pd.to_numeric(out[minutes_col], errors="coerce")
    m = m.replace(0, np.nan)  # avoid divide by zero
    for c in metrics:
        if c not in out.columns:
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce")
        out[c] = (out[c] / m) * 90.0
    return out

def normalize_0_100(dff: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    """Column-wise min-max to [0,100]. If col has zero variance, set 50."""
    out = dff.copy()
    for c in metrics:
        col = pd.to_numeric(out[c], errors="coerce")
        lo, hi = np.nanmin(col), np.nanmax(col)
        if not np.isfinite(lo) or not np.isfinite(hi):
            out[c] = np.nan
        elif hi - lo == 0:
            out[c] = 50.0
        else:
            out[c] = (col - lo) / (hi - lo) * 100.0
    return out

def radar_figure(series_dict: Dict[str, pd.Series],
                 title: str,
                 fill: bool,
                 colors: Dict[str, str] | None = None) -> go.Figure:
    """Build a Plotly polar radar chart from a dict of player_name -> metrics series."""
    metrics = series_dict[next(iter(series_dict))].index.tolist()
    fig = go.Figure()
    for player, s in series_dict.items():
        vals = s.fillna(0).tolist()
        # close loop
        r = vals + [vals[0]]
        theta = metrics + [metrics[0]]
        line_color = (colors or {}).get(player, None)
        fig.add_trace(
            go.Scatterpolar(
                r=r, theta=theta, name=player,
                mode="lines+markers",
                fill="toself" if fill else None,
                line=dict(width=2, color=line_color) if line_color else dict(width=2),
            )
        )
    fig.update_layout(
        title=title,
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        margin=dict(l=10, r=10, t=60, b=10),
        height=650,
    )
    return fig

# =========================================================
# UI bits
# =========================================================
def sidebar_nav() -> str:
    st.sidebar.markdown("## ☰ Menu")
    page = st.sidebar.radio(
        label="Navigate",
        options=[
            "Home",
            "Stats Dashboard",
            "Player Comparison",
            "Player Scout Report",
            "Player Clone",
            "Player Profiler",
            "Player Performance",
            "Player Screener",
            "Help",
        ],
        index=2,  # land on Player Comparison by default
        label_visibility="collapsed",
    )
    st.sidebar.markdown("---")
    return page

def stat_categories_from_columns(num_cols: List[str]) -> Dict[str, List[str]]:
    """Very light heuristic buckets based on common keywords."""
    def pick(*keys):
        keys_low = [k.lower() for k in keys]
        return [c for c in num_cols if any(k in c.lower() for k in keys_low)]

    return {
        "Comprehensive": num_cols[:20],  # first 20 numeric by order
        "Shooting": pick("shot", "xg", "npxg", "goals", "sca", "gca"),
        "Passing": pick("pass", "key pass", "assist", "xA", "cross", "prog pass"),
        "Possession/Progression": pick("carry", "progress", "dribble", "touch"),
        "Defending": pick("tackle", "interception", "clearance", "block", "press", "aerial", "duel"),
        "Goalkeeping": pick("save", "keeper", "ga", "psxg"),
    }

# =========================================================
# Pages
# =========================================================
def page_home(df, schema):
    st.title("🏠 Home")
    st.write("Welcome to the Scouting App. Use the left menu to explore your data.")
    st.write(f"Loaded dataset: **{len(df):,} rows × {len(df.columns)} cols**")
    with st.expander("Detected schema"):
        st.json(schema)

def page_player_comparison(df: pd.DataFrame, schema: Dict[str, List[str]]):
    st.title("⚖️ Player Comparison")
    st.caption("Compare two or more players on a radar chart.")

    # --- Top controls like your screenshot ---
    st.toggle("Search for Players", value=True, key="search_toggle", help="Enable type-to-search in the player selector.")
    player_col = get_primary_col(df, schema["player_cols"])
    if not player_col:
        st.warning("Couldn't find a player name column. Please ensure your CSV has a 'Player' or 'Name' column.")
        return

    # Player multi-select (searchable)
    players_all = sorted([p for p in df[player_col].dropna().unique().tolist() if str(p).strip() != ""])
    default_players = players_all[:2] if len(players_all) >= 2 else players_all
    selected_players = st.multiselect("🌐 Select Players:", options=players_all, default=default_players)

    # Color customization
    with st.expander("🎨 Customize Player Colors", expanded=False):
        chosen_colors: Dict[str, str] = {}
        for p in selected_players:
            chosen_colors[p] = st.color_picker(f"Color for {p}", value="#1f77b4")

    # Stat categories
    num_cols = [c for c in schema["numeric"] if c in df.columns]
    categories = stat_categories_from_columns(num_cols)
    cat_names = list(categories.keys())
    cat_name = st.selectbox("📂 Select a Stat Category:", options=cat_names, index=0)
    metrics_default = categories.get(cat_name, []) or num_cols[:12]

    # Filter Stats (choose axes)
    with st.expander("🔎 Filter Stats", expanded=False):
        metrics = st.multiselect(
            "Pick metrics to plot (3–18 recommended):",
            options=num_cols,
            default=metrics_default[:12],
        )

    # Options row
    c1, c2, c3 = st.columns(3)
    with c1:
        do_norm = st.checkbox("✅ Normalize Values", value=True, help="Scale each metric to a 0–100 range across the selected dataset.")
    with c2:
        do_per90 = st.checkbox("✅ Per 90 Minutes", value=False, help="Convert selected metrics to per 90 using the minutes column if available.")
    with c3:
        do_fill = st.checkbox("✅ Enable Fill Color", value=True, help="Fill the area under each player's radar.")

    if len(selected_players) < 1 or len(metrics) < 3:
        st.info("Select at least one player and 3+ metrics to render a radar chart.")
        return

    # --- Build working frame restricted to selected players ---
    dff = df.copy()
    # Only keep needed columns for speed/clarity
    keep_cols = [player_col] + metrics + schema.get("minutes_cols", [])
    keep_cols = [c for c in keep_cols if c in dff.columns]
    dff = dff[keep_cols]
    dff = dff[dff[player_col].isin(selected_players)]

    # Per 90 conversion if required
    min_col = get_primary_col(dff, schema.get("minutes_cols", []))
    if do_per90:
        dff = per90_transform(dff, metrics, min_col)

    # Normalize if required (on CURRENT subset only)
    if do_norm:
        dff = normalize_0_100(dff, metrics)
    else:
        # attempt to coerce numeric
        for c in metrics:
            dff[c] = pd.to_numeric(dff[c], errors="coerce")

    # Aggregate per player (if duplicates exist across seasons/teams)
    agg = dff.groupby(player_col, dropna=True)[metrics].mean(numeric_only=True)

    # Missing values → 0 for visualization, warn the user
    na_counts = agg.isna().sum(axis=1)
    has_na = int(na_counts.sum())
    if has_na:
        st.warning(f"Missing values were found in the player stats and have been replaced with 0 for visualization. ({has_na} missing metric values total across selected players)")

    # Prepare data for radar
    series_dict = {p: agg.loc[p].fillna(0) for p in agg.index}
    # If not normalized, bring radial axis to reasonable range
    radial_title = f"Player Comparison - {cat_name}"
    fig = radar_figure(series_dict, radial_title, fill=do_fill, colors=chosen_colors)

    st.plotly_chart(fig, use_container_width=True, theme="streamlit")

    # Download chart (PNG) using Plotly + kaleido
    try:
        import plotly.io as pio
        png_bytes = pio.to_image(fig, format="png", scale=2)
        st.download_button("⬇️ Download Chart", data=png_bytes, file_name="player_comparison.png", mime="image/png")
    except Exception:
        st.info("Install `kaleido` to enable PNG download: `pip install -U kaleido`")

def page_placeholder(title: str):
    st.title(title)
    st.info("This page is a placeholder. We can wire it up next.")

# =========================================================
# App bootstrap
# =========================================================
def main():
    # Upload OR load from path (sidebar)
    uploaded_df = upload_csv_ui()

    with st.sidebar:
        st.markdown("### Or load from path")
        default_path = "Wyscout_League_Export-new.csv"
        path = st.text_input("CSV path", value=default_path, help="If no file is uploaded, the app will try to load from this path.")

    if uploaded_df is not None:
        df = uploaded_df
    else:
        try:
            df = load_csv_from_path(path)
            df.columns = [c.strip() for c in df.columns]
            st.info(f"Loaded from path **{path}** – {len(df):,} rows × {len(df.columns)} cols")
        except Exception as e:
            st.error(f"Could not load data. Upload a CSV above or fix the path. Error: {e}")
            st.stop()

    schema = detect_schema(df)
    page = sidebar_nav()

    if page == "Home":
        page_home(df, schema)
    elif page == "Stats Dashboard":
        page_placeholder("📊 Stats Dashboard")
    elif page == "Player Comparison":
        page_player_comparison(df, schema)
    elif page == "Player Scout Report":
        page_placeholder("🕵️ Player Scout Report")
    elif page == "Player Clone":
        page_placeholder("🧬 Player Clone")
    elif page == "Player Profiler":
        page_placeholder("🧠 Player Profiler")
    elif page == "Player Performance":
        page_placeholder("📈 Player Performance")
    elif page == "Player Screener":
        page_placeholder("🧪 Player Screener")
    else:
        page_placeholder("❓ Help")

    st.markdown("---")
    st.caption("Tip: Use the sidebar menu to navigate. The Player Comparison page supports colors, categories, per-90, normalization, and PNG downloads.")

if __name__ == "__main__":
    main()

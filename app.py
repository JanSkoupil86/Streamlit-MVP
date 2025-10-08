
import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Tuple

st.set_page_config(page_title="Scouting App", layout="wide")

@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    # Strip whitespace from column names
    df.columns = [c.strip() for c in df.columns]
    return df

@st.cache_data(show_spinner=False)
def detect_schema(df: pd.DataFrame):
    # Heuristics to classify columns
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    obj_cols = [c for c in df.columns if pd.api.types.is_object_dtype(df[c])]
    # Categorical: low-cardinality object columns
    cat_cols = [c for c in obj_cols if df[c].nunique(dropna=True) <= max(50, len(df)//100)]
    # Try to detect dates
    date_cols = []
    for c in df.columns:
        s = df[c]
        if s.isna().all(): 
            continue
        if pd.api.types.is_datetime64_any_dtype(s):
            date_cols.append(c)
            continue
        # attempt parse on a small sample
        sample = s.dropna().astype(str).head(50)
        try:
            pd.to_datetime(sample, errors="raise", utc=True, infer_datetime_format=True)
            date_cols.append(c)
        except Exception:
            pass
    # Try to identify name-ish/ID-ish columns
    name_like = [c for c in df.columns if any(k in c.lower() for k in ["player", "name"])]
    team_like = [c for c in df.columns if any(k in c.lower() for k in ["team", "club"])]
    league_like = [c for c in df.columns if any(k in c.lower() for k in ["league", "competition"])]
    position_like = [c for c in df.columns if "position" in c.lower()]
    foot_like = [c for c in df.columns if "foot" in c.lower()]
    minutes_like = [c for c in df.columns if any(k in c.lower() for k in ["minutes", "mins", "min played", "90s"])]
    age_like = [c for c in df.columns if "age" in c.lower()]
    return {
        "numeric": num_cols,
        "categorical": cat_cols,
        "dates": date_cols,
        "player_cols": name_like,
        "team_cols": team_like,
        "league_cols": league_like,
        "position_cols": position_like,
        "foot_cols": foot_like,
        "minutes_cols": minutes_like,
        "age_cols": age_like,
    }

def shortlist_filters_ui(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    st.sidebar.header("Filters")
    dff = df.copy()

    # Basic entity filters (only render if present)
    def multi_filter(col_candidates: List[str], label: str):
        for c in col_candidates:
            if c in dff.columns:
                values = sorted([v for v in dff[c].dropna().unique().tolist() if v != ""])
                if len(values) and len(values) <= 200:
                    sel = st.sidebar.multiselect(label, values, default=[])
                    if sel:
                        return dff[dff[c].isin(sel)]
                return dff
        return dff

    dff = multi_filter(schema["league_cols"], "League")
    dff = multi_filter(schema["team_cols"], "Team/Club")
    dff = multi_filter(schema["position_cols"], "Position")
    dff = multi_filter(schema["foot_cols"], "Preferred Foot")
    dff = multi_filter(schema["categorical"], "Other category")

    # Age filter
    age_col = next((c for c in schema["age_cols"] if c in dff.columns), None)
    if age_col:
        amin, amax = int(np.nanmin(pd.to_numeric(dff[age_col], errors="coerce"))), int(np.nanmax(pd.to_numeric(dff[age_col], errors="coerce")))
        if amin < amax and amax - amin < 60:
            lo, hi = st.sidebar.slider("Age range", min_value=amin, max_value=amax, value=(amin, amax))
            dff = dff[(pd.to_numeric(dff[age_col], errors="coerce") >= lo) & (pd.to_numeric(dff[age_col], errors="coerce") <= hi)]

    # Minutes threshold (if available)
    min_col = next((c for c in schema["minutes_cols"] if c in dff.columns), None)
    if min_col:
        mmin, mmax = float(np.nanmin(pd.to_numeric(dff[min_col], errors="coerce"))), float(np.nanmax(pd.to_numeric(dff[min_col], errors="coerce")))
        default_min = int(np.clip(mmax * 0.2, 0, mmax))  # default to 20% of max
        thr = st.sidebar.number_input(f"Min minutes ({min_col})", min_value=0, max_value=int(mmax), value=int(default_min), step=10)
        dff = dff[pd.to_numeric(dff[min_col], errors="coerce") >= thr]

    return dff

def percentile_shortlist_ui(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    st.sidebar.header("Percentile Shortlist")
    num_cols = [c for c in schema["numeric"] if c in df.columns]
    if not num_cols:
        st.info("No numeric columns detected for percentile filtering.")
        return df

    # Choose metrics
    metrics = st.sidebar.multiselect("Metrics (choose 1–5)", num_cols[:200], default=num_cols[:2], max_selections=5)
    if not metrics:
        return df

    # Percentile threshold
    p = st.sidebar.slider("Minimum percentile across selected metrics", min_value=50, max_value=99, value=80, step=1)

    # Compute row-wise min-percentile across selected metrics
    # Convert to numeric safely
    dff = df.copy()
    for c in metrics:
        dff[c] = pd.to_numeric(dff[c], errors="coerce")

    # Population for percentiles = current filtered set
    ref = dff[metrics].astype(float)

    # per-column percentile ranks
    pr = ref.rank(pct=True).fillna(0.0)
    # min across selected metrics (strict AND)
    dff["_min_pct"] = pr.min(axis=1) * 100.0
    out = dff[dff["_min_pct"] >= p].copy()
    return out

def prettify(df: pd.DataFrame, schema: dict) -> Tuple[pd.DataFrame, List[str]]:
    # Choose display columns
    # Try to put player/age/team/position first if present
    front = []
    for group in ["player_cols", "team_cols", "league_cols", "position_cols", "age_cols", "foot_cols"]:
        for c in schema[group]:
            if c in df.columns and c not in front:
                front.append(c)
    # Add a handful of numeric metrics
    numeric_sample = [c for c in schema["numeric"] if c in df.columns and c not in front][:10]
    cols = front + numeric_sample
    cols = [c for c in cols if c in df.columns]
    # Avoid duplicate columns
    cols = list(dict.fromkeys(cols))
    return df[cols], cols

# ---- App ----
st.title("🧭 Scouting App – Wyscout Export")

# Data path input
default_path = "Wyscout_League_Export-new.csv"
path = st.sidebar.text_input("CSV path", value=default_path, help="Use the default if running in the provided workspace.")
df = load_data(path)
schema = detect_schema(df)

with st.expander("Detected schema (click to expand)"):
    st.json(schema)

# Filters
filtered = shortlist_filters_ui(df, schema)

# Percentile shortlist on the filtered set
shortlisted = percentile_shortlist_ui(filtered, schema)

# Display
tab1, tab2 = st.tabs(["Filtered dataset", "Shortlist"])

with tab1:
    view1, cols1 = prettify(filtered, schema)
    st.caption(f"{len(view1):,} rows after filters")
    st.dataframe(view1, use_container_width=True, hide_index=True)
    st.download_button("Download filtered CSV", data=view1.to_csv(index=False).encode("utf-8"), file_name="filtered.csv", mime="text/csv")

with tab2:
    if "_min_pct" in shortlisted.columns:
        view2, cols2 = prettify(shortlisted, schema)
        view2 = view2.copy()
        view2.insert(0, "Min %ile (across chosen metrics)", shortlisted["_min_pct"].round(1).values)
    else:
        view2, cols2 = prettify(shortlisted, schema)
    st.caption(f"{len(view2):,} players meet the percentile threshold")
    st.dataframe(view2, use_container_width=True, hide_index=True)
    st.download_button("Download shortlist CSV", data=view2.to_csv(index=False).encode("utf-8"), file_name="shortlist.csv", mime="text/csv")

st.markdown("---")
st.caption("Tip: Use the sidebar to set minutes threshold, pick leagues/positions, then choose metrics and a minimum percentile to generate a shortlist.")

# streamlit_app.py
# -*- coding: utf-8 -*-
import os, json, pickle, logging, glob, zipfile, pathlib
import numpy as np, pandas as pd, geopandas as gpd
import streamlit as st, pydeck as pdk
from shapely.ops import unary_union
from app_utils import ensure_bundle  # downloads & extracts when BUNDLE_URL is provided
# --- Hex → Sector lookup (uses ops/patrol_sectors.geojson) ---
def find_sector_for_hex(sel_hex: str, sel_date: pd.Timestamp, sel_dp: str):
    if not (os.path.exists(SECT_GJ) and sel_hex and sel_dp):
        return None
    try:
        sg = gpd.read_file(SECT_GJ).to_crs(4326)
    except Exception:
        return None

    # normalize types
    if "date" in sg.columns:
        with pd.option_context("future.no_silent_downcasting", True):
            try: sg["date"] = pd.to_datetime(sg["date"]).dt.normalize()
            except: pass

    # filter to the selected day
    mask = (sg.get("daypart") == sel_dp)
    if "date" in sg.columns:
        mask &= (sg["date"] == pd.to_datetime(sel_date).normalize())
    today_sectors = sg.loc[mask].copy()
    if today_sectors.empty:
        return None

    # hex geometry
    try:
        hex_geom = grid.loc[grid["h3"].astype(str) == sel_hex, "geometry"].iloc[0]
    except Exception:
        return None

    hit = today_sectors.loc[today_sectors.intersects(hex_geom)]
    if hit.empty:
        return None
    return hit.iloc[0]  # GeoSeries row with sector_id, geometry, etc.


# Quiet SHAP noise
logging.getLogger("shap").setLevel(logging.ERROR)

# ---------- BUNDLE LOCATION (robust) ----------
def _extract_local_zip_if_present() -> str | None:
    """
    If a bundle ZIP (streamlit_bundle_full_*.zip) is present in the repo working dir,
    extract it under ./data/bundle_local and return that absolute path. Otherwise None.
    """
    candidates = (
        glob.glob("streamlit_bundle_full_*.zip") +
        glob.glob("/mount/src/*/streamlit_bundle_full_*.zip") +
        glob.glob("/app/*/streamlit_bundle_full_*.zip")
    )
    if not candidates:
        return None
    zip_path = candidates[0]
    target = pathlib.Path("./data/bundle_local").resolve()
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(target)
    return str(target)

# Priority: explicit env (local/dev) → secrets/env BUNDLE_URL → local zip fallback
BUNDLE_URL = (os.environ.get("BUNDLE_URL", "") or st.secrets.get("BUNDLE_URL", "")).strip()

if os.environ.get("MVP_BASE"):
    BASE = os.environ["MVP_BASE"]
elif BUNDLE_URL:
    BASE = ensure_bundle(BUNDLE_URL)  # downloads & caches under ./data/bundle
else:
    # Last-chance fallback: a zip committed/uploaded next to the app (not recommended, but handy)
    local = _extract_local_zip_if_present()
    if local:
        BASE = local
    else:
        st.set_page_config(page_title="Mohammadpur Spatial Risk", layout="wide")
        st.error(
            "Missing bundle. Provide one of the following:\n\n"
            "1) **Secrets**: add\n"
            '```toml\nBUNDLE_URL = "https://.../streamlit_bundle_full_YYYYMMDDTHHMMSSZ.zip"\n```\n'
            "2) **Advanced settings → Environment variables**: set `BUNDLE_URL`\n"
            "3) **Env var** `MVP_BASE` pointing to an unzipped bundle directory\n"
            "4) Place a local ZIP named `streamlit_bundle_full_*.zip` in the repo root\n"
        )
        st.stop()

# ---------- PATHS ----------
MODEL_CAL = os.path.join(BASE, "models", "lgbm_isotonic.pkl")
MODEL_RAW = os.path.join(BASE, "models", "lgbm_raw.pkl")  # optional
META_JSON = os.path.join(BASE, "models", "features.json")
FEATS_PARQ= os.path.join(BASE, "features", "training.parquet")
GRID_GJ   = os.path.join(BASE, "h3_grid.geojson")
EVAL_SUM  = os.path.join(BASE, "metrics", "eval_summary.csv")
HRK_SW    = os.path.join(BASE, "metrics", "hrk_sweep.csv")
SECT_CSV  = os.path.join(BASE, "ops", "patrol_sectors.csv")
TOPK_CSV  = os.path.join(BASE, "ops", "topk_hexes.csv")
SECT_GJ   = os.path.join(BASE, "ops", "patrol_sectors.geojson")

# ---------- PAGE ----------
st.set_page_config(page_title="Mohammadpur Spatial Risk", layout="wide")
st.markdown(
    '''<div style="padding:10px;border-left:6px solid #ffcc00;background:#fff9e6">
<b>Policy</b>: <i>Spatial risk only.</i> <b>No person-level profiling.</b> Decision support, not a warrant.
</div>''',
    unsafe_allow_html=True
)
st.sidebar.caption(f"Bundle base: {BASE}")

# ---------- HELPERS ----------
def union_all_safe(geom_series):
    try:
        return geom_series.union_all()
    except Exception:
        return unary_union(list(geom_series))

@st.cache_data(show_spinner=False)
def load_grid(path):
    g = gpd.read_file(path).to_crs(4326)
    g["h3"] = g["h3"].astype(str)
    return g

@st.cache_resource(show_spinner=False)
def load_models():
    with open(MODEL_CAL, "rb") as f:
        clf_cal = pickle.load(f)
    clf_raw = None
    if os.path.exists(MODEL_RAW):
        try:
            with open(MODEL_RAW, "rb") as f:
                clf_raw = pickle.load(f)
        except Exception:
            pass
    with open(META_JSON, "r") as f:
        meta = json.load(f)
    return clf_cal, clf_raw, meta

@st.cache_data(show_spinner=False)
def parquet_available_columns():
    try:
        import pyarrow.parquet as pq
        return list(pq.ParquetFile(FEATS_PARQ).schema.names)
    except Exception:
        return list(pd.read_parquet(FEATS_PARQ, engine="pyarrow").columns)

@st.cache_data(show_spinner=False)
def months_from_ops_or_parquet():
    for p in (SECT_CSV, TOPK_CSV):
        if os.path.exists(p):
            d = pd.read_csv(p, parse_dates=["date"])
            return sorted(d["date"].dt.to_period("M").astype(str).unique())
    d = pd.read_parquet(FEATS_PARQ, columns=["date"], engine="pyarrow")
    return sorted(pd.to_datetime(d["date"]).dt.to_period("M").astype(str).unique())

@st.cache_data(show_spinner=False)
def load_month_slice(month_str, feat_cols, extra_cols):
    p = pd.Period(month_str, "M")
    start = pd.Timestamp(p.start_time).to_datetime64()
    end   = (pd.Timestamp(p.end_time) + pd.Timedelta(seconds=1)).to_datetime64()
    avail = set(parquet_available_columns())
    cols  = [c for c in (set(feat_cols) | set(extra_cols)) if c in avail]
    if "date" not in cols: cols.append("date")
    if "h3"   not in cols: cols.append("h3")
    X = pd.read_parquet(
        FEATS_PARQ, engine="pyarrow", columns=cols,
        filters=[("date", ">=", start), ("date", "<", end)]
    )
    X["date"] = pd.to_datetime(X["date"]).dt.normalize()
    return X

def infer_dayparts(df):
    if "daypart" in df.columns:
        return sorted(df["daypart"].astype(str).unique()), "column"
    dps = [c for c in df.columns if c.startswith("daypart_")]
    return sorted([c.replace("daypart_", "") for c in dps]), "dummies"

# ---------- SANITY ----------
for pth, label in [(GRID_GJ, "grid"), (MODEL_CAL, "model"), (META_JSON, "meta"), (FEATS_PARQ, "features")]:
    if not os.path.exists(pth):
        st.error(f"Missing {label}: {pth}")
        st.stop()

grid = load_grid(GRID_GJ)
clf_cal, clf_raw, meta = load_models()
feat_cols   = meta.get("feat_cols", [])
used_nowcast= bool(meta.get("used_nowcast", False))

# ---------- SIDEBAR ----------
months = months_from_ops_or_parquet()
if not months:
    st.error("No months found.")
    st.stop()

# Month stays the same
sel_month = st.sidebar.selectbox("Month", options=months, index=len(months)-1)

# How many places to flag
TOPK = st.sidebar.slider("Places to cover (%)", 1, 10, 5)
st.sidebar.caption(
    "We flag the top **K%** of grid cells for each time-of-day, every day. "
    "Example: 5% ≈ the riskiest 1 in 20 cells."
)

# Risk score used
risk_choice = st.sidebar.radio(
    "Risk score used",
    ["Model only (recommended)", "Model + recent activity (60/40)"],
    index=0
)
risk_mode = "pro_only" if risk_choice.startswith("Model only") else "blend"
st.sidebar.caption(
    "**Model only** = calibrated probability from the model.  "
    "**Model + recent** = 60% model + 40% short-term activity."
)


# ---------- LOAD THIS MONTH ----------
with st.spinner(f"Loading {sel_month}…"):
    maybe_dp = ["daypart", "daypart_00-06", "daypart_06-12", "daypart_12-18", "daypart_18-24"]
    extra    = ["date", "h3", "ri_norm"] + maybe_dp
    Xmon     = load_month_slice(sel_month, feat_cols, extra)

dayparts, dp_mode = infer_dayparts(Xmon)
if not dayparts:
    dayparts, dp_mode = ["12-18"], "dummies"

sel_dp   = st.sidebar.selectbox("Daypart", options=dayparts, index=min(2, len(dayparts)-1))
dates_in = sorted(Xmon["date"].unique())
if not dates_in:
    st.warning("No dates in this month.")
    st.stop()
sel_date = st.sidebar.selectbox("Date", options=[d.date() for d in dates_in])
sel_date = pd.Timestamp(sel_date)

# ---------- FILTER & ALIGN ----------
Df = Xmon[Xmon["date"] == sel_date].copy()
if dp_mode == "column":
    Df = Df[Df["daypart"].astype(str) == sel_dp]
else:
    dpcol = f"daypart_{sel_dp}"
    if dpcol in Df.columns:
        Df = Df[Df[dpcol] == 1]

for c in feat_cols:
    if c not in Df.columns:
        Df[c] = 0.0
Df = Df.drop(columns=[c for c in Df.columns if c not in feat_cols and c not in {"date", "h3", "ri_norm"}], errors="ignore")
rep  = Df[["date", "h3"] + (["ri_norm"] if "ri_norm" in Df.columns else [])].copy()
Xsel = Df[feat_cols].copy()

# ---------- PREDICT ----------
with st.spinner("Scoring…"):
    pro = clf_cal.predict_proba(Xsel)[:, 1]
ri   = rep["ri_norm"].to_numpy() if "ri_norm" in rep.columns else np.zeros_like(pro)
risk = pro if (risk_mode == "pro_only" or used_nowcast) else (0.6 * pro + 0.4 * ri)

pred = pd.DataFrame({"h3": rep["h3"].astype(str), "risk": risk})
top_per_h3 = pred.groupby("h3", as_index=False)["risk"].max().sort_values("risk", ascending=False)
k = max(1, int(len(top_per_h3) * TOPK / 100.0))
top = top_per_h3.head(k)

st.subheader(f"Top-{TOPK}% — {sel_date.date()} {sel_dp} (n={len(top)})")
dl = top.copy()
dl.insert(0, "date", sel_date.date())
dl.insert(1, "daypart", sel_dp)
st.download_button("Download Top-K hexes (CSV)", dl.to_csv(index=False).encode("utf-8"),
                   file_name=f"topk_{sel_date.date()}_{sel_dp.replace(':','-')}.csv")

# ---------- MAP ----------
use_folium = st.toggle("Use Folium map (fallback)", value=False)

try:
    joined = grid.merge(top, on="h3", how="inner")
    if joined.empty:
        st.info("No Top-K hexes for this selection.")
    else:
        # Ensure numeric & normalize
        joined["risk"] = pd.to_numeric(joined["risk"], errors="coerce").fillna(0.0)
        rmin, rmax = float(joined["risk"].min()), float(joined["risk"].max())
        denom = (rmax - rmin) if (rmax > rmin) else 1.0
        joined["risk_norm"] = (joined["risk"] - rmin) / denom

        # Rank (1 = highest risk for this day/daypart)
        joined = joined.sort_values("risk", ascending=False).reset_index(drop=True)
        joined["rank"] = joined.index + 1

        # Precompute color in Python (no JS in deck.gl accessors)
        alpha = (110 + (joined["risk_norm"] * 130)).round().astype(int).clip(0, 255)
        joined["fill_rgba"] = [[255, 136, 0, int(a)] for a in alpha]

        # Map center
        u = union_all_safe(joined.geometry)
        center = [float(u.centroid.y), float(u.centroid.x)]

        if not use_folium:
            import json, pydeck as pdk
            geojson_dict = json.loads(joined.to_json())  # pass dict, not string

            layer = pdk.Layer(
                "GeoJsonLayer",
                data=geojson_dict,
                filled=True,
                stroked=True,
                opacity=0.7,
                get_fill_color="properties.fill_rgba",   # reads from properties
                get_line_color=[0, 0, 0, 180],
                line_width_min_pixels=1,
                pickable=True,
                auto_highlight=True,
            )

            # Tooltip shows exact H3 id + risk + rank

            tooltip = {
                "html": "<b>H3:</b> {h3}<br/>"
                        "<b>Risk score:</b> {risk}<br/>"
                        "<b>Priority:</b> {rank}",
                "style": {"backgroundColor": "rgba(0,0,0,0.7)", "color": "white"}
            }
            

            st.pydeck_chart(
                pdk.Deck(
                    layers=[layer],
                    initial_view_state=pdk.ViewState(
                        latitude=center[0], longitude=center[1], zoom=13
                    ),
                    tooltip=tooltip,
                )
            )
        else:
            import folium
            from streamlit.components.v1 import html
            m = folium.Map(location=center, zoom_start=13, tiles="cartodbpositron")

            # Tooltip + popup with H3 id, risk, rank
            folium.GeoJson(
                joined.to_json(),
                name="TopK",
                style_function=lambda feat: {
                    "color": "#ff8800",
                    "weight": 1.5,
                    "fillOpacity": float(feat["properties"].get("risk_norm", 0))*0.7 + 0.3,
                },
                highlight_function=lambda feat: {"weight": 3, "color": "#000"},
                tooltip=folium.GeoJsonTooltip(
                    fields=["h3", "risk", "rank"],
                    aliases=["H3", "Risk score", "Priority"],
                    localize=True, sticky=True
                ),
                popup=folium.GeoJsonPopup(
                    fields=["h3", "risk", "rank"],
                    aliases=["H3", "Risk score", "Priority"],
                    localize=True
                )
            ).add_to(m)

            html(m._repr_html_(), height=520)

except Exception as e:
    st.warning(f"Map render failed: {e}")
    st.dataframe(top.head(20))



# ---------- VIGILANCE BY DAYPART (month summary) ----------
st.subheader("Vigilance by daypart (month summary)")

daily_budget_min = st.number_input(
    "Total patrol minutes per day (across all dayparts)",
    min_value=60, max_value=1440, value=480, step=30
)

method = st.radio(
    "Compute share using",
    ["Share of **month** total risk", "Average **daily** share"],
    index=0, horizontal=True
)

def _month_filter(df, month_str):
    return df[df["date"].dt.to_period("M").astype(str) == month_str].copy()

def _plot_bar(series, label):
    if series.empty:
        st.info("No data for the selected month.")
        return
    st.bar_chart(series.sort_values(ascending=False), height=240)
    st.caption(label)

summary = None
source_note = ""

if os.path.exists(SECT_CSV):
    # Best source: sector risk per (date, daypart)
    sect_all = pd.read_csv(SECT_CSV, parse_dates=["date"])
    sect_m   = _month_filter(sect_all, sel_month)
    # risk per (date, daypart)
    dd = sect_m.groupby(["date", "daypart"], as_index=False)["risk_sum"].sum()

    if method.startswith("Share of"):
        # Month-level aggregation (no per-day normalization)
        month_totals = dd.groupby("daypart", as_index=False)["risk_sum"].sum()
        month_totals = month_totals.rename(columns={"risk_sum": "risk_month_total"})
        month_totals["share"] = month_totals["risk_month_total"] / month_totals["risk_month_total"].sum()
        g = month_totals.set_index("daypart")["share"]
        label = "Share of **month** total risk by daypart"
    else:
        # Per-day normalization first, then average share across the month
        dd["daily_total"] = dd.groupby("date")["risk_sum"].transform("sum").replace(0, np.nan)
        dd["share"]       = dd["risk_sum"] / dd["daily_total"]
        g = dd.groupby("daypart")["share"].mean().fillna(0.0)
        label = "Average **daily** share by daypart (each day sums to 100%)"

    rec_minutes = (g * daily_budget_min).round(0).astype(int)
    summary = pd.DataFrame({
        "avg_share_%": (g * 100).round(1),
        "recommended_min_per_day": rec_minutes
    }).sort_values("avg_share_%", ascending=False)
    source_note = "Source: patrol_sectors.csv"

elif os.path.exists(TOPK_CSV):
    # Fallback: aggregate Top-K risk per (date, daypart)
    tk = pd.read_csv(TOPK_CSV, parse_dates=["date"])
    tkm = _month_filter(tk, sel_month)
    dd  = tkm.groupby(["date","daypart"], as_index=False)["risk"].sum()

    if method.startswith("Share of"):
        month_totals = dd.groupby("daypart", as_index=False)["risk"].sum()
        month_totals = month_totals.rename(columns={"risk": "risk_month_total"})
        month_totals["share"] = month_totals["risk_month_total"] / month_totals["risk_month_total"].sum()
        g = month_totals.set_index("daypart")["share"]
        label = "Share of **month** total Top-K risk by daypart"
    else:
        dd["daily_total"] = dd.groupby("date")["risk"].transform("sum").replace(0, np.nan)
        dd["share"]       = dd["risk"] / dd["daily_total"]
        g = dd.groupby("daypart")["share"].mean().fillna(0.0)
        label = "Average **daily** share by daypart (Top-K fallback)"

    rec_minutes = (g * daily_budget_min).round(0).astype(int)
    summary = pd.DataFrame({
        "avg_share_%": (g * 100).round(1),
        "recommended_min_per_day": rec_minutes
    }).sort_values("avg_share_%", ascending=False)
    source_note = "Source: topk_hexes.csv (fallback)"

else:
    st.info("No sectors or Top-K files found; add ops/patrol_sectors.csv or ops/topk_hexes.csv to enable this summary.")

if summary is not None:
    st.dataframe(summary.rename_axis("daypart"), use_container_width=True)
    _plot_bar(summary["recommended_min_per_day"], "Recommended patrol minutes per day by daypart")
    st.caption(source_note + ". Use this to apportion daily patrol time across dayparts.")

# ---------- SECTORS ----------
st.subheader("Patrol sectors")

SECT_OK = os.path.exists(SECT_CSV) and os.path.exists(SECT_GJ)
if SECT_OK:
    sect = pd.read_csv(SECT_CSV, parse_dates=["date"])
    s = sect[(sect["date"] == sel_date) & (sect["daypart"] == sel_dp)].copy()
    if s.empty:
        st.info("No exported sectors for this day/daypart.")
    else:
        # Friendlier labels/derived fields
        s["Priority"] = s["sector_rank"].astype(int)
        s["Date"]     = pd.to_datetime(s["sector_id"].str.slice(0, 8), format="%Y%m%d").dt.date
        s["Time"]     = s["sector_id"].str.extract(r"_(\d{2}-\d{2})_")[0]
        s["Label"]    = s.apply(lambda r: f"{r['Date']} {r['Time']} • S{r['Priority']}", axis=1)

        # Risk level bands relative to this day/time
        q50, q80, q95 = s["risk_sum"].quantile([0.50, 0.80, 0.95]).tolist()
        def band(v):
            if v <= q50: return "Low"
            if v <= q80: return "Medium"
            if v <= q95: return "High"
            return "Very high"
        s["Risk level"]       = s["risk_sum"].apply(band)
        s["Risk score"]       = s["risk_sum"].round(3)
        s["Risk share %"]     = (100 * s["risk_sum"] / s["risk_sum"].sum()).round(1)
        s["Cells"]            = s["n_hex"].astype(int)
        s["Area (ha)"]        = s["area_ha"].round(1)
        s["Suggested minutes"]= s["dwell_min"].astype(int)

        # Display table
        s_disp = s.sort_values("Priority")[
            ["Label","Priority","Risk level","Risk score","Risk share %","Cells","Area (ha)","Suggested minutes"]
        ].rename(columns={"Label":"Sector"})

        def _color_risk(col):
            colors = {"Low":"#E3F2FD","Medium":"#FFF3E0","High":"#FFEBEE","Very high":"#FFCDD2"}
            return [f"background-color: {colors.get(v,'')}" for v in col]
        try:
            st.dataframe(s_disp.style.apply(_color_risk, subset=["Risk level"]), use_container_width=True)
        except Exception:
            st.dataframe(s_disp, use_container_width=True)

        # 👉 Glossary (add here)
        with st.expander("What do these terms mean?"):
            st.markdown("""
- **Grid cell (H3)**: a small hexagon tile (~200 m) covering the map.
- **Cell risk score**: how risky this cell is today for the selected time-of-day.
- **Patrol sector**: a group of adjacent flagged cells merged into one patrol area.
- **Priority**: rank of a sector among today’s sectors (1 = most urgent).
- **Suggested minutes**: recommended time to spend inside a sector for this time-of-day.
""")

        # Downloads: friendly + raw
        st.download_button(
            "Download sectors (friendly CSV)",
            data=s_disp.to_csv(index=False).encode("utf-8"),
            file_name=f"patrol_sectors_{sel_date.date()}_{sel_dp.replace(':','-')}_friendly.csv",
        )
        st.download_button(
            "Download sectors (raw CSV)",
            data=open(SECT_CSV, "rb").read(),
            file_name="patrol_sectors_raw.csv",
        )
else:
    st.info("ops/patrol_sectors.* not found (optional).")




# ---------- WHY HERE ----------
st.subheader("Why here?")

import re, numpy as np

def friendly_label(col: str) -> str | None:
    if col == "ri_norm": return "Recent activity score (0–1)"
    if col == "neighbor_lag7d": return "Nearby incidents in last 7 days (avg)"
    if col == "same_daypart_last_week": return "Same time last week (flag)"
    if col == "days_since_last": return "Days since last incident"
    if col == "neighbor_dsl_min": return "Days since last in nearby cells (min)"
    m = re.match(r"^dist_(.+)_m$", col)
    if m:
        base = m.group(1).replace("_"," ").title().replace("Busstops","Bus stops").replace("Roads Primary","Primary road").replace("Roads Secondary","Secondary road")
        return f"Distance to {base} (m)"
    m = re.match(r"^cnt_(.+)_(\d+)m$", col)
    if m:
        what, r = m.groups()
        base = what.replace("_"," ").title().replace("Busstops","Bus stops")
        return f"{base} within {r} m (count)"
    m = re.match(r"^lag(\d+)d$", col)
    if m: return f"Incident {m.group(1)} days ago (flag)"
    m = re.match(r"^past(\d+)_sum$", col)
    if m: return f"Incidents in past {m.group(1)} days (sum)"
    m = re.match(r"^past(\d+)_exp$", col)
    if m: return f"Recent {m.group(1)}-day activity (weighted)"
    return None

def fmt_value(col: str, val):
    try: x = float(val)
    except: return val
    if col.startswith(("dist_","cnt_","days_since")) or re.match(r"^lag\d+d$", col):
        return int(round(x))
    if col in {"same_daypart_last_week"}: return int(round(x))
    if col in {"ri_norm"} or col.startswith("past") or col=="neighbor_lag7d":
        return round(x, 3)
    return round(x, 3)

sel_hex = st.selectbox("Inspect grid cell (H3)", options=top["h3"].tolist() if len(top) else [])
st.caption(f"Selected H3: {sel_hex}")

if sel_hex:
    # risk & priority of this hex for the current daypart/date
    today = pred.copy()
    today["_h3"] = today["h3"].astype(str)
    risk_here = float(today.loc[today["_h3"]==sel_hex, "risk"].max()) if (today["_h3"]==sel_hex).any() else np.nan
    today = today.sort_values("risk", ascending=False).reset_index(drop=True)
    today["rank"] = today.index + 1
    if (today["_h3"]==sel_hex).any():
        rank_here = int(today.loc[today["_h3"]==sel_hex, "rank"].min())
        pct = 100.0 * (1.0 - (rank_here / len(today)))
        band = ("Very high" if pct >= 95 else "High" if pct >= 80 else "Medium" if pct >= 50 else "Low")
        st.markdown(f"**Risk score:** {risk_here:.3f}  |  **Priority:** {rank_here}  |  **Level:** {band} (top {pct:.1f}%)")
    else:
        st.markdown("Risk score: n/a")

    # Show which sector this hex belongs to (if ops/geojson available)
    sector_row = find_sector_for_hex(sel_hex, sel_date, sel_dp)
    if sector_row is not None:
        sector_id = str(sector_row.get("sector_id"))
        # pull attributes from CSV
        if os.path.exists(SECT_CSV):
            s_all = pd.read_csv(SECT_CSV, parse_dates=["date"])
            s_one = s_all[(s_all["date"]==sel_date) & (s_all["daypart"]==sel_dp) & (s_all["sector_id"]==sector_id)]
            if not s_one.empty:
                info = s_one.iloc[0]
                st.info(
                    f"This cell is in **Sector S{int(info['sector_rank'])}** "
                    f"(Priority {int(info['sector_rank'])}) — "
                    f"**Suggested minutes:** {int(info['dwell_min'])}, "
                    f"**Cells:** {int(info['n_hex'])}, "
                    f"**Area:** {round(float(info['area_ha']),1)} ha."
                )
            else:
                st.info(f"This cell is in **{sector_id}**.")
        else:
            st.info(f"This cell is in **{sector_id}**.")
    else:
        st.info("This cell is not in today’s Top-K selection for this time-of-day.")

    # Friendly feature table for this hex
    row_X = Xsel.loc[rep["h3"].astype(str) == sel_hex]
    if row_X.empty:
        st.info("No feature row for that cell.")
    else:
        drop_like = [c for c in row_X.columns if c.endswith("_x") or c.endswith("_y") or c.startswith("daypart_")]
        row = row_X.drop(columns=drop_like, errors="ignore").iloc[0]
        items = []
        for col, val in row.items():
            if col in {"date","h3"}: continue
            lab = friendly_label(col)
            if not lab: continue
            items.append({"Feature": lab, "Value": fmt_value(col, val)})

        if items:
            df_nice = pd.DataFrame(items)
            def _grp(name):
                if "Distance" in name: return "Distances"
                if "within" in name:   return "Counts"
                if "Recent" in name or "last" in name or "flag" in name: return "Recent history"
                return "Other"
            df_nice["Group"] = df_nice["Feature"].map(_grp)
            order = {"Recent history":0, "Counts":1, "Distances":2, "Other":3}
            df_nice["__o"] = df_nice["Group"].map(order).fillna(9)
            df_nice = df_nice.sort_values(["__o","Feature"]).drop(columns="__o").reset_index(drop=True)
            st.dataframe(df_nice, use_container_width=True)
        else:
            st.info("No interpretable features to display.")

    # Optional: SHAP drivers (advanced)
    show_shap = st.checkbox("Show top drivers (SHAP — slower)", value=False)
    if show_shap and not row_X.empty:
        try:
            import shap
            model_for_shap = None
            if os.path.exists(MODEL_RAW):
                with open(MODEL_RAW, "rb") as f: model_for_shap = pickle.load(f)
            if model_for_shap is None:
                model_for_shap = getattr(clf_cal,"base_estimator",None) or getattr(clf_cal,"estimator",None)

            sv = shap.TreeExplainer(model_for_shap).shap_values(row_X)
            if isinstance(sv, list): sv = sv[1]
            contrib = pd.Series(sv[0], index=row_X.columns)
            contrib = contrib.drop(index=drop_like, errors="ignore")
            top = contrib.reindex([c for c in contrib.index if friendly_label(c)]).sort_values(key=lambda s: -np.abs(s)).head(10)
            if not top.empty:
                st.write("Top drivers: positive values ↑ increase risk; negative values ↓ reduce risk.")
                st.dataframe(pd.DataFrame({
                    "Feature": [friendly_label(c) for c in top.index],
                    "Impact": [round(float(v), 4) for v in top.values]
                }), use_container_width=True)
            else:
                st.info("No interpretable drivers to show.")
        except Exception as e:
            st.info(f"SHAP not available ({e}).")




# ---------- REPORTS ----------
st.subheader("Reports")
c1, c2 = st.columns(2)
with c1:
    if os.path.exists(EVAL_SUM):
        st.download_button("Download eval summary", data=open(EVAL_SUM, "rb"), file_name="eval_summary.csv")
with c2:
    if os.path.exists(HRK_SW):
        st.download_button("Download HR@K sweep", data=open(HRK_SW, "rb"), file_name="hrk_sweep.csv")

with st.expander("Limitations & notes"):
    st.markdown("""
- Data are **sparse**; incidents may be **under-reported**; some positives are **augmented** during training.
- Predictions are **area-based**; **no person-level** inference.
- Use with **local thana logs** and professional judgment.
""")

# ---------- HR@K SWEEP (friendly) ----------
import os
import pandas as pd

st.subheader("Hit rate by coverage (HR@K)")

if os.path.exists(HRK_SW):
    sweep = pd.read_csv(HRK_SW)
    # Rename columns for plain English
    rename_map = {
        "K_percent": "Coverage (%)",
        "HitRate@K": "Hit rate",
        "PAI@K": "PAI (× random)"
    }
    show_cols = [c for c in ["Coverage (%)","Hit rate","PAI (× random)"]
                 if c in rename_map.values() or c in sweep.columns]
    sweep_disp = sweep.rename(columns=rename_map)
    st.dataframe(sweep_disp[show_cols], use_container_width=True)

    st.caption(
        "At **K% coverage**, **Hit rate** is the percent of incidents that fell **inside** the flagged area "
        "on those days (higher is better).  "
        "**PAI** shows improvement vs random selection: **PAI = Hit rate / (K/100)**."
    )
else:
    st.info("HR@K sweep file not found. Add `metrics/hrk_sweep.csv` to show this table.")



import os, glob
import numpy as np
import pandas as pd
import streamlit as st
from datetime import date, timedelta

# Try plotting libs
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    import altair as alt
except Exception:
    alt = None

# Optional deps (graceful fallback)
try:
    import h3
except Exception:
    h3 = None

try:
    import pydeck as pdk
except Exception:
    pdk = None

# ---------------- UI SETUP ----------------
st.set_page_config(page_title="NYCSBUS • Holiday + Hotspots + Alarms", layout="wide")
st.markdown(
    '''
    <style>
      .stApp { background:#0f1116; color:#e8e8e8; }
      .block-container { padding-top: 1.2rem; }
      .card { background:#151a22; border-radius:16px; padding:16px; box-shadow:0 0 0 1px #23283b inset; }
      a, a:visited { color:#9ecbff; }
    </style>
    ''',
    unsafe_allow_html=True
)

st.title("NYCSBUS • Holiday Effects + Hotspots + H3 Alarms")
st.caption("All-in-one: clean crash data, analyze holiday windows & seasons, merge hotspot/trend files, and compute rolling-baseline alarms. Dark UI + light basemap.")

# ---------------- SIDEBAR ----------------
st.sidebar.header("Data Inputs")
use_discover = st.sidebar.checkbox("Auto-discover CSVs in working folder", value=True)
uploaded_crash = st.sidebar.file_uploader("Upload crash CSVs", type=["csv"], accept_multiple_files=True)
uploaded_hotspot = st.sidebar.file_uploader("Upload hotspot_analysis.csv (optional)", type=["csv"])
uploaded_trend = st.sidebar.file_uploader("Upload trend_analysis.csv (optional)", type=["csv"])
uploaded_spalarms = st.sidebar.file_uploader("Upload spatiotemporal_alarms.csv (optional)", type=["csv"])

h3_res = st.sidebar.slider("H3 resolution (if binning)", 5, 10, 8)
sigma = st.sidebar.slider("Alarm z-threshold", 1.0, 4.0, 2.0, 0.5)
window = st.sidebar.slider("Baseline window (days)", 7, 28, 14, 1)
min_count = st.sidebar.slider("Min count to alarm", 1, 10, 3, 1)

# ---------------- HELPERS ----------------
DATE_COLS = ["CRASH_DATE", "crash_date", "Crash Date"]
TIME_COLS = ["CRASH_TIME", "crash_time", "Crash Time"]
LAT_COLS  = ["LATITUDE", "latitude", "Latitude"]
LON_COLS  = ["LONGITUDE", "longitude", "Longitude"]
BORO_COLS = ["BOROUGH", "borough", "Borough"]

def discover_paths():
    pats = ["./*.csv","../*.csv","/mnt/data/*.csv"]
    out = []
    for p in pats:
        out.extend(glob.glob(p))
    return out

def first_col(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None

def harmonize(df):
    cols = df.columns
    cdate = first_col(cols, DATE_COLS)
    ctime = first_col(cols, TIME_COLS)
    clat  = first_col(cols, LAT_COLS)
    clon  = first_col(cols, LON_COLS)
    cbor  = first_col(cols, BORO_COLS)

    out = df.copy()
    if not cdate:
        raise KeyError("No crash date column found. Expected one of: " + ", ".join(DATE_COLS))
    d = pd.to_datetime(out[cdate], errors="coerce")
    if ctime and ctime in out:
        dt = pd.to_datetime(d.dt.date.astype(str) + " " + out[ctime].astype(str), errors="coerce")
    else:
        dt = pd.to_datetime(d.dt.date.astype(str), errors="coerce")
    out["crash_datetime"] = dt
    out["crash_date"] = out["crash_datetime"].dt.date

    out["latitude"] = pd.to_numeric(out[clat], errors="coerce") if clat else np.nan
    out["longitude"] = pd.to_numeric(out[clon], errors="coerce") if clon else np.nan
    out["borough"] = out[cbor] if cbor else np.nan

    out = out.dropna(subset=["crash_datetime"])
    lat_ok = out["latitude"].between(40.3, 41.1, inclusive="neither")
    lon_ok = out["longitude"].between(-74.5, -73.3, inclusive="neither")
    has_xy = (~out["latitude"].isna()) & (~out["longitude"].isna())
    out = out[(~has_xy) | (lat_ok & lon_ok)]
    return out

def nth_weekday_of_month(year, month, weekday, n):
    d = date(year, month, 1)
    while d.weekday() != weekday:
        d += timedelta(days=1)
    d += timedelta(weeks=n-1)
    return d

def holiday_windows_for_year(year):
    res = {}
    july4 = date(year, 7, 4)
    res["independence_day"] = {july4 - timedelta(days=1), july4, july4 + timedelta(days=1)}
    res["christmas"] = {date(year, 12, 24), date(year, 12, 25), date(year, 12, 26)}
    res["new_year"] = {date(year-1, 12, 31), date(year, 1, 1), date(year, 1, 2)}
    labor = nth_weekday_of_month(year, 9, weekday=0, n=1)  # Monday=0
    res["labor_day"] = {labor - timedelta(days=1), labor, labor + timedelta(days=1)}
    return res

def label_holiday_window(d):
    yrs = {d.year, d.year+1, d.year-1}
    for y in yrs:
        for name, days in holiday_windows_for_year(y).items():
            if d in days:
                return name
    return "none"

def label_holiday_season(d):
    y = d.year
    if date(y,6,30) <= d <= date(y,7,14):
        return "independence_season"
    if (d.month==12 and d.day>=24) or (d.month==1 and d.day<=2):
        return "christmas_newyear_season"
    labor = nth_weekday_of_month(y,9,weekday=0,n=1)
    if d in {labor - timedelta(days=1), labor, labor + timedelta(days=1)}:
        return "labor_day_season"
    return "none"

def compute_alarms(g, window=14, sigma=2.0, min_count=3):
    g = g.sort_values("crash_date").copy()
    s = pd.Series(g["count"].values)
    g["roll_mean"] = s.rolling(window, min_periods=5).mean().shift(1)
    g["roll_std"]  = s.rolling(window, min_periods=5).std(ddof=0).shift(1)
    g["z"] = (g["count"] - g["roll_mean"]) / g["roll_std"]
    g["z"] = g["z"].replace([np.inf, -np.inf], np.nan)
    g["alarm"] = np.where((g["count"]>=min_count) & (g["z"]>=sigma), "alarm", "none")
    return g

# ---------------- LOAD DATA ----------------
sources = []
if uploaded_crash:
    sources = uploaded_crash
elif use_discover:
    candidates = discover_paths()
    sources = [f for f in candidates if any(k in os.path.basename(f).lower() for k in ["crashes","collisions","motor_vehicle","crash"])]
dfs = []
for s in sources:
    try:
        if hasattr(s, "read"):
            df0 = pd.read_csv(s, low_memory=False)
            df0["__source_file"] = getattr(s, "name", "uploaded.csv")
        else:
            df0 = pd.read_csv(s, low_memory=False)
            df0["__source_file"] = os.path.basename(s)
        dfs.append(df0)
    except Exception as e:
        st.warning(f"Skipped {s}: {e}")

if dfs:
    crashes = pd.concat([harmonize(d) for d in dfs], ignore_index=True)
    crashes = crashes.drop_duplicates(subset=["crash_datetime","latitude","longitude"], keep="first")
else:
    crashes = pd.DataFrame(columns=["crash_datetime","crash_date","latitude","longitude","borough"])

def load_single(obj):
    if obj is None: return None
    try:
        if hasattr(obj, "read"): return pd.read_csv(obj, low_memory=False)
        return pd.read_csv(obj, low_memory=False)
    except Exception:
        return None

hotspots = load_single(uploaded_hotspot)
trends = load_single(uploaded_trend)
sp_alarms = load_single(uploaded_spalarms)

if use_discover:
    for f in discover_paths():
        name = os.path.basename(f).lower()
        if hotspots is None and name == "hotspot_analysis.csv": hotspots = load_single(f)
        if trends is None and name == "trend_analysis.csv": trends = load_single(f)
        if sp_alarms is None and name == "spatiotemporal_alarms.csv": sp_alarms = load_single(f)

# -------- Derive flags --------
if len(crashes):
    crashes["holiday_window"] = crashes["crash_date"].apply(label_holiday_window)
    crashes["season"] = crashes["crash_date"].apply(label_holiday_season)
    crashes["is_holiday_window"] = (crashes["holiday_window"]!="none").astype(int)
    crashes["is_holiday_season"] = (crashes["season"]!="none").astype(int)
    if h3 is None:
        crashes["h3"] = np.nan
    else:
        m = (~crashes["latitude"].isna()) & (~crashes["longitude"].isna())
        crashes.loc[m, "h3"] = [h3.geo_to_h3(lat, lon, h3_res) for lat, lon in crashes.loc[m, ["latitude","longitude"]].to_numpy()]
else:
    crashes["holiday_window"] = pd.Series(dtype=object)
    crashes["season"] = pd.Series(dtype=object)
    crashes["is_holiday_window"] = pd.Series(dtype=int)
    crashes["is_holiday_season"] = pd.Series(dtype=int)
    crashes["h3"] = pd.Series(dtype=object)

cell_col = "h3" if crashes.get("h3", pd.Series()).notna().any() else "__global_cell"
if cell_col == "__global_cell":
    crashes[cell_col] = "global"

# KPIs
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Crash rows", f"{len(crashes):,}")
span_days = (pd.to_datetime(crashes["crash_date"]).max() - pd.to_datetime(crashes["crash_date"]).min()).days + 1 if len(crashes) else 0
c2.metric("Days covered", f"{span_days:,}")
c3.metric("Holiday-window rows", f"{int(crashes['is_holiday_window'].sum() if len(crashes) else 0):,}")
c4.metric("Holiday-season rows", f"{int(crashes['is_holiday_season'].sum() if len(crashes) else 0):,}")
c5.metric("Cells", f"{crashes[cell_col].nunique() if len(crashes) else 0:,}")

st.markdown('---')
tab_overview, tab_holiday, tab_seasons, tab_hotspots, tab_alarms, tab_map = st.tabs(["Overview","Holiday windows","Holiday seasons","Hotspots & trends","Alarms","Map"])

# ---------------- OVERVIEW ----------------
with tab_overview:
    st.markdown("### What this app does")
    st.markdown("- Cleans & harmonizes crash CSVs")
    st.markdown("- Flags **holiday windows** and **holiday seasons**")
    st.markdown("- Optional **H3** binning; per-cell rolling **z-score alarms**")
    st.markdown("- Loads **hotspot_analysis / trend_analysis / spatiotemporal_alarms** if provided")
    if len(crashes):
        st.markdown("#### Sample rows")
        st.dataframe(crashes.head(25))

# ---------------- HOLIDAY WINDOWS ----------------
with tab_holiday:
    if not len(crashes):
        st.info("Upload crash CSVs to compute holiday-window effects.")
    else:
        daily = crashes.groupby("crash_date").size().rename("count").reset_index()
        daily["holiday_window"] = daily["crash_date"].apply(label_holiday_window)
        daily["is_holiday_window"] = (daily["holiday_window"]!="none").astype(int)

        avg_h = daily.query("is_holiday_window==1")["count"].mean()
        avg_n = daily.query("is_holiday_window==0")["count"].mean()
        pct = 100*(avg_h-avg_n)/max(avg_n,1)
        st.markdown(f"<div class='card'>Avg holiday-window: <b>{avg_h:.1f}</b> • Non-holiday: <b>{avg_n:.1f}</b> • Δ <b>{pct:+.1f}%</b></div>", unsafe_allow_html=True)
        st.line_chart(daily.set_index("crash_date")["count"])

# ---------------- HOLIDAY SEASONS (RICH VISUALS) ----------------
with tab_seasons:
    if not len(crashes):
        st.info('Upload crash CSVs to compute holiday-season effects.')
    else:
        # Prepare daily with season labels
        daily = crashes.groupby('crash_date').size().rename('count').reset_index()
        daily['season'] = daily['crash_date'].apply(label_holiday_season)

        # Summary by season
        season_stats = daily.groupby('season')['count'].agg(['count','mean']).reset_index()
        st.markdown('**Season summary**')
        st.dataframe(season_stats)

        # Season selector
        seasons = [s for s in daily['season'].unique() if s != 'none']
        if not seasons:
            st.info("No season-tagged days in this dataset.")
        else:
            sel = st.selectbox('Pick a season', options=seasons)

            # ---------- 1) Season-aligned spaghetti plot + mean ----------
            def season_span(y, season):
                if season == "independence_season":
                    return date(y, 6, 30), date(y, 7, 14)
                if season == "christmas_newyear_season":
                    return date(y, 12, 24), date(y+1, 1, 2)
                if season == "labor_day_season":
                    labor = nth_weekday_of_month(y, 9, weekday=0, n=1)  # Monday=0
                    return labor - timedelta(days=1), labor + timedelta(days=1)
                return None, None

            def season_aligned(daily_df, season):
                rows = []
                years = sorted(set(pd.to_datetime(daily_df["crash_date"]).dt.year))
                for y in years:
                    s, e = season_span(y, season)
                    if s is None:
                        continue
                    sl = daily_df[(daily_df["crash_date"]>=pd.to_datetime(s)) & (daily_df["crash_date"]<=pd.to_datetime(e))].copy()
                    if sl.empty:
                        continue
                    sl = sl.sort_values("crash_date").reset_index(drop=True)
                    sl["day_idx"] = np.arange(len(sl))
                    sl["year"] = y
                    rows.append(sl[["year","day_idx","count","crash_date"]])
                return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["year","day_idx","count","crash_date"])

            aligned = season_aligned(daily, sel)
            if aligned.empty:
                st.info("No data matched this season.")
            else:
                if plt is not None:
                    fig, ax = plt.subplots(figsize=(10, 4))
                    for y, g in aligned.groupby("year"):
                        ax.plot(g["day_idx"], g["count"], alpha=0.3)
                    m = aligned.groupby("day_idx")["count"].mean()
                    ax.plot(m.index, m.values, linewidth=3, label="Season mean")
                    ax.set_xlabel("Season day (aligned)")
                    ax.set_ylabel("Crashes / day")
                    ax.set_title(f"{sel}: aligned across years")
                    ax.legend()
                    st.pyplot(fig, clear_figure=True)
                elif alt is not None:
                    mean_df = aligned.groupby("day_idx")["count"].mean().reset_index()
                    chart = alt.Chart(aligned).mark_line(opacity=0.35).encode(
                        x=alt.X("day_idx:Q", title="Season day (aligned)"),
                        y=alt.Y("count:Q", title="Crashes / day"),
                        color=alt.Color("year:N", legend=None)
                    ) + alt.Chart(mean_df).mark_line(size=3).encode(
                        x="day_idx:Q", y="count:Q"
                    )
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.line_chart(aligned.pivot(index="day_idx", columns="year", values="count"))

            # ---------- 2) Baseline-relative season lift ----------
            def add_baseline_lift(daily_df):
                d = daily_df.sort_values("crash_date").copy()
                d["non_season"] = (d["season"]=="none").astype(int)
                baseline = []
                from collections import deque
                win = deque(maxlen=14)
                for _, row in d.iterrows():
                    if row["season"] == "none" and np.isfinite(row["count"]):
                        win.append(row["count"])
                    mu = np.mean(win) if len(win) > 0 else np.nan
                    baseline.append(mu)
                d["baseline_14_nonseason"] = baseline
                d["lift_pct"] = 100 * (d["count"] - d["baseline_14_nonseason"]) / d["baseline_14_nonseason"]
                return d

            daily2 = add_baseline_lift(daily)
            f = daily2[daily2["season"] == sel].copy()
            if not f.empty and f["lift_pct"].notna().any():
                if plt is not None:
                    fig, ax = plt.subplots(figsize=(10, 3.8))
                    ax.plot(f["crash_date"], f["lift_pct"], marker=".", linestyle="-", alpha=0.8)
                    ax.axhline(0, linewidth=1)
                    ax.set_title(f"{sel}: lift vs prior 14 non-season days")
                    ax.set_ylabel("% lift"); ax.set_xlabel("Date")
                    st.pyplot(fig, clear_figure=True)
                elif alt is not None:
                    base = alt.Chart(f).mark_line(point=True).encode(
                        x=alt.X("crash_date:T", title="Date"),
                        y=alt.Y("lift_pct:Q", title="% lift")
                    )
                    zero = alt.Chart(pd.DataFrame({"y":[0]})).mark_rule().encode(y="y:Q")
                    st.altair_chart(base + zero, use_container_width=True)
                else:
                    st.line_chart(f.set_index("crash_date")["lift_pct"])
            else:
                st.info("Not enough non-season history to compute lift.")

            # ---------- 3) DOW × Hour heatmap (season only) ----------
            df_season = crashes[crashes["season"] == sel].copy()
            if "crash_datetime" in df_season.columns and not df_season.empty:
                df_season["dow"]  = pd.to_datetime(df_season["crash_datetime"]).dt.day_name().str[:3]
                df_season["hour"] = pd.to_datetime(df_season["crash_datetime"]).dt.hour
                if alt is not None:
                    order = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
                    heat = alt.Chart(df_season).mark_rect().encode(
                        x=alt.X("hour:O", title="Hour of day"),
                        y=alt.Y("dow:O", sort=order, title="Day of week"),
                        color=alt.Color("count():Q", title="Crashes")
                    )
                    st.altair_chart(heat, use_container_width=True)
                else:
                    st.dataframe(df_season.groupby(["dow","hour"]).size().unstack("hour").fillna(0))
            else:
                st.info("Time-of-day fields not available to create a heatmap.")

            # ---------- 4) Spatial density HexagonLayer (season filter) ----------
            if pdk is not None:
                pts = crashes[crashes["season"] == sel].dropna(subset=["latitude","longitude"])
                if not pts.empty:
                    layer = pdk.Layer(
                        "HexagonLayer",
                        data=pts.sample(min(len(pts), 10000)),
                        get_position='[longitude, latitude]',
                        radius=150,
                        elevation_scale=30,
                        elevation_range=[0, 4000],
                        extruded=True,
                        coverage=1.0,
                        pickable=True,
                    )
                    view = pdk.ViewState(latitude=40.73, longitude=-73.94, zoom=10)
                    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, map_style="light"))
                else:
                    st.info("No lat/lon points for this season to map.")

            # ---------- Bonus: Borough comparison (season vs non-season) ----------
            if "borough" in crashes.columns and crashes["borough"].notna().any():
                b = (crashes.assign(is_season=lambda x: (x["season"]==sel).astype(int))
                            .groupby(["borough","is_season"]).size().rename("n").reset_index())
                pivot = b.pivot(index="borough", columns="is_season", values="n").fillna(0)
                # Normalize columns to names
                if 0 in pivot.columns and 1 in pivot.columns:
                    pivot.columns = ["non_season","season"]
                elif 0 in pivot.columns and 1 not in pivot.columns:
                    pivot.columns = ["non_season"]
                    pivot["season"] = 0
                elif 1 in pivot.columns and 0 not in pivot.columns:
                    pivot.columns = ["season"]
                    pivot["non_season"] = 0
                else:
                    pivot["non_season"] = pivot.get("non_season", 0)
                    pivot["season"] = pivot.get("season", 0)
                pivot = pivot.reset_index()
                bars = pivot.melt("borough", var_name="period", value_name="n")
                if alt is not None:
                    chart = alt.Chart(bars).mark_bar().encode(
                        y=alt.Y("borough:N", sort='-x'),
                        x=alt.X("n:Q", title="Crashes"),
                        color=alt.Color("period:N", title=""),
                        tooltip=["borough","period","n"]
                    )
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.dataframe(bars)

# ---------------- HOTSPOTS & TRENDS ----------------
with tab_hotspots:
    if hotspots is None and trends is None and sp_alarms is None:
        st.info("Upload hotspot/trend/spatiotemporal_alarms CSVs or place them in the folder.")
    else:
        if hotspots is not None:
            st.markdown("**Hotspot analysis**")
            st.dataframe(hotspots.head(30))
        if trends is not None:
            st.markdown("**Trend analysis**")
            st.dataframe(trends.head(30))
        if sp_alarms is not None:
            st.markdown("**Spatiotemporal alarms (precomputed)**")
            st.dataframe(sp_alarms.head(30))

# ---------------- ALARMS ----------------
with tab_alarms:
    if not len(crashes):
        st.info("Upload crash CSVs to compute alarms.")
    else:
        cell_daily = (crashes.groupby([cell_col,"crash_date"]).size().rename("count").reset_index().sort_values("crash_date"))
        alarms = cell_daily.groupby(cell_col, group_keys=False).apply(lambda g: compute_alarms(g, window=window, sigma=sigma, min_count=min_count))
        st.dataframe(alarms.sort_values(["crash_date","z"], ascending=[False, False]).reset_index(drop=True))

        # Simple next-day prediction
        def predict_next_day(g):
            g = g.sort_values("crash_date")
            if g.empty: return None
            last_date = g["crash_date"].max()
            next_date = pd.to_datetime(last_date) + pd.Timedelta(days=1)
            base = g["count"].tail(7).mean()
            # season-aware uplift
            hw = label_holiday_season(next_date.date())
            uplift = max(1.0, 0.15*base) if hw!='none' else 0.0
            pred = base + uplift
            prior = g.dropna(subset=["roll_mean","roll_std"]).tail(7)
            if len(prior)==0:
                mu, sd = base, 1.0
            else:
                mu = prior["roll_mean"].iloc[-1]
                sd = prior["roll_std"].iloc[-1] or 1.0
            z = (pred - mu) / sd if sd else np.nan
            will_alarm = (pred >= mu + 2.0*sd) and (pred >= min_count)
            return {'cell': g.iloc[0][cell_col], 'last_date': last_date, 'next_date': next_date.date(), 'pred_count': float(np.nan_to_num(pred)), 'pred_z': float(np.nan_to_num(z)), 'pred_alarm': bool(will_alarm), 'next_in_season': hw}

        preds = [predict_next_day(g) for _, g in alarms.groupby(cell_col)]
        preds = [p for p in preds if p is not None]
        preds_df = pd.DataFrame(preds).sort_values(["pred_alarm","pred_z","pred_count"], ascending=[False, False, False])
        st.markdown("**Next-day predictions (season-aware uplift)**")
        st.dataframe(preds_df)

# ---------------- MAP ----------------
with tab_map:
    if pdk is None:
        st.info("pydeck not available in this environment.")
    else:
        layers = []
        if len(crashes):
            cell_daily = crashes.groupby([cell_col,"crash_date"]).size().rename("count").reset_index()
            alarms = cell_daily.groupby(cell_col, group_keys=False).apply(lambda g: compute_alarms(g, window=window, sigma=sigma, min_count=min_count))
            latest_day = alarms["crash_date"].max() if len(alarms) else None
            if latest_day is not None:
                latest = alarms[(alarms["crash_date"]==latest_day) & (alarms["alarm"]=="alarm")]
                cents = crashes.dropna(subset=["latitude","longitude"]).groupby(cell_col)[["latitude","longitude"]].mean().reset_index()
                latest = latest.merge(cents, on=cell_col, how="left").dropna(subset=["latitude","longitude"])
                if not latest.empty:
                    layers.append(pdk.Layer("ScatterplotLayer", data=latest, get_position='[longitude, latitude]', get_radius=90, get_fill_color='[255,80,80]', pickable=True))
        if hotspots is not None and h3 is not None and "h3_cell" in hotspots.columns:
            pts = []
            for h in hotspots["h3_cell"].dropna().unique():
                try:
                    lat, lon = h3.h3_to_geo(h)
                    pts.append({"latitude": lat, "longitude": lon})
                except Exception:
                    pass
            if pts:
                layers.append(pdk.Layer("ScatterplotLayer", data=pts, get_position='[longitude, latitude]', get_radius=60, get_fill_color='[120,180,255]', pickable=False))
        if not layers and len(crashes):
            pts = crashes.dropna(subset=["latitude","longitude"]).copy()
            layers.append(pdk.Layer("ScatterplotLayer", data=pts.sample(min(5000, len(pts))), get_position='[longitude, latitude]', get_radius=25, pickable=True))
        view_state = pdk.ViewState(latitude=40.73, longitude=-73.94, zoom=10)
        st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view_state, map_style="light"))

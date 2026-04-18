# ─── app.py — Maritime Vessel Activity Monitoring | Streamlit Dashboard ───────
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
import joblib
import json

# ─── Load models, preprocessors & saved metrics ──────────────────────────────
with open('model_metrics.json') as f:
    model_metrics = json.load(f)

rf_model    = joblib.load('rf_model.pkl')
xgb_model   = joblib.load('xgb_model.pkl')
lgbm_model  = joblib.load('lgbm_model.pkl')
iso_model   = joblib.load('iso_forest.pkl')
scaler      = joblib.load('scaler.pkl')
le_gear     = joblib.load('le_gear.pkl')
le_flag     = joblib.load('le_flag.pkl')

# ─── Feature lists ────────────────────────────────────────────────────────────
# IsolationForest was trained on exactly these 11 features (Section 6 of notebook)
IF_FEATURES = [
    'Fishing_Hours_Log', 'Duration_Hours', 'AIS_History_Days',
    'Fishing_Intensity', 'AIS_Rate_PerDay', 'Has_IMO',
    'Has_CallSign', 'Suspicious_Gear', 'Entry_Month',
    'Entry_DayOfWeek', 'Entry_Hour'
]

# Supervised models (RF/XGB/LGBM) were trained on these 13 features (Section 7)
MODEL_FEATURES = [
    'Fishing_Hours_Log', 'Duration_Hours', 'AIS_History_Days',
    'Fishing_Intensity', 'AIS_Rate_PerDay', 'Has_IMO',
    'Has_CallSign', 'Suspicious_Gear', 'Entry_Month',
    'Entry_DayOfWeek', 'Entry_Hour', 'Gear_Encoded', 'Flag_Encoded'
]

# ─── Config ───────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
COLOUR_MAP   = {'Normal': '#2196F3', 'Anomaly': '#F44336'}
PLOTLY_THEME = 'plotly_dark'

st.set_page_config(
    page_title="Maritime Vessel Activity Monitor",
    page_icon="⚓",
    layout="wide"
)

st.markdown("""
    <style>
        .block-container { padding-top: 1rem; }
        .stMetric { background: #1E2A3A; border-radius: 8px; padding: 10px; }
    </style>
""", unsafe_allow_html=True)

st.title("⚓ Maritime Vessel Activity Monitoring Dashboard")
st.caption("AIS-Based Business Intelligence Pipeline · MSc IT with BI · Robert Gordon University")
st.markdown("---")


# ─── Sidebar — File Upload ────────────────────────────────────────────────────
st.sidebar.header("📂 Upload Dataset")
uploaded = st.sidebar.file_uploader("Upload fishing-data.csv", type=["csv"])

if uploaded is None:
    st.info("👈 Upload your **fishing-data.csv** file from the sidebar to begin.")
    st.stop()


# ─── Load & Clean (matches Section 2 + Section 4 of notebook exactly) ─────────
@st.cache_data
def load_and_prepare(file):
    df = pd.read_csv(file)

    # Parse timestamps
    time_cols = ['Entry Timestamp', 'Exit Timestamp',
                 'First Transmission Date', 'Last Transmission Date']
    for col in time_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors='coerce')

    # Rename
    df.rename(columns={'Apparent Fishing Hours': 'Fishing_Hours'}, inplace=True)

    # Fill missing
    df['Vessel Name'] = df['Vessel Name'].fillna('UNKNOWN') if 'Vessel Name' in df.columns else 'UNKNOWN'
    df['Gear Type']   = df['Gear Type'].fillna('UNKNOWN')   if 'Gear Type'   in df.columns else 'UNKNOWN'
    df['Flag']        = df['Flag'].fillna('UNKNOWN')         if 'Flag'        in df.columns else 'UNKNOWN'

    # Drop rows missing MMSI
    df.dropna(subset=['MMSI'], inplace=True)

    # IMO flag
    df['Has_IMO'] = df['IMO'].notnull().astype(int)

    # Drop unused columns if present
    for col in ['Time Range', 'Vessel Type']:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # ── Feature Engineering (Section 5) ──────────────────────────────────────
    df['Duration_Hours'] = (
        (df['Exit Timestamp'] - df['Entry Timestamp'])
        .dt.total_seconds() / 3600
    ).clip(lower=0)

    df['AIS_History_Days'] = (
        (df['Last Transmission Date'] - df['First Transmission Date'])
        .dt.total_seconds() / 86400
    ).clip(lower=0)

    df['Entry_Month']      = df['Entry Timestamp'].dt.month
    df['Entry_Month_Name'] = df['Entry Timestamp'].dt.strftime('%b')
    df['Entry_DayOfWeek']  = df['Entry Timestamp'].dt.dayofweek
    df['Entry_Hour']       = df['Entry Timestamp'].dt.hour

    df['Fishing_Intensity'] = pd.Series(np.where(
        df['Duration_Hours'] > 0,
        df['Fishing_Hours'] / df['Duration_Hours'], 0
    )).clip(upper=1)

    def month_to_season(m):
        if m in [12, 1, 2]:  return 'Winter'
        if m in [3, 4, 5]:   return 'Spring'
        if m in [6, 7, 8]:   return 'Summer'
        return 'Autumn'
    df['Season'] = df['Entry_Month'].apply(month_to_season)

    bins   = [-1, 0, 100, 500, 1500, np.inf]
    labels = ['Inactive', 'Low', 'Moderate', 'High', 'Very High']
    df['Activity_Tier'] = pd.cut(df['Fishing_Hours'], bins=bins, labels=labels)

    df['AIS_Rate_PerDay'] = np.where(
        df['AIS_History_Days'] > 0,
        df['Fishing_Hours'] / df['AIS_History_Days'], 0
    )

    df['Fishing_Hours_Log'] = np.log1p(df['Fishing_Hours'])
    df['Has_CallSign']      = df['CallSign'].notnull().astype(int)

    suspicious_gears      = ['INCONCLUSIVE', 'FISHING', 'UNKNOWN']
    df['Suspicious_Gear'] = df['Gear Type'].isin(suspicious_gears).astype(int)

    # Label encoding — use transform so it matches the saved encoders
    df['Gear_Encoded'] = df['Gear Type'].apply(
        lambda x: le_gear.transform([x])[0] if x in le_gear.classes_ else -1
    )
    df['Flag_Encoded'] = df['Flag'].apply(
        lambda x: le_flag.transform([x])[0] if x in le_flag.classes_ else -1
    )

    return df


df = load_and_prepare(uploaded)
st.sidebar.success(f"✅ Loaded {len(df):,} rows × {len(df.columns)} columns")


# ─── Sidebar Filters ──────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.header("🔍 Filters")

flags    = ["All"] + sorted(df['Flag'].dropna().unique().tolist())
sel_flag = st.sidebar.selectbox("Flag State", flags)

gears    = ["All"] + sorted(df['Gear Type'].dropna().unique().tolist())
sel_gear = st.sidebar.selectbox("Gear Type", gears)

df_view = df.copy()
if sel_flag != "All":
    df_view = df_view[df_view['Flag'] == sel_flag]
if sel_gear != "All":
    df_view = df_view[df_view['Gear Type'] == sel_gear]

st.sidebar.markdown(f"**Filtered records: {len(df_view):,}**")


# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "🎣 Vessel Activity",
    "📅 Temporal Trends",
    "🚨 Anomaly Detection",
    "🤖 ML Classifiers"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Fleet Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Vessels",  f"{len(df_view):,}")
    c2.metric("Unique MMSIs",   f"{df_view['MMSI'].nunique():,}")
    c3.metric("Flag States",    f"{df_view['Flag'].nunique():,}")
    c4.metric("No IMO Number",  f"{(df_view['Has_IMO'] == 0).sum():,}")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        top_flags = df_view['Flag'].value_counts().head(15).reset_index()
        top_flags.columns = ['Flag', 'Count']
        fig = px.bar(top_flags, x='Flag', y='Count',
                     title='Top 15 Flag States by Vessel Count',
                     color='Count', color_continuous_scale='Blues',
                     template=PLOTLY_THEME)
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        gear_dist = df_view['Gear Type'].value_counts().reset_index()
        gear_dist.columns = ['Gear Type', 'Count']
        fig2 = px.pie(gear_dist, names='Gear Type', values='Count',
                      title='Gear Type Distribution',
                      hole=0.4, template=PLOTLY_THEME)
        st.plotly_chart(fig2, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        imo_status = df_view['Has_IMO'].map({1: 'Has IMO', 0: 'No IMO'}).value_counts().reset_index()
        imo_status.columns = ['Status', 'Count']
        fig3 = px.bar(imo_status, x='Status', y='Count',
                      title='IMO Compliance Status',
                      color='Status',
                      color_discrete_map={'Has IMO': '#4CAF50', 'No IMO': '#F44336'},
                      template=PLOTLY_THEME)
        st.plotly_chart(fig3, use_container_width=True)

    with col_d:
        tier_order   = ['Inactive', 'Low', 'Moderate', 'High', 'Very High']
        tier_colours = ['#607D8B', '#4CAF50', '#FF9800', '#F44336', '#B71C1C']
        tier_counts  = df_view['Activity_Tier'].value_counts().reindex(tier_order).reset_index()
        tier_counts.columns = ['Tier', 'Count']
        fig4 = px.bar(tier_counts, x='Tier', y='Count',
                      title='Activity Tier Distribution',
                      color='Tier',
                      color_discrete_sequence=tier_colours,
                      template=PLOTLY_THEME)
        st.plotly_chart(fig4, use_container_width=True)

    with st.expander("🔎 Preview Raw Data (first 100 rows)"):
        st.dataframe(df_view.head(100))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — VESSEL ACTIVITY
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Vessel Fishing Activity Analysis")

    col_a, col_b = st.columns(2)

    with col_a:
        country_hours = (df_view.groupby('Flag')['Fishing_Hours']
                         .sum().sort_values(ascending=False).head(15).reset_index())
        country_hours.columns = ['Flag', 'Total Fishing Hours']
        fig = px.bar(country_hours, x='Flag', y='Total Fishing Hours',
                     title='Top 15 Countries — Total Fishing Hours',
                     color='Total Fishing Hours', color_continuous_scale='Oranges',
                     template=PLOTLY_THEME)
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        gear_avg = (df_view.groupby('Gear Type')['Fishing_Hours']
                    .mean().sort_values(ascending=False).reset_index())
        gear_avg.columns = ['Gear Type', 'Avg Fishing Hours']
        fig2 = px.bar(gear_avg, x='Gear Type', y='Avg Fishing Hours',
                      title='Average Fishing Hours by Gear Type',
                      color='Avg Fishing Hours', color_continuous_scale='Greens',
                      template=PLOTLY_THEME)
        fig2.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.histogram(df_view, x='Fishing_Hours', nbins=60,
                        title='Distribution of Apparent Fishing Hours',
                        color_discrete_sequence=['#42A5F5'],
                        template=PLOTLY_THEME)
    fig3.add_vline(x=df_view['Fishing_Hours'].mean(), line_dash='dash',
                   line_color='red',
                   annotation_text=f"Mean: {df_view['Fishing_Hours'].mean():.1f}h")
    fig3.add_vline(x=df_view['Fishing_Hours'].median(), line_dash='dash',
                   line_color='orange',
                   annotation_text=f"Median: {df_view['Fishing_Hours'].median():.1f}h")
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Top 20 Vessels by Fishing Hours")
    top_vessels = (df_view.nlargest(20, 'Fishing_Hours')
                   [['Vessel Name', 'Flag', 'Gear Type', 'Fishing_Hours', 'Has_IMO']]
                   .reset_index(drop=True))
    top_vessels['IMO Status'] = top_vessels['Has_IMO'].map({1: '✅ Has IMO', 0: '❌ No IMO'})
    top_vessels.drop(columns=['Has_IMO'], inplace=True)
    st.dataframe(top_vessels, use_container_width=True)

    st.subheader("Country × Gear Type Heatmap (Total Fishing Hours)")
    top_countries = df_view['Flag'].value_counts().head(12).index
    top_gears     = df_view['Gear Type'].value_counts().head(10).index
    pivot = (df_view[df_view['Flag'].isin(top_countries) & df_view['Gear Type'].isin(top_gears)]
             .groupby(['Flag', 'Gear Type'])['Fishing_Hours'].sum()
             .unstack(fill_value=0))
    fig4 = px.imshow(pivot, color_continuous_scale='YlOrRd',
                     title='Fishing Hours — Country × Gear Type',
                     template=PLOTLY_THEME, aspect='auto')
    st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — TEMPORAL TRENDS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Temporal Activity Patterns")

    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    monthly = (df_view.groupby('Entry_Month_Name')['Fishing_Hours']
               .agg(['sum', 'mean', 'count'])
               .reindex(month_order, fill_value=0)
               .reset_index())
    monthly.columns = ['Month', 'Total Hours', 'Avg Hours', 'Vessel Count']

    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.bar(monthly, x='Month', y='Total Hours',
                     title='Total Fishing Hours by Month',
                     color='Total Hours', color_continuous_scale='Blues',
                     template=PLOTLY_THEME)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        fig2 = px.line(monthly, x='Month', y='Avg Hours',
                       title='Average Fishing Hours per Entry by Month',
                       markers=True, template=PLOTLY_THEME,
                       color_discrete_sequence=['#EF5350'])
        st.plotly_chart(fig2, use_container_width=True)

    season_order   = ['Winter', 'Spring', 'Summer', 'Autumn']
    season_colours = ['#1565C0', '#4CAF50', '#FF9800', '#9C27B0']
    season_data    = (df_view.groupby('Season')['Fishing_Hours']
                      .sum().reindex(season_order).reset_index())
    season_data.columns = ['Season', 'Total Hours']
    fig3 = px.bar(season_data, x='Season', y='Total Hours',
                  title='Total Fishing Hours by Season',
                  color='Season',
                  color_discrete_sequence=season_colours,
                  template=PLOTLY_THEME)
    st.plotly_chart(fig3, use_container_width=True)

    dow_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    dow_data   = (df_view.groupby('Entry_DayOfWeek')['Fishing_Hours']
                  .mean().reset_index())
    dow_data['Day'] = dow_data['Entry_DayOfWeek'].apply(lambda x: dow_labels[x])
    fig4 = px.bar(dow_data, x='Day', y='Fishing_Hours',
                  title='Average Fishing Hours by Day of Week',
                  color='Fishing_Hours', color_continuous_scale='Purples',
                  template=PLOTLY_THEME)
    st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ANOMALY DETECTION
# Uses the SAVED iso_model (11 IF_FEATURES) — no retraining
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Anomaly Detection — Isolation Forest")
    st.info(
        "Predictions use the **pre-trained** Isolation Forest model loaded from `iso_forest.pkl`. "
        "It expects exactly **11 features** matching the notebook's Section 6."
    )

    if st.button("🔍 Run Isolation Forest Predictions"):
        with st.spinner("Running Isolation Forest on your dataset..."):
            # ── FIX: use IF_FEATURES (11) — NOT MODEL_FEATURES (13) ──────────
            X_if = df[IF_FEATURES].fillna(0)

            df['IF_Raw']        = iso_model.predict(X_if)
            df['IF_Score']      = iso_model.score_samples(X_if)
            df['Anomaly']       = (df['IF_Raw'] == -1).astype(int)
            df['Anomaly_Label'] = df['Anomaly'].map({0: 'Normal', 1: 'Anomaly'})

            n_normal  = (df['Anomaly'] == 0).sum()
            n_anomaly = (df['Anomaly'] == 1).sum()

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Records", f"{len(df):,}")
            c2.metric("🟢 Normal",     f"{n_normal:,}")
            c3.metric("🔴 Anomalies",  f"{n_anomaly:,} ({n_anomaly/len(df)*100:.1f}%)")

            col_a, col_b = st.columns(2)

            with col_a:
                fig = px.scatter(
                    df, x='Fishing_Hours_Log', y='Fishing_Intensity',
                    color='Anomaly_Label',
                    color_discrete_map=COLOUR_MAP,
                    title='Anomaly vs Normal — Fishing Hours vs Intensity',
                    opacity=0.55, template=PLOTLY_THEME
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_b:
                fig2 = px.histogram(
                    df, x='IF_Score', color='Anomaly_Label',
                    color_discrete_map=COLOUR_MAP,
                    barmode='overlay', opacity=0.65,
                    title='Isolation Forest Score Distribution',
                    template=PLOTLY_THEME
                )
                st.plotly_chart(fig2, use_container_width=True)

            anomaly_by_flag = (df[df['Anomaly'] == 1]['Flag']
                               .value_counts().head(15).reset_index())
            anomaly_by_flag.columns = ['Flag', 'Anomaly Count']
            fig3 = px.bar(anomaly_by_flag, x='Flag', y='Anomaly Count',
                          title='Anomalies by Flag State (Top 15)',
                          color='Anomaly Count', color_continuous_scale='Reds',
                          template=PLOTLY_THEME)
            fig3.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig3, use_container_width=True)

            st.session_state['anomaly_ready']    = True
            st.session_state['df_with_anomaly']  = df.copy()
            st.success("✅ Anomaly detection complete! Proceed to the ML Classifiers tab.")


# ─── Helper: preprocess for prediction using saved models ────────────────────
def preprocess_for_prediction(df_input):
    """
    Returns the full dataframe plus two scaled arrays:
      X_if_scaled  — 11 features for IsolationForest
      X_sup_scaled — 13 features for RF / XGB / LGBM
    """
    df_p = df_input.copy()

    # Select feature columns and fill NaN
    X_if  = df_p[IF_FEATURES].fillna(0).values
    X_sup = df_p[MODEL_FEATURES].fillna(0).values

    # scaler was fitted on the 13-feature supervised set
    X_sup_scaled = scaler.transform(X_sup)

    # IsolationForest was trained WITHOUT scaling — use raw values
    return df_p, X_if, X_sup_scaled


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ML CLASSIFIERS
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Supervised ML Classification")
    st.markdown(
        "Uses the **pre-trained** RF, XGBoost, and LightGBM models loaded from disk. "
        "Run **Anomaly Detection** first to generate Isolation Forest labels."
    )

    if 'anomaly_ready' not in st.session_state:
        st.info("⚠️ Please run **Anomaly Detection** in the previous tab first.")
        st.stop()

    df_ml = st.session_state['df_with_anomaly']

    # ── Pre-trained model predictions ────────────────────────────────────────
    st.markdown("### 🔍 Predictions from Saved Models")

    df_pred, X_if_raw, X_sup_sc = preprocess_for_prediction(df_ml)

    df_pred['RF_Pred']   = rf_model.predict(X_sup_sc)
    df_pred['XGB_Pred']  = xgb_model.predict(X_sup_sc)
    df_pred['LGBM_Pred'] = lgbm_model.predict(X_sup_sc)

    # ── FIX: IsolationForest uses 11 IF_FEATURES, not the 13-feature scaled set
    iso_raw              = iso_model.predict(X_if_raw)
    df_pred['IF_Pred']   = np.where(iso_raw == -1, 1, 0)

    df_pred['Ensemble_Vote'] = (
        df_pred[['RF_Pred', 'XGB_Pred', 'LGBM_Pred', 'IF_Pred']].sum(axis=1) >= 2
    ).astype(int)

    df_pred['RF_Score']   = rf_model.predict_proba(X_sup_sc)[:, 1]
    df_pred['XGB_Score']  = xgb_model.predict_proba(X_sup_sc)[:, 1]
    df_pred['LGBM_Score'] = lgbm_model.predict_proba(X_sup_sc)[:, 1]

    st.dataframe(
        df_pred[['Vessel Name', 'Flag', 'Gear Type', 'Fishing_Hours',
                 'RF_Pred', 'XGB_Pred', 'LGBM_Pred', 'IF_Pred', 'Ensemble_Vote']]
        .rename(columns={'Ensemble_Vote': 'FLAGGED (Ensemble)'}),
        use_container_width=True
    )

    flagged_pre = df_pred[df_pred['Ensemble_Vote'] == 1]
    st.metric('🚨 Ensemble-Flagged Vessels', len(flagged_pre))

    st.markdown("---")

    # ── Model Performance — loaded instantly from notebook training ───────────
    st.markdown("### 📋 Model Performance (from Notebook Training)")
    st.caption("Metrics were computed during notebook training and saved to model_metrics.json — no retraining needed.")

    comp = pd.DataFrame(model_metrics)
    comp['CV AUC'] = comp['CV AUC'].apply(lambda x: x if x is not None else '—')
    st.dataframe(comp, use_container_width=True)

    metrics   = ['Test AUC', 'F1', 'Accuracy', 'Precision', 'Recall']
    comp_plot = comp[comp['CV AUC'] != '—'].copy()
    comp_plot[metrics] = comp_plot[metrics].apply(pd.to_numeric, errors='coerce')
    fig_bar = px.bar(
        comp_plot.melt(id_vars='Model', value_vars=metrics,
                       var_name='Metric', value_name='Score'),
        x='Model', y='Score', color='Metric', barmode='group',
        title='Model Performance Comparison (Notebook Results)',
        template=PLOTLY_THEME,
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_bar.update_layout(yaxis_range=[0, 1.1])
    st.plotly_chart(fig_bar, use_container_width=True)


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small>⚓ Maritime Vessel Activity Monitoring · MSc IT with BI · "
    "Robert Gordon University · Supervised by Shahana Bano</small></center>",
    unsafe_allow_html=True
)

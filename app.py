# ─── app.py — Maritime Vessel Activity Monitoring | Streamlit Dashboard ───────
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
import joblib

# ----Load models & preprocessors
rf_model = joblib.load('rf_model.pk1')
xgb_model = joblib.load('xgb_model.pk1')
lgbm_model = joblib.load('lgbm_model.pk1')
iso_model = joblib.load('iso_forest.pk1')
scaler = joblib.load('scaler.pk1')
le_gear = joblib.load('le_gear.pk1')
le_flag = joblib.load('le_flag.pk1')
MODEL_FEATURES = joblib.load('model_features.pk1')
# ─── Config ───────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
COLOUR_MAP = {'Normal': '#2196F3', 'Anomaly': '#F44336'}
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


# ─── Load & Clean (matches your Section 2 + Section 4 exactly) ───────────────
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
    if 'Vessel Name' in df.columns:
        df['Vessel Name'] = df['Vessel Name'].fillna('UNKNOWN')
    if 'Gear Type' in df.columns:
        df['Gear Type'] = df['Gear Type'].fillna('UNKNOWN')
    if 'Flag' in df.columns:
        df['Flag'] = df['Flag'].fillna('UNKNOWN')

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

    df['Entry_Month'] = df['Entry Timestamp'].dt.month
    df['Entry_Month_Name'] = df['Entry Timestamp'].dt.strftime('%b')
    df['Entry_DayOfWeek'] = df['Entry Timestamp'].dt.dayofweek
    df['Entry_Hour'] = df['Entry Timestamp'].dt.hour

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
    df['Has_CallSign'] = df['CallSign'].notnull().astype(int)

    suspicious_gears = ['INCONCLUSIVE', 'FISHING', 'UNKNOWN']
    df['Suspicious_Gear'] = df['Gear Type'].isin(suspicious_gears).astype(int)

    # Label encoding for ML
    le_gear = LabelEncoder()
    le_flag = LabelEncoder()
    df['Gear_Encoded'] = le_gear.fit_transform(df['Gear Type'])
    df['Flag_Encoded'] = le_flag.fit_transform(df['Flag'])

    return df

df = load_and_prepare(uploaded)
st.sidebar.success(f"✅ Loaded {len(df):,} rows × {len(df.columns)} columns")


# ─── Sidebar Filters ──────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.header("🔍 Filters")

flags = ["All"] + sorted(df['Flag'].dropna().unique().tolist())
sel_flag = st.sidebar.selectbox("Flag State", flags)

gears = ["All"] + sorted(df['Gear Type'].dropna().unique().tolist())
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
    c1.metric("Total Vessels", f"{len(df_view):,}")
    c2.metric("Unique MMSIs", f"{df_view['MMSI'].nunique():,}")
    c3.metric("Flag States", f"{df_view['Flag'].nunique():,}")
    c4.metric("No IMO Number", f"{(df_view['Has_IMO'] == 0).sum():,}")

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
        st.plotly_chart(fig, width='stretch')

    with col_b:
        gear_dist = df_view['Gear Type'].value_counts().reset_index()
        gear_dist.columns = ['Gear Type', 'Count']
        fig2 = px.pie(gear_dist, names='Gear Type', values='Count',
                      title='Gear Type Distribution',
                      hole=0.4, template=PLOTLY_THEME)
        st.plotly_chart(fig2, width='stretch')

    # IMO compliance bar
    col_c, col_d = st.columns(2)
    with col_c:
        imo_status = df_view['Has_IMO'].map({1: 'Has IMO', 0: 'No IMO'}).value_counts().reset_index()
        imo_status.columns = ['Status', 'Count']
        fig3 = px.bar(imo_status, x='Status', y='Count',
                      title='IMO Compliance Status',
                      color='Status',
                      color_discrete_map={'Has IMO': '#4CAF50', 'No IMO': '#F44336'},
                      template=PLOTLY_THEME)
        st.plotly_chart(fig3, width='stretch')

    with col_d:
        tier_order = ['Inactive', 'Low', 'Moderate', 'High', 'Very High']
        tier_colours = ['#607D8B', '#4CAF50', '#FF9800', '#F44336', '#B71C1C']
        tier_counts = df_view['Activity_Tier'].value_counts().reindex(tier_order).reset_index()
        tier_counts.columns = ['Tier', 'Count']
        fig4 = px.bar(tier_counts, x='Tier', y='Count',
                      title='Activity Tier Distribution',
                      color='Tier',
                      color_discrete_sequence=tier_colours,
                      template=PLOTLY_THEME)
        st.plotly_chart(fig4, width='stretch')

    with st.expander("🔎 Preview Raw Data (first 100 rows)"):
        st.dataframe(df_view.head(100))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — VESSEL ACTIVITY
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Vessel Fishing Activity Analysis")

    col_a, col_b = st.columns(2)

    with col_a:
        # Top countries by total fishing hours
        country_hours = (df_view.groupby('Flag')['Fishing_Hours']
                         .sum().sort_values(ascending=False).head(15).reset_index())
        country_hours.columns = ['Flag', 'Total Fishing Hours']
        fig = px.bar(country_hours, x='Flag', y='Total Fishing Hours',
                     title='Top 15 Countries — Total Fishing Hours',
                     color='Total Fishing Hours', color_continuous_scale='Oranges',
                     template=PLOTLY_THEME)
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, width='stretch')

    with col_b:
        # Average fishing hours by gear type
        gear_avg = (df_view.groupby('Gear Type')['Fishing_Hours']
                    .mean().sort_values(ascending=False).reset_index())
        gear_avg.columns = ['Gear Type', 'Avg Fishing Hours']
        fig2 = px.bar(gear_avg, x='Gear Type', y='Avg Fishing Hours',
                      title='Average Fishing Hours by Gear Type',
                      color='Avg Fishing Hours', color_continuous_scale='Greens',
                      template=PLOTLY_THEME)
        fig2.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig2, width='stretch')

    # Fishing hours distribution
    fig3 = px.histogram(df_view, x='Fishing_Hours', nbins=60,
                        title='Distribution of Apparent Fishing Hours',
                        color_discrete_sequence=['#42A5F5'],
                        template=PLOTLY_THEME)
    fig3.add_vline(x=df_view['Fishing_Hours'].mean(), line_dash='dash',
                   line_color='red', annotation_text=f"Mean: {df_view['Fishing_Hours'].mean():.1f}h")
    fig3.add_vline(x=df_view['Fishing_Hours'].median(), line_dash='dash',
                   line_color='orange', annotation_text=f"Median: {df_view['Fishing_Hours'].median():.1f}h")
    st.plotly_chart(fig3, width='stretch')

    # Top 20 vessels
    st.subheader("Top 20 Vessels by Fishing Hours")
    top_vessels = (df_view.nlargest(20, 'Fishing_Hours')
                   [['Vessel Name', 'Flag', 'Gear Type', 'Fishing_Hours', 'Has_IMO']]
                   .reset_index(drop=True))
    top_vessels['IMO Status'] = top_vessels['Has_IMO'].map({1: '✅ Has IMO', 0: '❌ No IMO'})
    top_vessels.drop(columns=['Has_IMO'], inplace=True)
    st.dataframe(top_vessels, width='stretch')

    # Country × Gear heatmap
    st.subheader("Country × Gear Type Heatmap (Total Fishing Hours)")
    top_countries = df_view['Flag'].value_counts().head(12).index
    top_gears = df_view['Gear Type'].value_counts().head(10).index
    pivot = (df_view[df_view['Flag'].isin(top_countries) & df_view['Gear Type'].isin(top_gears)]
             .groupby(['Flag', 'Gear Type'])['Fishing_Hours'].sum()
             .unstack(fill_value=0))
    fig4 = px.imshow(pivot, color_continuous_scale='YlOrRd',
                     title='Fishing Hours — Country × Gear Type',
                     template=PLOTLY_THEME, aspect='auto')
    st.plotly_chart(fig4, width='stretch')


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
        st.plotly_chart(fig, width='stretch')

    with col_b:
        fig2 = px.line(monthly, x='Month', y='Avg Hours',
                       title='Average Fishing Hours per Entry by Month',
                       markers=True, template=PLOTLY_THEME,
                       color_discrete_sequence=['#EF5350'])
        st.plotly_chart(fig2, width='stretch')

    # Season
    season_order = ['Winter', 'Spring', 'Summer', 'Autumn']
    season_colours = ['#1565C0', '#4CAF50', '#FF9800', '#9C27B0']
    season_data = (df_view.groupby('Season')['Fishing_Hours']
                   .sum().reindex(season_order).reset_index())
    season_data.columns = ['Season', 'Total Hours']
    fig3 = px.bar(season_data, x='Season', y='Total Hours',
                  title='Total Fishing Hours by Season',
                  color='Season',
                  color_discrete_sequence=season_colours,
                  template=PLOTLY_THEME)
    st.plotly_chart(fig3, width='stretch')

    # Day of week
    dow_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    dow_data = (df_view.groupby('Entry_DayOfWeek')['Fishing_Hours']
                .mean().reset_index())
    dow_data['Day'] = dow_data['Entry_DayOfWeek'].apply(lambda x: dow_labels[x])
    fig4 = px.bar(dow_data, x='Day', y='Fishing_Hours',
                  title='Average Fishing Hours by Day of Week',
                  color='Fishing_Hours', color_continuous_scale='Purples',
                  template=PLOTLY_THEME)
    st.plotly_chart(fig4, width='stretch')


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Anomaly Detection — Isolation Forest")

    IF_FEATURES = [
        'Fishing_Hours_Log', 'Duration_Hours', 'AIS_History_Days',
        'Fishing_Intensity', 'AIS_Rate_PerDay', 'Has_IMO',
        'Has_CallSign', 'Suspicious_Gear', 'Entry_Month',
        'Entry_DayOfWeek', 'Entry_Hour'
    ]

    contamination = st.slider(
        "Contamination — expected anomaly proportion (your notebook uses 0.10)",
        min_value=0.05, max_value=0.20, value=0.10, step=0.01
    )

    if st.button("🔍 Run Isolation Forest"):
        with st.spinner("Running Isolation Forest on your dataset..."):
            X_if = df[IF_FEATURES].fillna(0)

            iso = IsolationForest(
                n_estimators=100,
                contamination=contamination,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
            df['IF_Raw']    = iso.fit_predict(X_if)
            df['IF_Score']  = iso.score_samples(X_if)
            df['Anomaly']   = (df['IF_Raw'] == -1).astype(int)
            df['Anomaly_Label'] = df['Anomaly'].map({0: 'Normal', 1: 'Anomaly'})

            n_normal  = (df['Anomaly'] == 0).sum()
            n_anomaly = (df['Anomaly'] == 1).sum()

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Records", f"{len(df):,}")
            c2.metric("🟢 Normal", f"{n_normal:,}")
            c3.metric("🔴 Anomalies", f"{n_anomaly:,} ({n_anomaly/len(df)*100:.1f}%)")

            col_a, col_b = st.columns(2)

            with col_a:
                fig = px.scatter(
                    df, x='Fishing_Hours_Log', y='Fishing_Intensity',
                    color='Anomaly_Label',
                    color_discrete_map=COLOUR_MAP,
                    title='Anomaly vs Normal — Fishing Hours vs Intensity',
                    opacity=0.55, template=PLOTLY_THEME
                )
                st.plotly_chart(fig, width='stretch')

            with col_b:
                fig2 = px.histogram(
                    df, x='IF_Score', color='Anomaly_Label',
                    color_discrete_map=COLOUR_MAP,
                    barmode='overlay', opacity=0.65,
                    title='Isolation Forest Score Distribution',
                    template=PLOTLY_THEME
                )
                st.plotly_chart(fig2, width='stretch')

            # Anomalies by flag
            anomaly_by_flag = (df[df['Anomaly'] == 1]['Flag']
                               .value_counts().head(15).reset_index())
            anomaly_by_flag.columns = ['Flag', 'Anomaly Count']
            fig3 = px.bar(anomaly_by_flag, x='Flag', y='Anomaly Count',
                          title='Anomalies by Flag State (Top 15)',
                          color='Anomaly Count', color_continuous_scale='Reds',
                          template=PLOTLY_THEME)
            fig3.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig3, width='stretch')

            # Save to session state for ML tab
            st.session_state['anomaly_ready'] = True
            st.session_state['df_with_anomaly'] = df.copy()
            st.success("✅ Anomaly detection complete! Proceed to the ML Classifiers tab.")

def preprocess_for_prediction(df_input):
    """
    Apply the same cleaning + feature engineering from Colab Sections 4 & 5,
    then encode and scale — ready for model prediction.
    """
    df = df_input.copy()

    # ── Section 4 cleaning ────────────────────────────────────────────────────
    time_cols = ['Entry Timestamp', 'Exit Timestamp',
                 'First Transmission Date', 'Last Transmission Date']
    for col in time_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors='coerce')

    df.rename(columns={'Apparent Fishing Hours': 'Fishing_Hours'}, inplace=True)
    df['Vessel Name'] = df['Vessel Name'].fillna('UNKNOWN')
    df['Gear Type']   = df['Gear Type'].fillna('UNKNOWN')
    df['Flag']        = df['Flag'].fillna('UNKNOWN')
    df['Has_IMO']     = df['IMO'].notnull().astype(int)

    # ── Section 5 feature engineering ────────────────────────────────────────
    df['Duration_Hours'] = (
        (df['Exit Timestamp'] - df['Entry Timestamp'])
        .dt.total_seconds() / 3600
    ).clip(lower=0)

    df['AIS_History_Days'] = (
        (df['Last Transmission Date'] - df['First Transmission Date'])
        .dt.total_seconds() / 86400
    ).clip(lower=0)

    df['Entry_Month']     = df['Entry Timestamp'].dt.month
    df['Entry_DayOfWeek'] = df['Entry Timestamp'].dt.dayofweek
    df['Entry_Hour']      = df['Entry Timestamp'].dt.hour

    df['Fishing_Intensity'] = np.where(
        df['Duration_Hours'] > 0,
        df['Fishing_Hours'] / df['Duration_Hours'],
        0
    ).clip(max=1)

    df['AIS_Rate_PerDay'] = np.where(
        df['AIS_History_Days'] > 0,
        df['Fishing_Hours'] / df['AIS_History_Days'],
        0
    )

    df['Fishing_Hours_Log'] = np.log1p(df['Fishing_Hours'])
    df['Has_CallSign']      = df['CallSign'].notnull().astype(int)

    suspicious_gears    = ['INCONCLUSIVE', 'FISHING', 'UNKNOWN']
    df['Suspicious_Gear'] = df['Gear Type'].isin(suspicious_gears).astype(int)

    # ── Section 7 label encoding ──────────────────────────────────────────────
    # Use transform (not fit_transform) — le_gear and le_flag are already fitted
    df['Gear_Encoded'] = df['Gear Type'].apply(
        lambda x: le_gear.transform([x])[0] if x in le_gear.classes_ else -1
    )
    df['Flag_Encoded'] = df['Flag'].apply(
        lambda x: le_flag.transform([x])[0] if x in le_flag.classes_ else -1
    )

    # ── Select and scale features ─────────────────────────────────────────────
    X = df[MODEL_FEATURES].fillna(0).values
    X_scaled = scaler.transform(X)   # transform only — scaler already fitted in Colab

    return df, X_scaled

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ML CLASSIFIERS
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Supervised ML Classification")
    st.markdown("Uses Isolation Forest labels as the target. Trains Random Forest, XGBoost, and LightGBM with SMOTE resampling — matching your notebook exactly.")

    if 'anomaly_ready' not in st.session_state:
        st.info("⚠️ Please run **Anomaly Detection** in the previous tab first.")
    else:
        df_ml = st.session_state['df_with_anomaly']

    df_processed, X_scaled = preprocess_for_prediction(df)

    df_processed['RF_Pred']   = rf_model.predict(X_scaled)
    df_processed['XGB_Pred']  = xgb_model.predict(X_scaled)
    df_processed['LGBM_Pred'] = lgbm_model.predict(X_scaled)

    iso_raw = iso_model.predict(X_scaled)
    df_processed['IF_Pred'] = np.where(iso_raw == -1, 1, 0)

    df_processed['Ensemble_Vote'] = (
        df_processed[['RF_Pred','XGB_Pred','LGBM_Pred','IF_Pred']].sum(axis=1) >= 2
            ).astype(int)

    df_processed['RF_Score']   = rf_model.predict_proba(X_scaled)[:, 1]
    df_processed['XGB_Score']  = xgb_model.predict_proba(X_scaled)[:, 1]
    df_processed['LGBM_Score'] = lgbm_model.predict_proba(X_scaled)[:, 1]

    st.subheader('🔍 Prediction Results')
    st.dataframe(
                    df_processed[['Vessel Name', 'Flag', 'Gear Type', 'Fishing_Hours',
                    'RF_Pred', 'XGB_Pred', 'LGBM_Pred', 'IF_Pred', 'Ensemble_Vote']]
                    .rename(columns={'Ensemble_Vote': 'FLAGGED (Ensemble)'})
                        )

    flagged = df_processed[df_processed['Ensemble_Vote'] == 1]
    st.metric('🚨 Ensemble-Flagged Vessels', len(flagged))

MODEL_FEATURES = [
            'Fishing_Hours_Log', 'Duration_Hours', 'AIS_History_Days',
            'Fishing_Intensity', 'AIS_Rate_PerDay', 'Has_IMO',
            'Has_CallSign', 'Suspicious_Gear', 'Entry_Month',
            'Entry_DayOfWeek', 'Entry_Hour', 'Gear_Encoded', 'Flag_Encoded'
        ]

X = df_ml[MODEL_FEATURES].fillna(0).values
y = df_ml['Anomaly'].values

st.write(f"**Class distribution:** Normal = {(y==0).sum():,} | Anomaly = {(y==1).sum():,}")

@st.cache_data
def run_ml_pipeline(X, y):
        X_train_raw, X_test, y_train_raw, y_test = train_test_split(
            X, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y
        )
        smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
        X_train, y_train = smote.fit_resample(X_train_raw, y_train_raw)
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc  = scaler.transform(X_test)
        return X_train_sc, X_test_sc, y_train, y_test, scaler

with tab5:
        # ....x, y definitions above....


        if st.button("🚀 Train RF · XGBoost · LightGBM"):
            with st.spinner("Splitting data, applying SMOTE, training 3 models + IF baseline..."):

                X_train_sc, X_test_sc, y_train, y_test, scaler = run_ml_pipeline(X, y)

                def evaluate_model(name, clf):
                    clf.fit(X_train_sc, y_train)
                    y_pred = clf.predict(X_test_sc)
                    y_prob = clf.predict_proba(X_test_sc)[:, 1]
                    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
                    cv_auc = cross_val_score(clf, X_train_sc, y_train, cv=cv,
                                             scoring='roc_auc', n_jobs=-1)
                    return {
                        'name': name, 'clf': clf,
                        'pred': y_pred, 'prob': y_prob,
                        'cv_auc': cv_auc.mean(), 'cv_std': cv_auc.std(),
                        'test_auc': roc_auc_score(y_test, y_prob),
                        'test_f1':  f1_score(y_test, y_pred),
                        'test_acc': accuracy_score(y_test, y_pred),
                        'precision': precision_score(y_test, y_pred, zero_division=0),
                        'recall':    recall_score(y_test, y_pred, zero_division=0),
                    }

                scale_pos = (y_train == 0).sum() / (y_train == 1).sum()

                models = [
                    ('Random Forest', RandomForestClassifier(
                         n_estimators=100,        # down from 400
                        min_samples_leaf=4,      # up from 2 (faster, less memory)
                        max_depth=10,            # added — prevents deep trees
                        class_weight='balanced',
                        random_state=RANDOM_STATE, n_jobs=-1)),
                    ('XGBoost', xgb.XGBClassifier(
                        n_estimators=100,        # down from 400
                        learning_rate=0.1,       # up from 0.05 (fewer trees needed)
                        max_depth=4,             # down from 6
                        subsample=0.8, colsample_bytree=0.8,
                        scale_pos_weight=scale_pos,
                        use_label_encoder=False, eval_metric='logloss',
                        random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)),
                    ('LightGBM', lgb.LGBMClassifier(
                        n_estimators=100,        # down from 400
                        learning_rate=0.1,       # up from 0.05
                        num_leaves=31,           # down from 63
                        subsample=0.8, colsample_bytree=0.8,
                        class_weight='balanced',
                        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)),
                ]

                all_results = [evaluate_model(name, clf) for name, clf in models]

                # Isolation Forest baseline
                iso_base = IsolationForest(n_estimators=100, contamination=0.10,
                                           random_state=RANDOM_STATE, n_jobs=-1)
                iso_base.fit(X_train_sc)
                y_if_pred = np.where(iso_base.predict(X_test_sc) == -1, 1, 0)
                y_if_prob = -iso_base.score_samples(X_test_sc)
                all_results.append({
                    'name': 'Isolation Forest (baseline)',
                    'pred': y_if_pred, 'prob': y_if_prob,
                    'cv_auc': None, 'cv_std': None,
                    'test_auc': roc_auc_score(y_test, y_if_prob),
                    'test_f1':  f1_score(y_test, y_if_pred),
                    'test_acc': accuracy_score(y_test, y_if_pred),
                    'precision': precision_score(y_test, y_if_pred, zero_division=0),
                    'recall':    recall_score(y_test, y_if_pred, zero_division=0),
                    'clf': iso_base
                })

                # ── Model Comparison Table ────────────────────────────────────
                st.markdown("### 📋 Model Comparison")
                comp = pd.DataFrame([{
                    'Model':     r['name'],
                    'CV AUC':    round(r['cv_auc'], 4) if r['cv_auc'] else '—',
                    'Test AUC':  round(r['test_auc'], 4),
                    'F1':        round(r['test_f1'], 4),
                    'Accuracy':  round(r['test_acc'], 4),
                    'Precision': round(r['precision'], 4),
                    'Recall':    round(r['recall'], 4),
                } for r in all_results])
                st.dataframe(comp, width='stretch')

                # ── Grouped bar chart ─────────────────────────────────────────
                metrics = ['Test AUC', 'F1', 'Accuracy', 'Precision', 'Recall']
                comp_plot = comp[comp['CV AUC'] != '—']  # exclude IF baseline from chart
                fig_bar = px.bar(
                    comp_plot.melt(id_vars='Model', value_vars=metrics,
                                   var_name='Metric', value_name='Score'),
                    x='Model', y='Score', color='Metric', barmode='group',
                    title='Model Performance Comparison',
                    template=PLOTLY_THEME,
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig_bar.update_layout(yaxis_range=[0, 1.1])
                st.plotly_chart(fig_bar, width='stretch')

                # ── ROC Curves ────────────────────────────────────────────────
                st.markdown("### 📈 ROC Curves")
                colours = ['#42A5F5', '#EF5350', '#4CAF50', '#FF9800']
                fig_roc = go.Figure()
                for res, colour in zip(all_results, colours):
                    fpr, tpr, _ = roc_curve(y_test, res['prob'])
                    fig_roc.add_trace(go.Scatter(
                        x=fpr, y=tpr, mode='lines',
                        name=f"{res['name']} (AUC={res['test_auc']:.4f})",
                        line=dict(color=colour, width=2)
                    ))
                fig_roc.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1], mode='lines',
                    line=dict(dash='dash', color='grey'), name='Random'
                ))
                fig_roc.update_layout(
                    title='ROC Curves — All Models (Test Set)',
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate',
                    template=PLOTLY_THEME, height=500
                )
                st.plotly_chart(fig_roc, width='stretch')

                # ── Confusion Matrices ────────────────────────────────────────
                st.markdown("### 🔢 Confusion Matrices")
                sup_results = [r for r in all_results if r['name'] != 'Isolation Forest (baseline)']
                cols = st.columns(len(sup_results))
                for col, res in zip(cols, sup_results):
                    cm = confusion_matrix(y_test, res['pred'])
                    fig_cm = px.imshow(
                        cm, text_auto=True,
                        labels=dict(x='Predicted', y='Actual'),
                        x=['Normal', 'Anomaly'],
                        y=['Normal', 'Anomaly'],
                        title=res['name'],
                        color_continuous_scale='Blues',
                        template=PLOTLY_THEME
                    )
                    col.plotly_chart(fig_cm, width='stretch')

                # ── Feature Importances ───────────────────────────────────────
                st.markdown("### 🔑 Feature Importances")
                imp_cols = st.columns(3)
                for col, res in zip(imp_cols, all_results[:3]):
                    if hasattr(res['clf'], 'feature_importances_'):
                        fi = pd.Series(res['clf'].feature_importances_,
                                       index=MODEL_FEATURES).sort_values()
                        fig_fi = px.bar(fi, orientation='h',
                                        title=res['name'],
                                        template=PLOTLY_THEME)
                        col.plotly_chart(fig_fi, width='stretch')

                # ── Ensemble Vote ─────────────────────────────────────────────
                st.markdown("### 🗳️ Ensemble-Flagged Vessels")
                X_full_sc = scaler.transform(df_ml[MODEL_FEATURES].fillna(0).values)
                df_ml['RF_Pred']   = all_results[0]['clf'].predict(X_full_sc)
                df_ml['XGB_Pred']  = all_results[1]['clf'].predict(X_full_sc)
                df_ml['LGBM_Pred'] = all_results[2]['clf'].predict(X_full_sc)
                df_ml['IF_Pred']   = df_ml['Anomaly']

                df_ml['Ensemble_Vote'] = (
                    df_ml[['RF_Pred', 'XGB_Pred', 'LGBM_Pred', 'IF_Pred']].sum(axis=1) >= 2
                ).astype(int)

                flagged = (df_ml[df_ml['Ensemble_Vote'] == 1]
                           [['Vessel Name', 'Flag', 'Gear Type', 'Fishing_Hours',
                             'Has_IMO', 'RF_Pred', 'XGB_Pred', 'LGBM_Pred', 'IF_Pred']]
                           .sort_values('Fishing_Hours', ascending=False)
                           .head(50).reset_index(drop=True))
                flagged['Has IMO'] = flagged['Has_IMO'].map({1: '✅', 0: '❌'})
                flagged.drop(columns=['Has_IMO'], inplace=True)

                st.metric("Ensemble-Flagged Vessels",
                          f"{df_ml['Ensemble_Vote'].sum():,} ({df_ml['Ensemble_Vote'].mean()*100:.1f}%)")
                st.dataframe(flagged, width='stretch')

                st.success("✅ All models trained and evaluated!")


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small>⚓ Maritime Vessel Activity Monitoring · MSc IT with BI · "
    "Robert Gordon University · Supervised by Shahana Bano</small></center>",
    unsafe_allow_html=True
)
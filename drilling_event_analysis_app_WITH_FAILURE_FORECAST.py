import pandas as pd
import os
from dotenv import load_dotenv
import streamlit as st
from openai import AzureOpenAI
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ---------------------------
# Page Setup & Azure Client
# ---------------------------
st.set_page_config(layout="wide")
load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-raj")

# ---------------------------
# Helper Functions
# ---------------------------
def get_genai_response(prompt, max_tokens=250, temperature=0.4):
    """Wrapper for Azure OpenAI chat completions."""
    try:
        resp = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ GenAI error: {e}"

def is_exception(row):
    """Business logic to flag drilling anomalies as exceptions."""
    return (
        row.get("event_type", "") in ["Overload", "Leak", "Vibration Surge"] or
        ">" in str(row.get("breach", "")) or
        row.get("severity", "").lower() in ["critical", "high"]
    )

@st.cache_data(show_spinner=False)
def analyze_event_cached(row_dict):
    """Use GenAI to analyze a single drilling event (cached for performance)."""
    prompt = (
        f"You are a drilling reliability assistant. Analyze the following drilling event:\n\n"
        f"Timestamp: {row_dict.get('timestamp')}\n"
        f"Well Site: {row_dict.get('well_site')}\n"
        f"Zone: {row_dict.get('zone')}\n"
        f"Equipment: {row_dict.get('Equipment')}\n"
        f"Parameter: {row_dict.get('Parameter')}\n"
        f"Anomaly Type: {row_dict.get('event_type')}\n"
        f"Threshold Breach: {row_dict.get('breach')}\n"
        f"Severity: {row_dict.get('severity')}\n"
        f"Duration (min): {row_dict.get('duration')}\n\n"
        f"Provide root cause analysis and suggested action."
    )
    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=180,
            temperature=0.4,
        )
        reply = response.choices[0].message.content
        if "Action:" in reply:
            parts = reply.split("Action:")
            return parts[0].strip(), parts[1].strip()
        return reply, "N/A"
    except Exception as e:
        return "GenAI error", str(e)

# ---------------------------
# Tabs
# ---------------------------
main_tabs = st.tabs(["📌 Overview", "📥 Load Drilling Log"])

# ---------------------------
# Tab 0 - Overview
# ---------------------------
with main_tabs[0]:
    st.title("🛢️ AI-Powered Drilling Event Exception Analyzer")
    st.subheader("📌 Application Overview")
    st.markdown("""
### 🎯 Purpose
Detect anomalies in drilling logs, analyze root causes, and get GenAI-powered suggestions for safer, more efficient operations.

**Tabs:**
- 🧾 Exceptions → GenAI analysis of anomalies  
- 📊 Trends → Time-series anomaly visualization  
- 🔁 Repetitions → Repeated anomaly detection  
- 🧠 Clustering → KMeans grouping of anomalies  
- 📌 Summary → AI summary of risks  
- 📋 Insights → Executive-level insights  
- 🔮 Forecast → Predictive failure outlook  
""")

# ---------------------------
# Tab 1 - Load Drilling Log
# ---------------------------
with main_tabs[1]:
    uploaded_file = st.file_uploader("📁 Upload your drilling event log", type=["csv", "xlsx"])

# ---------------------------
# Main Logic after Upload
# ---------------------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    df.columns = [col.strip() for col in df.columns]

    # Normalize column names
    df.rename(columns={
        'Timestamp': 'timestamp',
        'Well_Site': 'well_site',
        'Zone': 'zone',
        'Anomaly Type': 'event_type',
        'Severity': 'severity',
        'Duration (min)': 'duration',
        'Threshold Breach': 'breach',
        'Event Code': 'event_code'
    }, inplace=True)

    # Filters
    all_wells = df["well_site"].unique().tolist()
    selected_wells = st.multiselect("Select Well Sites", options=["All"] + all_wells, default=["All"])
    if "All" not in selected_wells:
        df = df[df["well_site"].isin(selected_wells)]

    all_event_types = df["event_type"].unique().tolist()
    selected_event_types = st.multiselect("Select Event Types", options=["All"] + all_event_types, default=["All"])
    if "All" not in selected_event_types:
        df = df[df["event_type"].isin(selected_event_types)]

    # Exception Analysis
    results = []
    analyze_all = st.checkbox("🔄 Analyze all exceptions (may be slow)", value=False)
    max_rows = None if analyze_all else 20
    analyzed_rows = 0

    with st.spinner("Analyzing exceptions..."):
        for _, row in df.iterrows():
            if is_exception(row):
                if max_rows is not None and analyzed_rows >= max_rows:
                    break
                cause, action = analyze_event_cached(row.to_dict())
                record = row.to_dict()
                record["Root Cause"] = cause
                record["Suggested Action"] = action
                results.append(record)
                analyzed_rows += 1

    result_df = pd.DataFrame(results) if results else pd.DataFrame()

    # ---------------------------
    # Sub-Tabs
    # ---------------------------
    sub_tabs = st.tabs([
        "🧾 Exceptions", "📊 Trends", "🔁 Repetitions", "🧠 Clustering", "📌 Summary", "📋 Insights", "🔮 Forecast"
    ])

    # ---------------------------
    # Tab: Exceptions
    # ---------------------------
    with sub_tabs[0]:
        st.subheader("🧾 Detected Exceptions")
        st.dataframe(result_df)
        if not result_df.empty:
            st.download_button("📥 Download Exception Report", result_df.to_csv(index=False), "drilling_exceptions.csv")

    # ---------------------------
    # Tab: Trends
    # ---------------------------
    with sub_tabs[1]:
        st.subheader("📊 Trends Over Time")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['event_type'] = df['event_type'].fillna("Unknown")
        trend = df.groupby([pd.Grouper(key='timestamp', freq='D'), 'event_type']).size().unstack(fill_value=0)
        st.line_chart(trend)

    # ---------------------------
    # Tab: Repetitions
    # ---------------------------
    with sub_tabs[2]:
        st.subheader("🔁 Repetitions")
        repeated = df.groupby(['well_site', 'event_type']).size().reset_index(name='count')
        repeated = repeated[repeated['count'] > 1]
        st.dataframe(repeated)

    # ---------------------------
    # Tab: Clustering
    # ---------------------------
    with sub_tabs[3]:
        st.subheader("🧠 Clustering by Duration")
        if "duration" in df.columns:
            features = df[['duration']].fillna(0)
            scaler = StandardScaler()
            scaled = scaler.fit_transform(features)
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            df['cluster'] = kmeans.fit_predict(scaled)
            st.dataframe(df[['well_site', 'event_type', 'duration', 'cluster']])
        else:
            st.warning("⚠️ No 'duration' column available for clustering.")

    # ---------------------------
    # Tab: Summary
    # ---------------------------
    with sub_tabs[4]:
        st.subheader("📌 GenAI Summary")
        if not result_df.empty:
            sample = result_df[['event_type', 'zone', 'severity']].head(10).to_string(index=False)
            summary_prompt = f"As an AI drilling analyst, review {len(result_df)} exceptions and summarize:\n{sample}"
            st.info(get_genai_response(summary_prompt))
        else:
            st.warning("⚠️ No exceptions found to summarize.")

    # ---------------------------
    # Tab: Insights
    # ---------------------------
    with sub_tabs[5]:
        st.subheader("📋 Manager's Insights")
        if not result_df.empty:
            sample = result_df[['well_site', 'event_type', 'zone', 'severity']].head(10).to_string(index=False)
            insight_prompt = f"Provide actionable safety & reliability insights for:\n{sample}"
            st.success(get_genai_response(insight_prompt, max_tokens=300))
        else:
            st.warning("⚠️ No exception data available for insights.")

    # ---------------------------
    # Tab: Forecast
    # ---------------------------
    with sub_tabs[6]:
        st.subheader("🔮 Forecast Future Drilling Risks")
        forecast_prompt = (
            f"You are a predictive drilling analyst. Forecast risks for the next 30 days and recommend safety actions.\n\n"
            f"Recent Drilling Events:\n{df[['timestamp','well_site','zone','event_type','breach','severity']].tail(25).to_string(index=False)}"
        )
        st.markdown(get_genai_response(forecast_prompt, max_tokens=500))

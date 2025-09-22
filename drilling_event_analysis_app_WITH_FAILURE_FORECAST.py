import pandas as pd
import os
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# -----------------------------
# Streamlit + OpenAI setup
# -----------------------------
st.set_page_config(layout="wide")
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------------
# Tabs
# -----------------------------
main_tabs = st.tabs(["📌 Overview", "📥 Load Drilling Log"])

with main_tabs[0]:
    st.title("🛢️ AI-Powered Drilling Event Exception Analyzer")
    st.subheader("📌 Application Overview")
    st.markdown("""
### 🎯 Purpose
This AI-powered drilling event analysis application helps engineers and operators:
- Detect anomalies and exceptions in drilling logs
- Analyze root causes with GenAI suggestions
- Visualize event trends
- Identify repetitive risks
- Cluster anomalies by duration
- Forecast failures before they happen

---

### 📥 Inputs
Upload a drilling log (`.csv` or `.xlsx`) with at least:
- Timestamp, Well_Site, Zone, Equipment, Parameter, Anomaly Type, Threshold Breach, Severity, Duration (min)

---

### 📊 Tabs
- **🧾 Exceptions** → Show anomalies with GenAI RCA + action
- **📊 Trends** → Frequency of anomalies over time
- **🔁 Repetitions** → Repeated anomalies by well/type
- **🧠 Clustering** → KMeans grouping by duration
- **📌 Summary** → GenAI summary of risks/issues
- **📋 Insights** → Management insights
- **🔮 Forecast** → Forward-looking risk forecast
""")

# -----------------------------
# File uploader
# -----------------------------
with main_tabs[1]:
    uploaded_file = st.file_uploader("📁 Upload drilling log", type=["csv", "xlsx"])

# -----------------------------
# Load + preprocess data
# -----------------------------
if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    df.columns = [c.strip() for c in df.columns]
    df.rename(columns={
        'Timestamp': 'timestamp',
        'Well_Site': 'well_site',
        'Zone': 'zone',
        'Anomaly Type': 'event_type',
        'Severity': 'severity',
        'Duration (min)': 'duration',
        'Threshold Breach': 'breach'
    }, inplace=True)

    # -----------------------------
    # Filters
    # -----------------------------
    wells = df["well_site"].unique().tolist()
    selected_wells = st.multiselect("Select Well Sites", ["All"] + wells, default=["All"])
    if "All" not in selected_wells:
        df = df[df["well_site"].isin(selected_wells)]

    events = df["event_type"].unique().tolist()
    selected_events = st.multiselect("Select Event Types", ["All"] + events, default=["All"])
    if "All" not in selected_events:
        df = df[df["event_type"].isin(selected_events)]

    # -----------------------------
    # Exception detection logic
    # -----------------------------
    def is_exception(row):
        return (
            str(row.get("event_type", "")).lower() in ["overload", "leak", "vibration surge"] or
            ">" in str(row.get("breach", "")) or
            str(row.get("severity", "")).lower() in ["critical", "high"]
        )

    @st.cache_data(show_spinner=False)
    def analyze_event_cached(row_dict):
        prompt = (
            f"You are a drilling reliability assistant. Analyze this event:\n\n"
            f"Timestamp: {row_dict.get('timestamp')}\n"
            f"Well Site: {row_dict.get('well_site')}\n"
            f"Zone: {row_dict.get('zone')}\n"
            f"Equipment: {row_dict.get('Equipment','N/A')}\n"
            f"Parameter: {row_dict.get('Parameter','N/A')}\n"
            f"Anomaly Type: {row_dict.get('event_type')}\n"
            f"Threshold Breach: {row_dict.get('breach')}\n"
            f"Severity: {row_dict.get('severity')}\n"
            f"Duration (min): {row_dict.get('duration')}\n\n"
            f"Provide root cause and suggested action."
        )
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
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

    results = []
    analyze_all = st.checkbox("🔄 Analyze all exceptions (slow)", value=False)
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

    # -----------------------------
    # Sub-tabs
    # -----------------------------
    sub_tabs = st.tabs([
        "🧾 Exceptions", "📊 Trends", "🔁 Repetitions", "🧠 Clustering",
        "📌 Summary", "📋 Insights", "🔮 Forecast"
    ])

    with sub_tabs[0]:
        st.subheader("🧾 Exceptions")
        st.dataframe(result_df)
        if not result_df.empty:
            st.download_button("📥 Download Report", result_df.to_csv(index=False), "exceptions.csv")

    with sub_tabs[1]:
        st.subheader("📊 Trends Over Time")
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors="coerce")
        trend = df.groupby([pd.Grouper(key='timestamp', freq='D'), 'event_type']).size().unstack(fill_value=0)
        st.line_chart(trend)

    with sub_tabs[2]:
        st.subheader("🔁 Repetitions")
        repeated = df.groupby(['well_site', 'event_type']).size().reset_index(name='count')
        st.dataframe(repeated[repeated['count'] > 1])

    with sub_tabs[3]:
        st.subheader("🧠 Clustering by Duration")
        if "duration" in df.columns:
            features = pd.to_numeric(df['duration'], errors="coerce").fillna(0).to_frame()
            if features['duration'].sum() == 0:
                st.warning("⚠️ No valid numeric durations available for clustering.")
            else:
                scaler = StandardScaler()
                scaled = scaler.fit_transform(features)

                n_clusters = 3 if len(features) >= 3 else max(1, len(features))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                df['cluster'] = kmeans.fit_predict(scaled)

                st.dataframe(df[['well_site', 'event_type', 'duration', 'cluster']])
        else:
            st.warning("⚠️ No 'duration' column in data.")

    with sub_tabs[4]:
        st.subheader("📌 GenAI Summary")
        if not result_df.empty:
            summary_prompt = (
                f"Review {len(result_df)} exceptions. "
                f"Highlight zones of concern, recurring issues, and urgent actions.\n\n"
                f"Sample:\n{result_df[['event_type','zone','severity']].head(10).to_string(index=False)}"
            )
            try:
                summary = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": summary_prompt}],
                    max_tokens=250,
                    temperature=0.4,
                )
                st.info(summary.choices[0].message.content)
            except Exception as e:
                st.warning(f"Summary error: {e}")

    with sub_tabs[5]:
        st.subheader("📋 Manager's Insights")
        if not result_df.empty:
            insight_prompt = (
                f"Give actionable insights for these {len(result_df)} exceptions "
                f"to improve safety and reduce downtime.\n\n"
                f"{result_df[['well_site','event_type','zone','severity']].head(10).to_string(index=False)}"
            )
            try:
                insights = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": insight_prompt}],
                    max_tokens=300,
                    temperature=0.4,
                )
                st.success(insights.choices[0].message.content)
            except Exception as e:
                st.warning(f"Insight error: {e}")

    with sub_tabs[6]:
        st.subheader("🔮 Forecast Risks")
        forecast_prompt = (
            f"Forecast risks for the next 30 days from recent drilling events.\n\n"
            f"{df[['timestamp','well_site','zone','event_type','breach','severity']].tail(25).to_string(index=False)}"
        )
        try:
            forecast = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": forecast_prompt}],
                max_tokens=500,
                temperature=0.4,
            )
            st.markdown(forecast.choices[0].message.content)
        except Exception as e:
            st.error(f"❌ Forecast error: {e}")

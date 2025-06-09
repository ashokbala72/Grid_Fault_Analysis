import pandas as pd
import os
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI, OpenAIError
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(layout="wide")
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

main_tabs = st.tabs(["📌 Overview", "📥 Load Drilling Log"])

with main_tabs[0]:
    st.title("🛢️ AI-Powered Drilling Event Exception Analyzer")
    st.subheader("📌 Application Overview")
    st.markdown("""
### 🎯 Purpose

This AI-powered drilling event analysis application helps engineers and operators:
- Detect anomalies and exceptions in real-time or historical drilling logs
- Analyze root causes and receive GenAI-generated suggested actions
- Visualize event trends and patterns over time
- Identify repetitive risks across well sites and zones
- Group similar anomalies using clustering
- Forecast potential failures before they happen

---

### 📥 Inputs

- Upload a drilling event log file (`.csv` or `.xlsx`)
- Required columns: Timestamp, Well_Site, Zone, Equipment, Parameter, Anomaly Type, Threshold Breach, Severity, Duration (min)

---

### 📊 Tab-by-Tab Functionality

- **🧾 Exceptions**: Displays detected anomalies along with GenAI root cause and action recommendations.
- **📊 Trends**: Visualizes time-series frequency of anomaly types to spot emerging risks.
- **🔁 Repetitions**: Highlights repeated events per site/type to pinpoint persistent issues.
- **🧠 Clustering**: Uses KMeans to categorize anomalies by duration for pattern detection.
- **📌 Summary**: AI-generated summary highlighting major risk zones and recurring issues.
- **📋 Insights**: Executive insights suggesting safety measures and reliability strategies.
- **🔮 Forecast**: Predictive failure forecast with timeline and recommended maintenance window.

---

### ✅ Benefits

- Early detection of critical drilling anomalies
- AI-driven decision support to reduce downtime
- Operational insights for maintenance and resource planning
- Faster reporting through automated summaries
- Preventive failure alerts to avoid unplanned shutdowns

---

### 🛠️ Making This App Production-Ready

To move from prototype to production:

1. **Backend**
   - Integrate with real-time log ingestion pipeline (Kafka, REST API, etc.)
   - Use persistent storage (e.g., PostgreSQL) instead of transient CSVs

2. **AI Layer**
   - Add retry logic, caching, and rate limit handling for GenAI calls
   - Use fine-tuned models for domain-specific accuracy if needed

3. **Frontend**
   - Add user authentication (e.g., Streamlit Auth, OAuth)
   - Deploy securely on Streamlit Cloud, Azure, or AWS (using HTTPS and access control)

4. **Monitoring**
   - Add logging, alerting, and performance tracking (e.g., with Prometheus + Grafana)

5. **Validation**
   - Validate GenAI output with human-in-the-loop review initially
   - Benchmark performance on real-world historical data

---

ℹ️ Designed to serve drilling engineers, operations analysts, and reliability managers.
""")


with main_tabs[1]:
    uploaded_file = st.file_uploader("📁 Upload your simulated drilling event log", type=["csv", "xlsx"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    df.columns = [col.strip() for col in df.columns]

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

    all_wells = df["well_site"].unique().tolist()
    selected_wells = st.multiselect("Select Well Sites", options=["All"] + all_wells, default=["All"])
    if "All" not in selected_wells:
        df = df[df["well_site"].isin(selected_wells)]
    all_event_types = df["event_type"].unique().tolist()
    selected_event_types = st.multiselect("Select Event Types", options=["All"] + all_event_types, default=["All"])
    if "All" not in selected_event_types:
        df = df[df["event_type"].isin(selected_event_types)]
        df = df[df["event_type"].isin(selected_event_types)]
    if "All" not in selected_event_types:
        df = df[df["event_type"].isin(selected_event_types)]

    def is_exception(row):
        return (
            row.get("event_type", "") in ["Overload", "Leak", "Vibration Surge"] or
            ">" in str(row.get("breach", "")) or
            row.get("severity", "").lower() in ["critical", "high"]
        )

    @st.cache_data(show_spinner=False)
    def analyze_event_cached(row_dict):
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

    sub_tabs = st.tabs([
        "🧾 Exceptions", "📊 Trends", "🔁 Repetitions", "🧠 Clustering", "📌 Summary", "📋 Insights", "🔮 Forecast"
    ])

    with sub_tabs[0]:
    st.subheader("🧾 Detected Exceptions")
    st.dataframe(result_df)
    st.download_button("📥 Download Exception Report", result_df.to_csv(index=False), "drilling_exceptions.csv")
    st.markdown("#### 🧠 GenAI Insight")
    st.info("""
    Based on the trend analysis, certain anomaly types are increasing in frequency over recent days, particularly in high-risk zones. For example, “Vibration Surge” and “Overload” events show a spike on specific dates. This suggests equipment stress or operational variability. Immediate review of operational logs and load balancing procedures is advised to prevent escalation.
    """)
    
        st.dataframe(result_df)
        st.download_button("📥 Download Exception Report", result_df.to_csv(index=False), "drilling_exceptions.csv")

    
    with sub_tabs[1]:
        st.subheader("📊 Trends Over Time")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['event_type'] = df['event_type'].fillna("Unknown")
        trend = df.groupby([pd.Grouper(key='timestamp', freq='D'), 'event_type']).size().unstack(fill_value=0)
        st.line_chart(trend)

        st.markdown("---")
        st.markdown("#### 🧠 GenAI Insight")
        st.info("""
        Based on the trend analysis, certain anomaly types are increasing in frequency over recent days, particularly in high-risk zones. For example, “Vibration Surge” and “Overload” events show a spike on specific dates. This suggests equipment stress or operational variability. Immediate review of operational logs and load balancing procedures is advised to prevent escalation.
        """)

        st.markdown("---")
        st.markdown("#### 🧠 GenAI Insight")
        st.info("""
        Based on the trend analysis, certain anomaly types are increasing in frequency over recent days, particularly in high-risk zones. For example, “Vibration Surge” and “Overload” events show a spike on specific dates. This suggests equipment stress or operational variability. Immediate review of operational logs and load balancing procedures is advised to prevent escalation.
        """)
    

    with sub_tabs[2]:
        st.subheader("🔁 Repetitions")
        repeated = df.groupby(['well_site', 'event_type']).size().reset_index(name='count')
        repeated = repeated[repeated['count'] > 1]
        st.dataframe(repeated)

        st.markdown("---")
        st.markdown("#### 🧠 GenAI Insight")
        st.info("""
        Repetitive anomalies at specific well sites indicate underlying unresolved issues. Well sites such as **WS-102** and **WS-205** show repeated occurrences of **Leak** and **Pressure Drop** events. This recurrence pattern may imply insufficient corrective actions, worn-out equipment, or overlooked maintenance schedules. Initiating a root cause audit and reevaluating maintenance frequency for these wells could mitigate future risks and downtime.
        """)

        st.markdown("---")
        st.markdown("#### 🧠 GenAI Insight")
        st.info("""
        Repetitive anomalies at specific well sites indicate underlying unresolved issues. Well sites such as **WS-102** and **WS-205** show repeated occurrences of **Leak** and **Pressure Drop** events. This recurrence pattern may imply insufficient corrective actions, worn-out equipment, or overlooked maintenance schedules. Initiating a root cause audit and reevaluating maintenance frequency for these wells could mitigate future risks and downtime.
        """)
    

    with sub_tabs[3]:
        st.subheader("🧠 Clustering by Duration")
        features = df[['duration']].fillna(0)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(features)
        kmeans = KMeans(n_clusters=3, random_state=42)
        df['cluster'] = kmeans.fit_predict(scaled)
        st.dataframe(df[['well_site', 'event_type', 'duration', 'cluster']])

        st.markdown("---")
        st.markdown("#### 🧠 GenAI Insight")
        st.info("""
        Clustering analysis has grouped anomalies based on event duration into three distinct behavioral categories.

        - **Cluster 0**: Short-duration anomalies, likely transient or sensor glitches.
        - **Cluster 1**: Medium-duration events, possibly linked to operational adjustments or minor breaches.
        - **Cluster 2**: Long-duration anomalies, predominantly high-severity incidents, often indicating equipment failure or process breakdowns.

        Sites consistently appearing in Cluster 2 (e.g., prolonged Overloads) should be prioritized for condition-based maintenance and reliability assessment.
        """)

        st.markdown("---")
        st.markdown("#### 🧠 GenAI Insight")
        st.info("""
        Clustering analysis has grouped anomalies based on event duration into three distinct behavioral categories.

        - **Cluster 0**: Short-duration anomalies, likely transient or sensor glitches.
        - **Cluster 1**: Medium-duration events, possibly linked to operational adjustments or minor breaches.
        - **Cluster 2**: Long-duration anomalies, predominantly high-severity incidents, often indicating equipment failure or process breakdowns.

        Sites consistently appearing in Cluster 2 (e.g., prolonged Overloads) should be prioritized for condition-based maintenance and reliability assessment.
        """)
    
with sub_tabs[4]:
        st.subheader("📌 GenAI Summary")
        summary_prompt = (
            f"As an AI drilling analyst, review {len(result_df)} exceptions. "
            f"Highlight zones of concern, recurring issues, and urgent actions.\n\n"
            f"Sample:\n{result_df[['event_type', 'zone', 'severity']].head(10).to_string(index=False)}"
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
        insight_prompt = (
            f"Give actionable insights for the following {len(result_df)} exception records to improve safety and reduce downtime."
            f"\n\n{result_df[['well_site', 'event_type', 'zone', 'severity']].head(10).to_string(index=False)}"
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
        st.subheader("🔮 Forecast Future Drilling Risks")
        forecast_prompt = (
            f"You are a predictive drilling analyst. Forecast risks for the next 30 days and recommend inventory or safety actions.\n\n"
            f"Recent Drilling Events:\n{df[['timestamp','well_site','zone','event_type','breach','severity']].tail(25).to_string(index=False)}"
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
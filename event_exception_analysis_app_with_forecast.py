import pandas as pd
import os
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI, OpenAIError
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd
import os
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI, OpenAIError
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(layout="wide")


# Merged: Original + Forecast Future Issues Tab

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


main_tabs = st.tabs(["📌 Overview", "📥 Load Event Log"])

with main_tabs[0]:
    st.title("⚡ AI-Powered Grid Event Exception Analyzer")
    st.subheader("📌 Application Overview")
    st.markdown("""
### 🎯 Purpose
This AI-powered tool assists grid operators in identifying and analyzing exceptional grid events using intelligent automation. It flags outages, overloads, and anomalies, and provides root cause analysis and actionable insights using GenAI.

### 📥 Inputs
- Grid event log files (`.csv` or `.xlsx`)
- Fields: timestamp, substation, event type, fault code, load (MW)
- Filters to scope substations and event types

### 🔍 Analysis Provided
- **Exception Detection**
- **Root Cause + Actions** (GenAI)
- **Event Trends Over Time**
- **Repetitive Faults Detection**
- **Load Pattern Clustering**
- **Executive Summary & Manager Insights** (GenAI)
- **Forecast Future Issues & Inventory Suggestions**

### 🛠️ Technologies Used
- Streamlit, OpenAI GPT-3.5, pandas, matplotlib, sklearn, dotenv
""")


# event_exception_analysis_app.py (Final: Overview + Load Event Tabs with Subtabs)


# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))



with main_tabs[0]:
    st.subheader("📌 Application Overview")
    st.markdown("""
### 🎯 Purpose
This AI-powered tool assists grid operators in identifying and analyzing exceptional grid events using intelligent automation. It flags outages, overloads, and anomalies, and provides root cause analysis and actionable insights using GenAI.

### 📥 Inputs
- Grid event log files (`.csv` or `.xlsx`)
- Fields: timestamp, substation, event type, fault code, load (MW)
- Filters to scope substations and event types

### 🔍 Analysis Provided
- **Exception Detection**
- **Root Cause + Actions** (GenAI)
- **Event Trends Over Time**
- **Repetitive Faults Detection**
- **Load Pattern Clustering**
- **Executive Summary & Manager Insights** (GenAI)

### 🛠️ Technologies Used
- Streamlit, OpenAI GPT-3.5, pandas, matplotlib, sklearn, dotenv

### 🌟 Benefits
- Real-time insights, reduced fault repetition, data-driven strategy

### 🚀 Production Readiness
- SCADA integration, GenAI logging, security, Docker/CI/CD
    """)

with main_tabs[1]:
    uploaded_file = st.file_uploader("📁 Upload your grid event log (.csv or .xlsx)", type=["csv", "xlsx"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)

        substations = sorted(df["substation"].dropna().unique().tolist())
        event_types = sorted(df["event_type"].dropna().unique().tolist())

        selected_substations = st.multiselect("🏭 Focus on Substations", ["All"] + substations, default=["All"])
        selected_event_types = st.multiselect("🚨 Focus on Event Types", ["All"] + event_types, default=["All"])

        if "All" not in selected_substations:
            df = df[df["substation"].isin(selected_substations)]
        if "All" not in selected_event_types:
            df = df[df["event_type"].isin(selected_event_types)]

        def is_exception(row):
            return (
                row.get("event_type") in ["Outage", "Overload", "RelayTrip"] or
                row.get("fault_code") not in ["None", "", None] or
                row.get("load_MW", 0) > 80
            )

        @st.cache_data(show_spinner=False)
        def analyze_event_cached(row_dict):
            prompt = (
                f"You are a grid reliability assistant. Analyze the following grid event:\n\n"
                f"Timestamp: {row_dict.get('timestamp')}\n"
                f"Substation: {row_dict.get('substation')}\n"
                f"Event Type: {row_dict.get('event_type')}\n"
                f"Fault Code: {row_dict.get('fault_code')}\n"
                f"Load (MW): {row_dict.get('load_MW')}\n\n"
                f"Give a short root cause analysis and advice to the operator."
            )
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=150,
                    temperature=0.4,
                )
                reply = response.choices[0].message.content
                if "Advice:" in reply:
                    parts = reply.split("Advice:")
                    return parts[0].strip(), parts[1].strip()
                return reply, "N/A"
            except OpenAIError as e:
                return "Error generating response", str(e)

        results = []
        max_rows = 20
        analyze_all = st.checkbox("🔄 Analyze all exceptions (may be slow)", value=False)
        max_rows = None if analyze_all else 20
        analyzed_rows = 0

        if not df.empty:
            with st.spinner("Analyzing exceptions row by row..."):
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
                "🧾 Exceptions", "📊 Trends", "🔁 Repetitions", "🧠 Clustering", "📌 Summary", "📋 Insights", "🔮 Forecast Future Issues"
            ])

            with sub_tabs[0]:
                st.subheader("🧾 Detected Exceptions")
                def highlight_critical(row):
                    if row.get("load_MW", 0) > 90 or row.get("fault_code") not in ["None", "", None]:
                        return ["background-color: #ffcccc"] * len(row)
                    return [""] * len(row)

                st.dataframe(result_df.style.apply(highlight_critical, axis=1))
                st.download_button("📥 Download CSV Report", result_df.to_csv(index=False), "grid_event_exceptions.csv", "text/csv")

            with sub_tabs[1]:
                st.subheader("📊 Trends Over Time")
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                trend_df = df[df['event_type'].isin(event_types)]
                event_trend = trend_df.groupby([pd.Grouper(key='timestamp', freq='D'), 'event_type']).size().unstack(fill_value=0)
                st.line_chart(event_trend)

            with sub_tabs[2]:
                st.subheader("🔁 Repetitive Faults and Inefficiencies")
                repeat_faults = df[df['fault_code'].notnull()].groupby(['substation', 'fault_code']).size().reset_index(name='count')
                repeat_faults = repeat_faults[repeat_faults['count'] > 1]
                st.dataframe(repeat_faults)

            with sub_tabs[3]:
                st.subheader("🧠 Pattern Clustering")
                features = df[['load_MW']].fillna(0)
                scaler = StandardScaler()
                scaled_features = scaler.fit_transform(features)
                kmeans = KMeans(n_clusters=3, random_state=42)
                df['cluster'] = kmeans.fit_predict(scaled_features)
                st.dataframe(df[['substation', 'event_type', 'load_MW', 'cluster']])

            with sub_tabs[4]:
                st.subheader("📌 Executive Summary")
                summary_prompt = (
                    f"As an AI grid analyst, review {len(result_df)} exceptions.\n"
                    f"Highlight:\n- Common event types\n- Risky substations\n- Fault patterns\n- Recommendations\n\n"
                    f"Sample Data:\n{result_df[['event_type', 'substation', 'fault_code', 'load_MW']].head(10).to_string(index=False)}"
                )
                try:
                    summary_resp = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": summary_prompt}],
                        max_tokens=200,
                        temperature=0.4,
                    )
                    st.info(summary_resp.choices[0].message.content)
                except Exception as e:
                    st.warning(f"Could not generate summary: {e}")

            with sub_tabs[5]:
                st.subheader("📋 Manager's Insights")
                insight_prompt = (
                    f"As a grid operations strategist, analyze the following {len(result_df)} exception events.\n"
                    f"Give recommendations for grid reliability and operational efficiency.\n\n"
                    f"Data:\n{result_df[['substation', 'event_type', 'fault_code', 'load_MW']].head(10).to_string(index=False)}"
                )
                try:
                    insight_resp = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": insight_prompt}],
                        max_tokens=300,
                        temperature=0.4,
                    )
                    st.success(insight_resp.choices[0].message.content)
                except Exception as e:
                    st.warning(f"Could not generate insights: {e}")

            with sub_tabs[6]:
                st.subheader("🔮 Forecast Future Issues")
                forecast_prompt = f"""You are a predictive reliability analyst. Forecast operational risks for the next 30 days.

Tasks:
1. Identify substations likely to face repeat faults or overloads
2. Detect event patterns that signal emerging risks
3. Suggest preventive actions
4. Recommend inventory to stock (e.g., relays, breakers, fuses) with quantities by substation

Recent Events:
{df[['timestamp','substation','event_type','fault_code','load_MW']].tail(25).to_string(index=False)}
"""
                try:
                    forecast = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You forecast grid issues and recommend parts inventory."},
                            {"role": "user", "content": forecast_prompt}
                        ],
                        max_tokens=500,
                        temperature=0.4,
                    )
                    st.markdown(forecast.choices[0].message.content)
                except Exception as e:
                    st.error(f"❌ Forecast error: {e}")
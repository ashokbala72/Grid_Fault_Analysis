import pandas as pd
import os
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI, OpenAIError
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(layout="wide")
st.title("⚡ AI-Powered Grid Event Exception Analyzer")

# Upload interface
uploaded_file = st.file_uploader("📁 Upload your grid event log (.csv or .xlsx)", type=["csv", "xlsx"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)

    # Filter selection with 'All' tab effect
    substations = sorted(df["substation"].dropna().unique().tolist())
    event_types = sorted(df["event_type"].dropna().unique().tolist())

    selected_substations = st.multiselect("🏭 Focus on Substations", ["All"] + substations, default=["All"])
    selected_event_types = st.multiselect("🚨 Focus on Event Types", ["All"] + event_types, default=["All"])

    # Apply filters
    if "All" not in selected_substations:
        df = df[df["substation"].isin(selected_substations)]
    if "All" not in selected_event_types:
        df = df[df["event_type"].isin(selected_event_types)]

    # Exception criteria
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

    # Precompute exceptions and results before tab creation
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
    else:
        result_df = pd.DataFrame()

    if max_rows is not None and analyzed_rows >= max_rows:
        st.warning(f"Showing results for first {max_rows} exception rows only for faster performance.")

    tab_labels = ["🧾 Exception Analysis", "📊 Trends", "🔁 Inefficiencies", "🧠 Clustering", "📌 Summary", "📋 Manager's Insights"]
    tabs = st.tabs([f"{label} {'🔍' if 'All' not in selected_substations or 'All' not in selected_event_types else ''}" for label in tab_labels])

    with tabs[0]:
        st.subheader("🧾 Detected Exceptions")
        if not result_df.empty:
            def highlight_critical(row):
                if row.get("load_MW", 0) > 90 or row.get("fault_code") not in ["None", "", None]:
                    return ["background-color: #ffcccc"] * len(row)
                return [""] * len(row)

            st.dataframe(result_df.style.apply(highlight_critical, axis=1))

            st.download_button(
                label="📥 Download CSV Report",
                data=result_df.to_csv(index=False),
                file_name="grid_event_exceptions.csv",
                mime="text/csv"
            )
        else:
            st.info("✅ No exceptions found based on the selected filters.")

    with tabs[1]:
        st.subheader("📊 Trends Over Time")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        trend_df = df[df['event_type'].isin(event_types)]
        event_trend = trend_df.groupby([pd.Grouper(key='timestamp', freq='D'), 'event_type']).size().unstack(fill_value=0)
        st.line_chart(event_trend)

    with tabs[2]:
        st.subheader("🔁 Repetitive Faults and Inefficiencies")
        repeat_faults = df[df['fault_code'].notnull()].groupby(['substation', 'fault_code']).size().reset_index(name='count')
        repeat_faults = repeat_faults[repeat_faults['count'] > 1]
        st.dataframe(repeat_faults)

    with tabs[3]:
        st.subheader("🧠 Pattern Clustering")
        features = df[['load_MW']].fillna(0)
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        kmeans = KMeans(n_clusters=3, random_state=42)
        df['cluster'] = kmeans.fit_predict(scaled_features)
        st.dataframe(df[['substation', 'event_type', 'load_MW', 'cluster']])

    with tabs[4]:
        st.subheader("📌 Executive Summary")
        if not result_df.empty:
            summary_prompt = (
                f"As an AI grid analyst, review {len(result_df)} exceptions.\n"
                f"Highlight:\n"
                f"- Most common event types and substations\n"
                f"- Repetitive faults\n"
                f"- Load-related risks\n"
                f"- Suggestions to reduce recurring issues\n\n"
                f"Sample Data:\n{result_df[['event_type', 'substation', 'fault_code', 'load_MW']].head(10).to_string(index=False)}"
            )
            try:
                summary_resp = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": summary_prompt}],
                    max_tokens=200,
                    temperature=0.4,
                )
                summary = summary_resp.choices[0].message.content
                st.info(summary)
            except Exception as e:
                st.warning(f"Could not generate summary: {e}")

    with tabs[5]:
        st.subheader("📋 Manager's Insights")
        if not result_df.empty:
            insight_prompt = (
                f"As a grid operations strategist, analyze the following {len(result_df)} exception events.\n"
                f"Provide strategic and managerial recommendations based on the key findings.\n"
                f"Focus on root causes, systemic issues, and high-level guidance that can improve grid reliability.\n\n"
                f"Data:\n{result_df[['substation', 'event_type', 'fault_code', 'load_MW']].head(10).to_string(index=False)}"
            )
            try:
                insight_resp = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": insight_prompt}],
                    max_tokens=300,
                    temperature=0.4,
                )
                insight = insight_resp.choices[0].message.content
                st.success(insight)
            except Exception as e:
                st.warning(f"Could not generate insights: {e}")

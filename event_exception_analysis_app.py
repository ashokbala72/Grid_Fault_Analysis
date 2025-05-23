import pandas as pd
import os
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI, OpenAIError

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(layout="wide")
st.title("⚡ AI-Powered Grid Event Exception Analyzer")

# Upload interface
uploaded_file = st.file_uploader("Upload your grid event log (.csv or .xlsx)", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Read uploaded file
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)

    # Allow filtering
    substations = df["substation"].dropna().unique().tolist()
    event_types = df["event_type"].dropna().unique().tolist()

    selected_substations = st.multiselect("Select Substations (optional)", substations, default=substations)
    selected_events = st.multiselect("Select Event Types (optional)", event_types, default=event_types)

    df = df[df["substation"].isin(selected_substations) & df["event_type"].isin(selected_events)]

    # Exception criteria
    def is_exception(row):
        return (
            row.get("event_type") in ["Outage", "Overload", "RelayTrip"] or
            row.get("fault_code") not in ["None", "", None] or
            row.get("load_MW", 0) > 80
        )

    # GenAI analysis
    def analyze_event(row):
        prompt = (
            f"You are a grid reliability assistant. Analyze the following grid event:\n\n"
            f"Timestamp: {row.get('timestamp')}\n"
            f"Substation: {row.get('substation')}\n"
            f"Event Type: {row.get('event_type')}\n"
            f"Fault Code: {row.get('fault_code')}\n"
            f"Load (MW): {row.get('load_MW')}\n\n"
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

    # Analyze and stream results
    results = []
    container = st.container()

    with st.spinner("Analyzing exceptions row by row..."):
        for _, row in df.iterrows():
            if is_exception(row):
                cause, action = analyze_event(row)
                record = row.to_dict()
                record["Root Cause"] = cause
                record["Suggested Action"] = action
                results.append(record)
                container.write(pd.DataFrame([record]))

    # Highlight critical exceptions
    st.subheader("📋 All Exceptions Detected")
    if results:
        result_df = pd.DataFrame(results)

        def highlight_critical(row):
            if row.get("load_MW", 0) > 90 or row.get("fault_code") not in ["None", "", None]:
                return ["background-color: #ffcccc"] * len(row)
            return [""] * len(row)

        st.dataframe(result_df.style.apply(highlight_critical, axis=1))

        # Executive Summary
        summary_prompt = (
            f"As an AI grid analyst, summarize these {len(result_df)} exception events.\n"
            f"Identify key patterns, critical issues, and operational suggestions.\n\n"
            f"Data:\n{result_df[['event_type', 'fault_code', 'load_MW']].head(10).to_string(index=False)}"
        )
        try:
            summary_resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": summary_prompt}],
                max_tokens=200,
                temperature=0.4,
            )
            summary = summary_resp.choices[0].message.content
            st.markdown("### 🧠 Executive Summary")
            st.info(summary)
        except Exception as e:
            st.warning(f"Could not generate summary: {e}")

        # Download button
        st.download_button(
            label="📥 Download CSV Report",
            data=result_df.to_csv(index=False),
            file_name="grid_event_exceptions.csv",
            mime="text/csv"
        )
    else:
        st.info("✅ No exceptions found based on the selected filters.")

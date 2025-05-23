# event_analysis_app.py
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import pandas as pd
import os
from openai import OpenAI

st.set_page_config(page_title="⚡ Grid Fault Analysis Assistant")
st.title("⚠️ Grid Fault Analysis with GenAI")

# Step 1: Upload the event log CSV
uploaded_file = st.file_uploader("Upload Grid Event Log (CSV)", type="csv")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head(20))
        st.session_state["event_df"] = df
    except Exception as e:
        st.error(f"Failed to load CSV: {e}")
        st.stop()

# Step 2: Let user pick a row (event) for analysis
if "event_df" in st.session_state:
    df = st.session_state["event_df"]
    selected_row = st.selectbox("Select an event to analyze", df.index)
    row = df.loc[selected_row]

    context = f"""
Event Details:
Timestamp: {row['timestamp']}
Substation: {row['substation']}
Event Type: {row['event_type']}
Fault Code: {row['fault_code']}
Load (MW): {row['load_MW']}
"""

    st.code(context.strip(), language="markdown")

    # Step 3: Generate Root Cause + Recommendation
    if st.button("🔍 Analyze with GenAI"):
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                st.error("OPENAI_API_KEY not set. Use .env or Streamlit secrets.")
                st.stop()

            client = OpenAI(api_key=api_key)
            prompt = f"""
You are a senior grid operations engineer. Given the following event log entry, identify the likely root cause and recommend immediate mitigation steps.

{context}

Respond in this format:
- Root Cause:
- Suggested Next Steps:
"""

            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a power grid fault analysis expert."},
                    {"role": "user", "content": prompt}
                ]
            )
            result = response.choices[0].message.content
            st.markdown("### 🧠 GenAI Suggestion:")
            st.text_area("Root Cause & Action Plan", result, height=250)

        except Exception as e:
            st.error(f"GenAI analysis failed: {e}")
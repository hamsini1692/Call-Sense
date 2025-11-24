from dotenv import load_dotenv
import os

load_dotenv()  # <-- MUST be FIRST THING before reading API key


import os
import pathlib

import streamlit as st
import pandas as pd
from openai import OpenAI

from graph.supervisor import run_pipeline, GLOBAL_MEMORY
from typing import Optional

# ------------- Setup ------------- #

# Load environment variables (expects OPENAI_API_KEY in .env or env)
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY is None:
    raise RuntimeError(
        "OPENAI_API_KEY not found. Please add it to your .env or environment."
    )

client = OpenAI(api_key=OPENAI_API_KEY)


# ------------- Data loading helpers ------------- #

@st.cache_data
def load_calls_csv() -> Optional[pd.DataFrame]:
    """
    Try to load a CSV of calls from the data/ folder.

    You can adjust the path or filename if needed.
    """
    possible_paths = [
        "data/callsense_calls.csv",
        "data/calls.csv",
        "data/calls_master.csv",
    ]
    for p in possible_paths:
        if pathlib.Path(p).exists():
            df = pd.read_csv(p)
            return df
    return None


def get_transcript_from_row(row: pd.Series) -> str:
    """
    Try to extract transcript text from a row.

    Adjust the column names here based on your actual CSV.
    """
    for col in ["transcript", "text", "call_text", "utterance"]:
        if col in row and isinstance(row[col], str):
            return row[col]
    # fallback: first string-like column
    for val in row.values:
        if isinstance(val, str):
            return val
    return ""


# ------------- Streamlit UI ------------- #

st.set_page_config(
    page_title="CallSense – Multi-Agent Call Analyzer",
    layout="wide",
)

st.title("📞 CallSense – Multi-Agent Call Analysis")

st.markdown(
    """
This app runs your **LangGraph-style multi-agent pipeline** on a call transcript.

Agents in the pipeline:
- Cleaning → Entities → Summarization → Sentiment  
- Frustration Loop → Pain Points → Action Recommendations → Evaluation  
"""
)

df = load_calls_csv()

with st.sidebar:
    st.header("Call Input")

    input_mode = st.radio(
        "Choose input mode:",
        ["Select from CSV", "Paste transcript manually"],
        index=0 if df is not None else 1,
    )

    selected_transcript = ""

    if input_mode == "Select from CSV":
        if df is None:
            st.warning(
                "No CSV found in data/. "
                "Place a file like `data/callsense_calls.csv` and restart the app."
            )
        else:
            st.success(f"Loaded {len(df)} calls from CSV.")
            st.write("Sample of loaded data:")
            st.dataframe(df.head())

            row_idx = st.number_input(
                "Select row index", min_value=0, max_value=len(df) - 1, value=0
            )
            row = df.iloc[int(row_idx)]
            selected_transcript = get_transcript_from_row(row)
    else:
        selected_transcript = st.text_area(
            "Paste a call transcript here",
            height=200,
            placeholder="Paste raw call transcript...",
        )

    run_button = st.button("🔍 Analyze Call")

# ----------- Main pipeline run ----------- #

if run_button:
    if not selected_transcript.strip():
        st.error("Please provide a transcript (via CSV or text area) before running.")
    else:
        with st.spinner("Running multi-agent pipeline..."):
            call_state = run_pipeline(
                raw_transcript=selected_transcript,
                llm_client=client,
                cleaner=None,
                data_loader=None,
                update_memory=True,
            )

        st.success("Analysis complete ✅")

        # Layout: left = transcript + summary, right = insights
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("📄 Raw Transcript")
            st.write(selected_transcript)

            st.subheader("📝 Summary")
            st.write(call_state.summary or "_No summary generated._")

            st.subheader("🙂 Sentiment")
            st.write(call_state.sentiment or "_Unknown_")

        with col_right:
            st.subheader("🧩 Extracted Entities / Context")
            st.json(call_state.entities or {})

            st.subheader("📉 Frustration Timeline")
            if call_state.frustration_timeline:
                st.json(call_state.frustration_timeline)
            else:
                st.write("_No frustration events detected._")

            st.subheader("🔥 Pain Points")
            if call_state.pain_points:
                for p in call_state.pain_points:
                    st.markdown(f"- {p}")
            else:
                st.write("_No pain points extracted._")

            st.subheader("✅ Recommended Actions")
            if call_state.recommended_actions:
                for a in call_state.recommended_actions:
                    st.markdown(f"- {a}")
            else:
                st.write("_No actions generated._")

        # Show evaluation + observability
        st.markdown("---")
        st.subheader("📊 Evaluation & Observability")

        st.write("**Per-call evaluation (call_state.evaluation):**")
        st.json(call_state.evaluation or {})

        st.write("**Basic runtime stats:**")
        st.write(
            {
                "step_count": call_state.step_count,
                "tool_calls": call_state.tool_calls,
                "tool_successes": call_state.tool_successes,
            }
        )

        # Show memory snapshot
        st.markdown("### 🧠 Memory Snapshot (Global Trends)")
        st.write(
            {
                "total_calls": GLOBAL_MEMORY.total_calls,
                "sentiment_counts": GLOBAL_MEMORY.sentiment_counts,
                "top_pain_points": GLOBAL_MEMORY.pain_point_counts,
                "product_issue_counts": GLOBAL_MEMORY.product_issue_counts,
                "avg_faithfulness": GLOBAL_MEMORY.avg_faithfulness,
                "avg_coverage": GLOBAL_MEMORY.avg_coverage,
                "avg_consistency": GLOBAL_MEMORY.avg_consistency,
            }
        )
else:
    st.info("Select or paste a transcript, then click **Analyze Call** to run the pipeline.")

#!/usr/bin/env python3


import streamlit as st
import requests
import pandas as pd
import io

API_BASE_URL = "http://localhost:8084"
SCORE_ENDPOINT = f"{API_BASE_URL}/score_headlines"
STATUS_ENDPOINT = f"{API_BASE_URL}/status"

st.set_page_config(
    page_title="Headline Sentiment Scorer",
    page_icon="H",
    layout="wide",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Header */
    .app-header {
        border-bottom: 2px solid #000;
        padding-bottom: 1rem;
        margin-bottom: 2rem;
    }
    .app-header h1 {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .app-header p {
        margin: 0.3rem 0 0 0;
        color: #555;
        font-size: 0.95rem;
    }

    /* Section headers */
    .section-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #888;
        margin-bottom: 0.5rem;
    }

    /* Stat boxes */
    .stat-box {
        border: 1px solid #ddd;
        border-radius: 4px;
        padding: 1rem;
        text-align: center;
    }
    .stat-box .stat-value {
        font-size: 1.8rem;
        font-weight: 700;
    }
    .stat-box .stat-label {
        font-size: 0.8rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)



def check_api_status() -> bool:
    """Return True if the FastAPI scoring service is reachable."""
    try:
        resp = requests.get(STATUS_ENDPOINT, timeout=3)
        return resp.status_code == 200
    except requests.exceptions.RequestException:
        return False


def score_headlines_api(headlines: list[str]) -> dict | None:
    """Send headlines to the API and return the JSON response."""
    try:
        resp = requests.post(
            SCORE_ENDPOINT,
            json={"headlines": headlines},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as exc:
        st.error(f"API request failed: {exc}")
        return None


def parse_uploaded_file(uploaded_file) -> list[str]:
    """Extract non-empty lines from an uploaded text file."""
    content = uploaded_file.read().decode("utf-8", errors="replace")
    return [line.strip() for line in content.splitlines() if line.strip()]


if "headlines" not in st.session_state:
    st.session_state.headlines = []
if "results" not in st.session_state:
    st.session_state.results = None


st.markdown("""
<div class="app-header">
    <h1>Headline Sentiment Scorer</h1>
    <p>Classify news headlines as Optimistic, Pessimistic, or Neutral</p>
</div>
""", unsafe_allow_html=True)


with st.sidebar:
    st.subheader("Settings")

    # API URL override
    custom_url = st.text_input(
        "API Base URL",
        value=API_BASE_URL,
        help="Change if the scoring API runs on a different host or port.",
    )
    if custom_url != API_BASE_URL:
        API_BASE_URL = custom_url
        SCORE_ENDPOINT = f"{API_BASE_URL}/score_headlines"
        STATUS_ENDPOINT = f"{API_BASE_URL}/status"

    st.markdown("---")

    # Connection check
    st.subheader("API Connection")
    if st.button("Check Connection"):
        if check_api_status():
            st.success("API is online")
        else:
            st.error("Cannot reach the API. Ensure the FastAPI server is running.")

    st.markdown("---")
    st.caption(
        "Start the API with:\n\n"
        "`python score_headlines_api.py`\n\n"
        f"Expected at: `{API_BASE_URL}`"
    )


st.markdown('<div class="section-label">Input</div>', unsafe_allow_html=True)

tab_paste, tab_manual, tab_file = st.tabs([
    "Paste Headlines",
    "Add One-by-One",
    "Upload File",
])

# Tab 1 – paste multiple headlines
with tab_paste:
    pasted_text = st.text_area(
        "One headline per line",
        height=180,
        placeholder="Paste headlines here, one per line...",
        key="paste_area",
    )
    if st.button("Add Pasted Headlines", key="btn_add_paste"):
        new_lines = [ln.strip() for ln in pasted_text.splitlines() if ln.strip()]
        if new_lines:
            st.session_state.headlines.extend(new_lines)
            st.success(f"Added {len(new_lines)} headline(s).")
            st.rerun()
        else:
            st.warning("No headlines detected.")

# Tab 2 – add one at a time
with tab_manual:
    single_headline = st.text_input(
        "Headline",
        placeholder="Type a headline...",
        key="single_input",
    )
    if st.button("Add Headline", key="btn_add_single"):
        if single_headline.strip():
            st.session_state.headlines.append(single_headline.strip())
            st.success("Headline added.")
            st.rerun()
        else:
            st.warning("Enter a headline first.")

# Tab 3 – upload a file
with tab_file:
    uploaded = st.file_uploader(
        "Choose a .txt file",
        type=["txt"],
        key="file_upload",
    )
    if uploaded is not None:
        if st.button("Add Headlines from File", key="btn_add_file"):
            file_lines = parse_uploaded_file(uploaded)
            if file_lines:
                st.session_state.headlines.extend(file_lines)
                st.success(f"Added {len(file_lines)} headline(s) from {uploaded.name}.")
                st.rerun()
            else:
                st.warning("The file appears to be empty.")


st.markdown("---")
st.markdown('<div class="section-label">Headline Queue</div>', unsafe_allow_html=True)

if not st.session_state.headlines:
    st.info("No headlines in queue. Use the input section above to add some.")
else:
    st.write(f"**{len(st.session_state.headlines)}** headline(s) ready to score.")

    # Action buttons
    col_clear, col_score = st.columns([1, 1])
    with col_clear:
        if st.button("Clear All", key="btn_clear_all"):
            st.session_state.headlines = []
            st.session_state.results = None
            st.rerun()
    with col_score:
        score_btn = st.button("Score Headlines", key="btn_score", type="primary")

    # Editable list
    st.caption("Edit inline or click X to remove.")
    indices_to_remove = []
    for idx, headline in enumerate(st.session_state.headlines):
        col_text, col_del = st.columns([11, 1])
        with col_text:
            new_val = st.text_input(
                f"#{idx + 1}",
                value=headline,
                key=f"edit_{idx}",
                label_visibility="collapsed",
            )
            if new_val != headline:
                st.session_state.headlines[idx] = new_val
        with col_del:
            if st.button("X", key=f"del_{idx}"):
                indices_to_remove.append(idx)

    # Remove marked headlines
    if indices_to_remove:
        for i in sorted(indices_to_remove, reverse=True):
            st.session_state.headlines.pop(i)
        st.rerun()

    # Score
    if score_btn:
        if not st.session_state.headlines:
            st.warning("Add at least one headline before scoring.")
        else:
            with st.spinner("Scoring headlines..."):
                result = score_headlines_api(st.session_state.headlines)
            if result and "labels" in result:
                st.session_state.results = result["labels"]
                st.rerun()
            elif result:
                st.error("Unexpected response format from the API.")


if st.session_state.results is not None:
    st.markdown("---")
    st.markdown('<div class="section-label">Results</div>', unsafe_allow_html=True)

    labels = st.session_state.results
    headlines = st.session_state.headlines
    display_count = min(len(labels), len(headlines))

    # Summary statistics
    label_series = pd.Series(labels[:display_count])
    label_counts = label_series.value_counts()

    cols = st.columns(4)
    with cols[0]:
        st.markdown(
            '<div class="stat-box">'
            f'<div class="stat-value">{display_count}</div>'
            '<div class="stat-label">Total</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    for i, sentiment in enumerate(["Optimistic", "Neutral", "Pessimistic"]):
        count = int(label_counts.get(sentiment, 0))
        with cols[i + 1]:
            st.markdown(
                '<div class="stat-box">'
                f'<div class="stat-value">{count}</div>'
                f'<div class="stat-label">{sentiment}</div>'
                '</div>',
                unsafe_allow_html=True,
            )

    st.write("")  # spacer

    # Results table
    results_df = pd.DataFrame({
        "#": range(1, display_count + 1),
        "Headline": headlines[:display_count],
        "Sentiment": labels[:display_count],
    })

    st.dataframe(
        results_df,
        use_container_width=True,
        hide_index=True,
        height=400,
    )

    # Download options
    st.markdown('<div class="section-label">Download</div>', unsafe_allow_html=True)
    dl_col1, dl_col2 = st.columns(2)

    with dl_col1:
        csv_buffer = io.StringIO()
        results_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="Download CSV",
            data=csv_buffer.getvalue(),
            file_name="headline_scores.csv",
            mime="text/csv",
        )

    with dl_col2:
        txt_lines = [
            f"{label},{headline}"
            for label, headline in zip(labels[:display_count], headlines[:display_count])
        ]
        st.download_button(
            label="Download TXT",
            data="\n".join(txt_lines),
            file_name="headline_scores.txt",
            mime="text/plain",
        )


st.markdown("---")
st.caption("Headline Sentiment Scorer | Assignment 3 | ML Engineering")

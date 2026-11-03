import streamlit as st
import streamlit.components.v1 as components
import subprocess
import sys
from pathlib import Path

# Project directory
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DASHBOARD_FILE = BASE_DIR / "dashboard.html"
ANALYSIS_FILE = BASE_DIR / "analyze.py"

st.set_page_config(
    page_title="Yield Advisor",
    page_icon="??",
    layout="wide"
)

st.title("?? Yield Advisor")
st.caption("AI-powered Semiconductor Yield Analysis and Optimization")

st.sidebar.title("Yield Advisor")
st.sidebar.markdown("### Navigation")
st.sidebar.info(
    "Explore yield monitoring, wafer patterns, "
    "root-cause analysis, probe performance and recommendations."
)

if st.sidebar.button("?? Refresh Analysis"):
    st.cache_data.clear()
    st.rerun()


@st.cache_data(show_spinner=False)
def load_dashboard():
    result = subprocess.run(
        [sys.executable, str(ANALYSIS_FILE)],
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
        check=True
    )

    if not DASHBOARD_FILE.exists():
        raise FileNotFoundError("dashboard.html was not generated.")

    return DASHBOARD_FILE.read_text(encoding="utf-8"), result.stdout


if not DATA_DIR.exists():
    st.error("The data folder was not found.")
    st.stop()

try:
    with st.spinner("Analyzing semiconductor yield data... Please wait."):
        html, logs = load_dashboard()

    st.success("Yield analysis completed successfully!")

    with st.expander("Analysis execution details"):
        st.text(logs)

    components.html(
        html,
        height=1600,
        scrolling=True
    )

except subprocess.CalledProcessError as error:
    st.error("An error occurred while running the yield analysis.")
    st.code(error.stderr or str(error))

except Exception as error:
    st.error(f"Unable to load Yield Advisor: {error}")

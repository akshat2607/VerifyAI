import os

import requests
import streamlit as st

st.set_page_config(page_title="VerifyAI", page_icon="🔍")
st.title("🔍 VerifyAI")
st.caption(
    "Paste a message, news item or social media post. VerifyAI finds the "
    "factual claims, searches the web, and shows its evidence."
)


# ---------- Backend address ----------
def get_backend_url():
    try:
        if "BACKEND_URL" in st.secrets:
            return st.secrets["BACKEND_URL"].rstrip("/")
    except Exception:
        pass
    return os.environ.get("BACKEND_URL", "").rstrip("/")


BACKEND_URL = get_backend_url()

with st.sidebar:
    st.header("Settings")
    if not BACKEND_URL:
        BACKEND_URL = st.text_input("Backend URL", placeholder="https://your-app.onrender.com").rstrip("/")
    st.info(
        "VerifyAI shows evidence and a confidence level. "
        "It can make mistakes, so always check the sources."
    )
    if BACKEND_URL and st.button("Show recent checks"):
        try:
            h = requests.get(f"{BACKEND_URL}/history", timeout=60).json().get("history", [])
            if not h:
                st.write("No checks yet.")
            for row in h:
                st.markdown(f"**{row['time']}**  \n{row['text']}  \n{', '.join(row['verdicts'])}")
        except Exception as e:
            st.error(f"Could not load history: {e}")

icons = {
    "Supported": "✅",
    "Contradicted": "❌",
    "Mixed": "⚠️",
    "Not enough evidence": "❓",
    "Error": "🚫",
}

user_text = st.text_area(
    "Paste the text you want to verify",
    height=160,
    placeholder="Example: Drinking hot water every hour cures COVID-19.",
)

if st.button("Verify", type="primary"):
    if not BACKEND_URL:
        st.error("Please add the backend URL in the sidebar (top-left arrow).")
        st.stop()
    if not user_text.strip():
        st.warning("Please paste some text first.")
        st.stop()

    try:
        with st.spinner("Checking claims... (the first request can take up to a minute while the server wakes up)"):
            response = requests.post(
                f"{BACKEND_URL}/verify",
                json={"text": user_text},
                timeout=180,
            )

        if response.status_code != 200:
            detail = response.json().get("detail", response.text)
            st.error(f"Backend error: {detail}")
            st.stop()

        results = response.json().get("results", [])
        if not results:
            st.info("No checkable factual claims were found in this text.")
            st.stop()

        for r in results:
            verdict = r.get("verdict", "Not enough evidence")
            confidence = r.get("confidence", 0)

            st.divider()
            st.subheader(f"Claim: {r.get('claim', '')}")
            st.markdown(f"### {icons.get(verdict, '❓')} {verdict}")
            st.progress(min(max(int(confidence), 0), 100) / 100, text=f"Confidence: {confidence}%")
            st.write(r.get("explanation", ""))

            st.markdown("**Sources checked:**")
            for i, s in enumerate(r.get("sources", []), start=1):
                mark = " ⭐ used in verdict" if s.get("used") else ""
                st.markdown(f"{i}. [{s.get('title', 'Source')}]({s.get('url', '')}){mark}")

    except Exception as e:
        st.error(f"Could not reach the backend: {e}")

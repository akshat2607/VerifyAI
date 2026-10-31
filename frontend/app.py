import html
import os
import uuid
from collections import Counter
from urllib.parse import urlparse

import requests
import streamlit as st

st.set_page_config(page_title="VerifyAI", page_icon="🔍", layout="centered")

st.markdown(
    """<style>
.hero{text-align:center;padding:1.2rem 0 .6rem}
.hero h1{font-size:2.8rem;font-weight:800;margin:0;background:linear-gradient(90deg,#14b8a6,#6366f1);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.hero p{opacity:.75;margin-top:.4rem}
.card{border-radius:14px;padding:1rem 1.2rem;margin:1rem 0 .4rem;background:rgba(128,128,128,.08);border-left:6px solid var(--c)}
.claim{font-size:1.05rem;font-weight:600;margin-bottom:.6rem}
.badge{display:inline-block;padding:.2rem .75rem;border-radius:999px;color:#fff;font-weight:700;font-size:.85rem;background:var(--c)}
.conf{float:right;font-weight:700;color:var(--c)}
.bar{height:10px;border-radius:999px;background:rgba(128,128,128,.25);overflow:hidden;margin:.7rem 0 .2rem}
.bar>div{height:100%;border-radius:999px;background:var(--c);animation:grow 1s ease-out}
@keyframes grow{from{width:0}}
.expl{margin-top:.6rem;line-height:1.55}
.pill{display:inline-block;padding:.25rem .8rem;margin:0 .4rem .4rem 0;border-radius:999px;border:1px solid var(--c);color:var(--c);font-weight:600;font-size:.9rem}
.hist{border-radius:10px;padding:.5rem .7rem;margin-bottom:.5rem;background:rgba(128,128,128,.1);font-size:.85rem}
.hist small{opacity:.7}
</style>""",
    unsafe_allow_html=True,
)

ICONS = {"Supported": "✅", "Contradicted": "❌", "Mixed": "⚠️", "Not enough evidence": "❓", "Error": "🚫"}
COLORS = {
    "Supported": "#16a34a",
    "Contradicted": "#dc2626",
    "Mixed": "#d97706",
    "Not enough evidence": "#6b7280",
    "Error": "#6b7280",
}
EXAMPLES = [
    "Drinking hot water every hour cures COVID-19.",
    "The Great Wall of China is visible from space with the naked eye.",
    "Mount Everest is the tallest mountain on Earth, and it is in India.",
]
LANGS = ["Original", "English", "Hindi", "Punjabi", "Spanish", "French", "German", "Arabic", "Chinese", "Japanese"]
MAX_CHARS = 3000

for key, default in [
    ("user_text", ""),
    ("results", None),
    ("history", None),
    ("history_error", None),
    ("confirm_clear", False),
    ("translated", None),
    ("lang", "Original"),
    ("tr_error", None),
    ("backend_url", ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# A random ID for this visitor's session. It is sent to the backend so each
# visitor only sees and clears their own checks.
if "client_id" not in st.session_state:
    st.session_state.client_id = str(uuid.uuid4())


def esc(x):
    return html.escape(str(x))


def get_backend_url():
    try:
        if "BACKEND_URL" in st.secrets:
            return st.secrets["BACKEND_URL"].rstrip("/")
    except Exception:
        pass
    return os.environ.get("BACKEND_URL", "").rstrip("/")


def set_text(t):
    st.session_state.user_text = t


def clear_all():
    st.session_state.user_text = ""
    st.session_state.results = None
    st.session_state.translated = None
    st.session_state.lang = "Original"
    st.session_state.tr_error = None


def clear_history():
    """Clears the history shown in the sidebar and asks the backend to delete it."""
    st.session_state.history = []
    st.session_state.history_error = None
    st.session_state.confirm_clear = False  # require confirmation again next time
    backend = st.session_state.backend_url
    if backend:
        try:
            r = requests.delete(
                f"{backend}/history",
                params={"client_id": st.session_state.client_id},
                timeout=60,
            )
            if r.status_code not in (200, 204):
                try:
                    detail = r.json().get("detail", r.text)
                except Exception:
                    detail = r.text
                st.session_state.history_error = (
                    f"Backend did not clear history ({r.status_code}): {str(detail)[:200] or 'no details'}"
                )
        except Exception as e:
            st.session_state.history_error = f"Could not clear backend history: {e}"


def do_translate():
    """Runs when the language dropdown changes."""
    lang = st.session_state.lang
    results = st.session_state.results
    st.session_state.tr_error = None
    if lang == "Original" or not results:
        st.session_state.translated = None
        return
    texts = [r.get("claim", "") for r in results] + [r.get("explanation", "") for r in results]
    try:
        tr = requests.post(
            f"{st.session_state.backend_url}/translate",
            json={"texts": texts, "language": lang},
            timeout=120,
        )
        if tr.status_code == 200:
            out = tr.json()["translations"]
            k = len(results)
            st.session_state.translated = {"lang": lang, "claims": out[:k], "expl": out[k:]}
        else:
            st.session_state.translated = None
            try:
                detail = tr.json().get("detail", tr.text)
            except Exception:
                detail = tr.text
            st.session_state.tr_error = f"Translation failed ({tr.status_code}): {detail}"
    except Exception as e:
        st.session_state.translated = None
        st.session_state.tr_error = f"Translation failed: {e}"


def safe_conf(v):
    try:
        return min(max(int(v), 0), 100)
    except Exception:
        return 0


BACKEND_URL = get_backend_url()

# ---------- Sidebar ----------
with st.sidebar:
    st.header("🔍 VerifyAI")
    if not BACKEND_URL:
        BACKEND_URL = st.text_input("Backend URL", placeholder="https://your-app.onrender.com").rstrip("/")

    with st.expander("How it works", expanded=True):
        st.markdown(
            "1. 🧠 AI extracts checkable claims\n"
            "2. 🌐 Web search finds evidence\n"
            "3. ⚖️ AI judges using only that evidence\n"
            "4. 📚 You see the verdict and sources"
        )
    st.info("It can make mistakes. Always check the sources.")

    # Keep the URL in session state before any callbacks run
    st.session_state.backend_url = BACKEND_URL

    if BACKEND_URL:
        col_show, col_clear = st.columns(2)
        if col_show.button("🕘 Show", use_container_width=True):
            try:
                data = requests.get(
                    f"{BACKEND_URL}/history",
                    params={"client_id": st.session_state.client_id},
                    timeout=60,
                ).json()
                st.session_state.history = data.get("history", [])
                st.session_state.history_error = None
            except Exception as e:
                st.error(f"Could not load history: {e}")

        col_clear.button(
            "🗑️ Clear",
            on_click=clear_history,
            use_container_width=True,
            disabled=not st.session_state.confirm_clear,
            help="Delete all your saved past checks",
        )
        st.checkbox("Yes, permanently delete all my past checks", key="confirm_clear")

    if st.session_state.history_error:
        st.error(st.session_state.history_error)

    if st.session_state.history is not None:
        if not st.session_state.history:
            st.caption("No checks yet.")
        for row in st.session_state.history:
            v = " ".join(ICONS.get(x, "❓") for x in row.get("verdicts", []))
            st.markdown(
                f"<div class='hist'><small>{esc(row.get('time',''))}</small><br>"
                f"{esc(row.get('text','')[:90])}<br>{v}</div>",
                unsafe_allow_html=True,
            )

# ---------- Hero ----------
st.markdown(
    "<div class='hero'><h1>🔍 VerifyAI</h1>"
    "<p>Paste a message, news item or social media post. VerifyAI finds the factual claims, "
    "searches the web, and shows its evidence.</p></div>",
    unsafe_allow_html=True,
)

# ---------- Input ----------
st.markdown("**Try an example:**")
cols = st.columns(len(EXAMPLES))
for col, ex in zip(cols, EXAMPLES):
    col.button(ex[:38] + "…", key=f"ex_{ex[:10]}", on_click=set_text, args=(ex,), use_container_width=True)

user_text = st.text_area(
    "Paste the text you want to verify",
    key="user_text",
    height=150,
    placeholder="Example: Drinking hot water every hour cures COVID-19.",
)
n = len(user_text)
st.caption(f"{n} / {MAX_CHARS} characters" + (" ⚠️ too long" if n > MAX_CHARS else ""))

b1, b2 = st.columns([4, 1])
verify_clicked = b1.button(
    "🔎 Verify claims",
    type="primary",
    use_container_width=True,
    disabled=(not user_text.strip() or n > MAX_CHARS),
)
b2.button("Clear", on_click=clear_all, use_container_width=True)

# ---------- Run ----------
if verify_clicked:
    if not BACKEND_URL:
        st.error("Please add the backend URL in the sidebar (top-left arrow).")
        st.stop()
    try:
        with st.status("Checking claims… the first request can take up to a minute while the server wakes up.", expanded=True) as status:
            st.write("🧠 Extracting claims, 🌐 searching the web, ⚖️ judging the evidence…")
            response = requests.post(
                f"{BACKEND_URL}/verify",
                json={"text": user_text, "client_id": st.session_state.client_id},
                timeout=180,
            )
            try:
                body = response.json()
            except Exception:
                body = None
            if response.status_code != 200 or not isinstance(body, dict):
                status.update(label="Something went wrong", state="error")
                detail = body.get("detail") if isinstance(body, dict) else None
                st.error(f"Backend error ({response.status_code}): {detail or response.text[:300] or 'empty response'}")
                st.stop()
            st.session_state.results = body.get("results") or []
            st.session_state.translated = None
            st.session_state.lang = "Original"
            st.session_state.tr_error = None
            status.update(label="Done!", state="complete", expanded=False)
    except Exception as e:
        st.error(f"Could not reach the backend: {e}")
        st.stop()

# ---------- Results ----------
results = st.session_state.results
if results is not None:
    if not results:
        st.info("No checkable factual claims were found in this text.")
    else:
        st.markdown("### Results")
        counts = Counter(r.get("verdict", "Not enough evidence") for r in results)
        pills = "".join(
            f"<span class='pill' style='--c:{COLORS.get(v, '#6b7280')}'>{ICONS.get(v, '❓')} {c} {esc(v)}</span>"
            for v, c in counts.items()
        )
        st.markdown(pills, unsafe_allow_html=True)

        st.selectbox("🌐 Translate results to", LANGS, key="lang", on_change=do_translate)
        if st.session_state.tr_error:
            st.error(st.session_state.tr_error)
        tdata = st.session_state.translated
        if tdata:
            st.caption(f"Showing translation in {tdata['lang']}")

        for idx, r in enumerate(results):
            verdict = r.get("verdict", "Not enough evidence")
            conf = safe_conf(r.get("confidence", 0))
            color = COLORS.get(verdict, "#6b7280")
            claim_txt = tdata["claims"][idx] if tdata else r.get("claim", "")
            expl_txt = tdata["expl"][idx] if tdata else r.get("explanation", "")
            st.markdown(
                f"<div class='card' style='--c:{color}'>"
                f"<div class='claim'>{esc(claim_txt)}</div>"
                f"<span class='badge'>{ICONS.get(verdict, '❓')} {esc(verdict)}</span>"
                f"<span class='conf'>{conf}% confidence</span>"
                f"<div class='bar'><div style='width:{conf}%'></div></div>"
                f"<div class='expl'>{esc(expl_txt)}</div></div>",
                unsafe_allow_html=True,
            )
            sources = r.get("sources", [])
            used = sum(1 for s in sources if s.get("used"))
            if sources:
                with st.expander(f"📚 Sources ({len(sources)}) · {used} used in verdict"):
                    for s in sources:
                        title = str(s.get("title", "Source")).replace("[", "(").replace("]", ")")
                        url = s.get("url", "")
                        domain = urlparse(url).netloc.replace("www.", "")
                        mark = "⭐ **used**" if s.get("used") else "▫️"
                        st.markdown(f"{mark} [{title}]({url}) · `{domain}`")

        report = "VerifyAI report\n\n" + "\n\n".join(
            f"Claim: {r.get('claim','')}\nVerdict: {r.get('verdict','')} ({safe_conf(r.get('confidence',0))}%)\n"
            f"Why: {r.get('explanation','')}\nSources: "
            + ", ".join(s.get("url", "") for s in r.get("sources", []) if s.get("used"))
            for r in results
        )
        st.download_button("⬇️ Download report", report, file_name="verifyai_report.txt", use_container_width=True)

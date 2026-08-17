import os
import base64
import io
import time
from datetime import datetime
from openai import OpenAI, RateLimitError, APIError
import streamlit as st

try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

SAMBANOVA_API_KEY = st.secrets.get("SAMBANOVA_API_KEY")
if not SAMBANOVA_API_KEY:
    st.error("SAMBANOVA_API_KEY not found in st.secrets. Add it to .streamlit/secrets.toml (locally) or your app's Secrets settings (on Streamlit Cloud).")
    st.stop()

client = OpenAI(api_key=SAMBANOVA_API_KEY, base_url="https://api.sambanova.ai/v1")


def get_stream_with_fallback(messages, primary_model, is_vision=False):
    """
    Try the requested model first. If SambaNova returns a 429 (high demand),
    move down the fallback chain until one model accepts the request.
    Returns (stream, model_actually_used).
    Raises the last error if every candidate in the chain is rate-limited.
    """
    chain = VISION_FALLBACK_CHAIN if is_vision else TEXT_FALLBACK_CHAIN
    candidates = [primary_model] + [m for m in chain if m != primary_model]

    last_error = None
    for model in candidates:
        try:
            stream = client.chat.completions.create(
                messages=messages,
                model=model,
                stream=True,
                stream_options={"include_usage": True},
            )
            return stream, model
        except RateLimitError as e:
            last_error = e
            continue
        except APIError as e:
            # Non-rate-limit API errors (bad request, etc.) shouldn't be
            # silently retried against a different model — surface them.
            if getattr(e, "status_code", None) == 429:
                last_error = e
                continue
            raise

    raise last_error

TEXT_MODEL   = "Meta-Llama-3.3-70B-Instruct"
VISION_MODEL = "Llama-4-Maverick-17B-128E-Instruct"

AVAILABLE_MODELS = {
    "Meta-Llama-3.3-70B-Instruct":         "LLaMA 3.3 70B (Default)",
    "Meta-Llama-3.1-8B-Instruct":          "LLaMA 3.1 8B (Fast)",
    "Meta-Llama-3.1-405B-Instruct":        "LLaMA 3.1 405B",
    "Llama-4-Scout-17B-16E-Instruct":      "LLaMA 4 Scout (Vision)",
    "Llama-4-Maverick-17B-128E-Instruct":  "LLaMA 4 Maverick (Vision)",
}

# Order matters: tried left-to-right until one responds without a 429.
TEXT_FALLBACK_CHAIN = [
    "Meta-Llama-3.3-70B-Instruct",
    "Meta-Llama-3.1-8B-Instruct",
    "Meta-Llama-3.1-405B-Instruct",
]
VISION_FALLBACK_CHAIN = [
    "Llama-4-Maverick-17B-128E-Instruct",
    "Llama-4-Scout-17B-16E-Instruct",
]

DEFAULT_SYSTEM_PROMPT = (
    "You are an advanced AI assistant (LLM). "
    "Your Founder is 'Swagato Bhattacharya'. "
    "Founder's Email: swagatobhattacharya576@gmail.com. "
    "Be helpful, concise, and clear. Format code in markdown code blocks."
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="SambaNova AI Chatbot", page_icon="🤖", layout="wide")

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0d1a 0%, #1a1a2e 100%);
        border-right: 1px solid #2d2d4e;
    }
    .user-bubble {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border-radius: 18px 18px 4px 18px;
        padding: 12px 18px;
        margin: 6px 0 6px 15%;
        box-shadow: 0 4px 15px rgba(102,126,234,0.3);
        font-size: 15px;
        line-height: 1.6;
    }
    .bot-bubble {
        background: linear-gradient(135deg, #1e2140, #252a4a);
        color: #e8e8f0;
        border-radius: 18px 18px 18px 4px;
        padding: 12px 18px;
        margin: 6px 15% 6px 0;
        border: 1px solid #3d3d6b;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        font-size: 15px;
        line-height: 1.6;
    }
    .role-label {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 1px;
        opacity: 0.65;
        margin-bottom: 4px;
        text-transform: uppercase;
    }
    .token-badge {
        font-size: 10px;
        color: #7878a0;
        text-align: right;
        margin-top: 4px;
        padding-right: 4px;
    }
    .context-badge {
        background: linear-gradient(135deg, #f6d365, #fda085);
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 12px;
        font-weight: 600;
        color: #1a1a2e;
        display: inline-block;
        margin: 2px;
    }
    .stats-bar {
        background: #1e2140;
        border: 1px solid #3d3d6b;
        border-radius: 10px;
        padding: 8px 14px;
        font-size: 12px;
        color: #9898c0;
        margin-bottom: 10px;
    }
    [data-testid="stSidebar"] * { color: #c8c8e8 !important; }
    [data-testid="stSidebar"] .stTextArea textarea {
        background: #1e2140 !important;
        border: 1px solid #3d3d6b !important;
        color: #e8e8f0 !important;
        border-radius: 8px;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
    }
    .stButton>button:hover { opacity: 0.85 !important; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  SESSION STATE BOOTSTRAP
# ════════════════════════════════════════════════════════════
def new_session_id():
    return f"session_{int(time.time()*1000)}"

if "sessions" not in st.session_state:
    sid = new_session_id()
    st.session_state.sessions = {
        sid: {
            "name":         "Chat 1",
            "history":      [],
            "total_tokens": 0,
            "created":      datetime.now().strftime("%H:%M"),
        }
    }
    st.session_state.active_session = sid

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT

if "selected_model" not in st.session_state:
    st.session_state.selected_model = TEXT_MODEL


def active():
    return st.session_state.sessions[st.session_state.active_session]

def all_sessions():
    return st.session_state.sessions


# ════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712039.png", width=80)
    st.title("SambaNova AI Chatbot 🤖")
    st.markdown("---")

    # ── Sessions ─────────────────────────────────────────────
    st.markdown("### 💬 Chat Sessions")

    if st.button("➕ New Chat", use_container_width=True):
        sid = new_session_id()
        n   = len(st.session_state.sessions) + 1
        st.session_state.sessions[sid] = {
            "name":         f"Chat {n}",
            "history":      [],
            "total_tokens": 0,
            "created":      datetime.now().strftime("%H:%M"),
        }
        st.session_state.active_session = sid
        st.rerun()

    for sid, sess in list(all_sessions().items()):
        is_active = sid == st.session_state.active_session
        label     = f"{'▶ ' if is_active else ''}{sess['name']}  ({sess['created']})"
        cols      = st.columns([5, 1])
        with cols[0]:
            if st.button(label, key=f"sel_{sid}", use_container_width=True):
                st.session_state.active_session = sid
                st.rerun()
        with cols[1]:
            if len(all_sessions()) > 1:
                if st.button("🗑", key=f"del_{sid}"):
                    del st.session_state.sessions[sid]
                    st.session_state.active_session = list(st.session_state.sessions.keys())[-1]
                    st.rerun()

    new_name = st.text_input("Rename current chat:", value=active()["name"], key="rename_input")
    if new_name and new_name != active()["name"]:
        active()["name"] = new_name

    st.markdown("---")

    # ── Model selector ────────────────────────────────────────
    st.markdown("### 🧠 Model")
    st.session_state.selected_model = st.selectbox(
        "Choose model:",
        options=list(AVAILABLE_MODELS.keys()),
        format_func=lambda k: AVAILABLE_MODELS[k],
        index=list(AVAILABLE_MODELS.keys()).index(st.session_state.selected_model),
    )

    st.markdown("---")

    # ── System Prompt ─────────────────────────────────────────
    st.markdown("### ⚙️ System Prompt")
    st.session_state.system_prompt = st.text_area(
        "Customize bot personality:",
        value=st.session_state.system_prompt,
        height=130,
        key="sys_prompt_input",
    )
    if st.button("↺ Reset to Default", use_container_width=True):
        st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT
        st.rerun()

    st.markdown("---")

    # ── Memory toggle ─────────────────────────────────────────
    st.markdown("### 🧠 Chat Memory")
    memory_enabled = st.toggle("Enable Chat Memory", value=True)

    st.markdown("---")

    # ── File uploads ──────────────────────────────────────────
    st.markdown("### 📎 Upload Files")
    st.markdown("**📄 PDF Document**")
    pdf_file = st.file_uploader("Upload a PDF", type=["pdf"], key="pdf_uploader")

    st.markdown("**🖼️ Image**")
    image_file = st.file_uploader(
        "Upload an image (JPG / PNG / WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
        key="image_uploader",
    )

    if st.button("🗑️ Clear Uploaded Files", use_container_width=True):
        for k in ["pdf_text", "pdf_name", "image_b64", "image_mime", "image_name"]:
            st.session_state.pop(k, None)
        st.rerun()

# ── Process PDF ───────────────────────────────────────────────────────────────
if pdf_file is not None:
    if not PDF_SUPPORT:
        st.sidebar.error("PyPDF2 not installed. Run: pip install PyPDF2")
    elif st.session_state.get("pdf_name") != pdf_file.name:
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
        st.session_state["pdf_text"] = "\n\n".join(
            p.extract_text() or "" for p in reader.pages
        )
        st.session_state["pdf_name"] = pdf_file.name

# ── Process Image ─────────────────────────────────────────────────────────────
if image_file is not None:
    if st.session_state.get("image_name") != image_file.name:
        raw = image_file.read()
        st.session_state["image_b64"]  = base64.b64encode(raw).decode()
        st.session_state["image_mime"] = image_file.type
        st.session_state["image_name"] = image_file.name
    with st.sidebar:
        st.image(
            f"data:{st.session_state['image_mime']};base64,{st.session_state['image_b64']}",
            caption=st.session_state["image_name"],
            use_column_width=True,
        )


# ════════════════════════════════════════════════════════════
#  MAIN CHAT AREA
# ════════════════════════════════════════════════════════════
has_pdf   = "pdf_text"  in st.session_state
has_image = "image_b64" in st.session_state
history   = active()["history"]

st.title(f"💬 {active()['name']}")
st.caption(
    f"Model: **{AVAILABLE_MODELS[st.session_state.selected_model]}**  |  "
    f"Memory: {'✅ On' if memory_enabled else '❌ Off'}"
)

# Context badges
if has_pdf or has_image:
    if has_pdf:
        st.markdown(f"<span class='context-badge'>📄 {st.session_state.get('pdf_name','PDF')}</span>", unsafe_allow_html=True)
    if has_image:
        st.markdown(f"<span class='context-badge'>🖼️ {st.session_state.get('image_name','Image')}</span>", unsafe_allow_html=True)

# Stats bar
sess = active()
if sess["total_tokens"] > 0:
    st.markdown(
        f"<div class='stats-bar'>🔢 Session tokens: <b>{sess['total_tokens']:,}</b> &nbsp;|&nbsp; "
        f"💬 Messages: <b>{len(history)}</b></div>",
        unsafe_allow_html=True,
    )

# Download chat history
if history:
    chat_text = "\n\n".join(
        f"User: {m['content']}" if m["role"] == "user" and isinstance(m["content"], str)
        else (f"Assistant: {m['content']}" if m["role"] == "assistant" else "")
        for m in history
    ).strip()
    st.download_button(
        "💾 Download Chat",
        data=chat_text,
        file_name=f"{active()['name']}.txt",
        mime="text/plain",
    )

# ── Render history ────────────────────────────────────────────────────────────
for msg in history:
    role    = msg["role"]
    content = msg["content"]
    tokens  = msg.get("tokens")

    if isinstance(content, list):
        display = " ".join(p["text"] for p in content if isinstance(p, dict) and p.get("type") == "text")
    else:
        display = content

    if role == "user":
        st.markdown(
            f"<div class='user-bubble'><div class='role-label'>You</div>{display}</div>",
            unsafe_allow_html=True,
        )
    else:
        token_line = f"<div class='token-badge'>🔢 {tokens:,} tokens</div>" if tokens else ""
        st.markdown(
            f"<div class='bot-bubble'><div class='role-label'>Assistant</div>{display}{token_line}</div>",
            unsafe_allow_html=True,
        )

# ── Chat input ────────────────────────────────────────────────────────────────
user_input = st.chat_input(
    "Ask me anything…" if not (has_pdf or has_image) else "Ask about your PDF or image…"
)

# ════════════════════════════════════════════════════════════
#  HANDLE SUBMISSION WITH STREAMING
# ════════════════════════════════════════════════════════════
if user_input:
    chosen_model = st.session_state.selected_model
    sys_prompt   = st.session_state.system_prompt

    # ── Build user content ────────────────────────────────────
    if has_image:
        user_content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{st.session_state['image_mime']};base64,{st.session_state['image_b64']}"
                },
            },
            {"type": "text", "text": user_input},
        ]
        if has_pdf:
            user_content.append({
                "type": "text",
                "text": f"\n\n[PDF context]:\n{st.session_state['pdf_text'][:6000]}",
            })
        chosen_model = VISION_MODEL

    elif has_pdf:
        pdf_ctx      = f"\n\n[PDF — '{st.session_state['pdf_name']}']:\n{st.session_state['pdf_text'][:8000]}"
        user_content = user_input + pdf_ctx

    else:
        user_content = user_input

    # ── Build messages list ───────────────────────────────────
    base_messages = [{"role": "system", "content": sys_prompt}]
    if memory_enabled:
        for m in history:
            base_messages.append({"role": m["role"], "content": m["content"]})
    base_messages.append({"role": "user", "content": user_content})

    # ── Show user bubble immediately ──────────────────────────
    st.markdown(
        f"<div class='user-bubble'><div class='role-label'>You</div>{user_input}</div>",
        unsafe_allow_html=True,
    )

    # ── Stream response ───────────────────────────────────────
    reply_placeholder = st.empty()
    full_reply        = ""
    prompt_tokens     = 0
    completion_tokens = 0

    with st.spinner(""):
        # SambaNova's OpenAI-compatible API returns usage in a final chunk
        # when stream_options={"include_usage": True} is set.
        # get_stream_with_fallback retries with alternate models on a 429.
        try:
            stream, model_used = get_stream_with_fallback(
                base_messages, chosen_model, is_vision=has_image
            )
        except RateLimitError:
            st.error(
                "🚦 All available models are currently experiencing high demand "
                "on SambaNova's side. Please wait a moment and try again."
            )
            st.stop()

        if model_used != chosen_model:
            st.info(
                f"⚠️ **{AVAILABLE_MODELS.get(chosen_model, chosen_model)}** is "
                f"currently overloaded — this reply was generated with "
                f"**{AVAILABLE_MODELS.get(model_used, model_used)}** instead."
            )

        try:
            for chunk in stream:
                # Accumulate streamed text
                if chunk.choices and chunk.choices[0].delta.content:
                    full_reply += chunk.choices[0].delta.content
                    reply_placeholder.markdown(
                        f"<div class='bot-bubble'><div class='role-label'>Assistant</div>{full_reply}▌</div>",
                        unsafe_allow_html=True,
                    )

                # Final chunk carries usage stats (choices is empty on this chunk)
                if getattr(chunk, "usage", None):
                    usage             = chunk.usage
                    prompt_tokens     = getattr(usage, "prompt_tokens",     0) or 0
                    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        except RateLimitError:
            st.warning(
                "🚦 Hit a rate limit mid-response. Partial reply shown below — "
                "try sending your message again."
            )

    total_msg_tokens = prompt_tokens + completion_tokens

    # Final render without cursor, with token info if available
    token_line = (
        f"<div class='token-badge'>🔢 {total_msg_tokens:,} tokens "
        f"(prompt: {prompt_tokens:,} + completion: {completion_tokens:,})</div>"
        if total_msg_tokens > 0 else ""
    )
    reply_placeholder.markdown(
        f"<div class='bot-bubble'><div class='role-label'>Assistant</div>{full_reply}{token_line}</div>",
        unsafe_allow_html=True,
    )

    # ── Persist ───────────────────────────────────────────────
    active()["history"].append({"role": "user",      "content": user_content})
    active()["history"].append({"role": "assistant",  "content": full_reply, "tokens": total_msg_tokens or None})
    active()["total_tokens"] += total_msg_tokens

    st.rerun()

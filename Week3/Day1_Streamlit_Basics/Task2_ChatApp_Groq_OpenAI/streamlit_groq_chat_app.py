import os
import streamlit as st
from groq import Groq
from typing import List, Dict
from dotenv import load_dotenv   


load_dotenv()

st.set_page_config(page_title="Streaming Chat Box...!", layout="wide")


def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
    if "client" not in st.session_state:
        st.session_state.client = None


def get_client():
    if st.session_state.client is None:
        api_key = os.getenv("GROQ_API_KEY")  # ✅ Read from .env
        if not api_key:
            st.error("⚠ GROQ_API_KEY not set in .env file")
            return None
        st.session_state.client = Groq(api_key=api_key)
    return st.session_state.client


def append_message(role: str, content: str):
    st.session_state.messages.append({"role": role, "content": content})



init_session()

with st.sidebar:
    st.header("⚙ Groq settings")
    model = st.selectbox(
        "Model",
        options=["llama-3.3-70b-versatile", "llama3-8b-8192", "mixtral-8x7b"],
        index=0,
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2)
    max_tokens = st.number_input("Max tokens", min_value=64, max_value=8192, value=1024, step=64)
    system_prompt = st.text_area(
        "System prompt (optional)",
        value=st.session_state.messages[0]["content"],
        height=100,
    )

    if st.button("Update system prompt"):
        st.session_state.messages[0]["content"] = system_prompt
        st.rerun()

st.title("🗣 Let's Chat — (streaming)")

with st.sidebar:
    st.markdown("### Controls")
    if st.button("Clear chat"):
        st.session_state.messages = [{"role": "system", "content": system_prompt}]
        st.rerun()



for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])



user_input = st.chat_input("Type your message...")

if user_input:
    client = get_client()
    if client is None:
        st.stop()

    append_message("user", user_input)
    with st.chat_message("user"):
        st.markdown(user_input)

    messages_for_api: List[Dict] = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]

    with st.chat_message("assistant"):
        assistant_placeholder = st.empty()
        assistant_text = ""

        try:
            stream = client.chat.completions.create(
                model=model,
                messages=messages_for_api,
                temperature=float(temperature),
                max_tokens=int(max_tokens),
                stream=True,
            )

            for chunk in stream:
                delta = getattr(chunk.choices[0], "delta", None)
                if delta and getattr(delta, "content", None):
                    assistant_text += delta.content
                    assistant_placeholder.markdown(assistant_text)

            append_message("assistant", assistant_text)

        except Exception as e:
            st.error(f"❌ Error while calling Groq API: {e}")


st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit and Groq SDK")

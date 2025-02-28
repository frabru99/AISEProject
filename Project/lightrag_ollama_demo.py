import asyncio
import os
import inspect
import time
import logging
import streamlit as st
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc
from ollama import Client

WORKING_DIR = "./"

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=ollama_model_complete,
    llm_model_name="llama3.1:8b",
    llm_model_max_async=4,
    llm_model_max_token_size=32768,
    llm_model_kwargs={"host": "http://localhost:11434", "options": {"num_ctx": 32768}},
    embedding_func=EmbeddingFunc(
        embedding_dim=768,
        max_token_size=8192,
        func=lambda texts: ollama_embed(
            texts, embed_model="nomic-embed-text", host="http://localhost:11434"
        ),
    ),
)

#inseri
directory = os.fsencode("./input")

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    print("NOME FILE: "+ filename)
    with open("./input" + "/" + filename, "r", encoding="utf-8") as f:
        rag.insert(f.read())

#Interfaccia WEB
st.markdown(
    """
    <script>
    function scrollToBottom() {
        window.scrollTo(0, document.body.scrollHeight);
    }
    </script>
    """,
    unsafe_allow_html=True
)

st.markdown("""
    <style>
        .button-container {
            position: relative;
            top: 50px;
            left: 10px;
            right: 20px;
        }
    </style>
    <div class="button-container">
""", unsafe_allow_html=True)

if st.button('Show Graph'):
    exec(open("graph_visual_with_html.py").read())

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>Privacy AI</h1>", unsafe_allow_html=True)


if "agent" not in st.session_state:
    st.session_state.agent = rag
    st.session_state.messages = []

if "messages" in st.session_state:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

scroll_placeholder = st.empty()

prompt = st.chat_input("Ask a question...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        response_text = ""

        # Simula una risposta in streaming (sostituire con chiamata reale)
        for word in st.session_state.agent.query(prompt).split():
            response_text += word + " "
            response_placeholder.markdown(response_text)
            time.sleep(0.05)  # Simula il tempo di risposta
        
        # Salva la risposta completa nella sessione
        st.session_state.messages.append({"role": "assistant", "content": response_text})

    # Forza lo scroll in basso
    scroll_placeholder.markdown('<script>scrollToBottom();</script>', unsafe_allow_html=True)
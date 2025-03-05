import asyncio
import os
import inspect
import time
import logging
import streamlit as st
import re
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

st.markdown("<h3 style='text-align: center;'>Welcome!</h3>", unsafe_allow_html=True)


if "agent" not in st.session_state:
    st.session_state.agent = rag
    st.session_state.messages = []

if "messages" in st.session_state:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

scroll_placeholder = st.empty()

prompt = st.chat_input("Ask a question...")

#col1, col2 = st.columns([4,2])
tab1, tab2 = st.tabs(["Privacy AI","DeepSeek Evaluation"])

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with tab1:
        with st.chat_message("user"):
            st.markdown(prompt)

    with tab1:
        with st.chat_message("assistant"):
            # Simula una risposta in streaming
            with tab1:
                response_placeholder = st.empty()
                response_text = "" 
                for word in st.session_state.agent.query(prompt,system_prompt="You are a Reasoner, an agent specialized in analyzing and solving complex questions through problem decomposition, concept explanation, and structured reasoning. When you receive a question: provide a clear and complete explanation of the relevant concepts, break down the problem into simple sub-questions and answer each one, use the answers to the sub-questions to build a coherent and well-justified reasoning process leading to the final answer, your response must be clear, well-structured, and supported by step-by-step justification.").split():
                    response_text += word + " "
                    response_placeholder.markdown(response_text)
                    time.sleep(0.05)  # Simula il tempo di risposta

        #mandiamo la risposta di llama a deepseek per valutarne la qualità 
        client = Client(
            host='http://localhost:11434/',
            headers={'Content-Type': 'application/json'}
        )
        response = client.chat(model='deepseek-r1:8b', messages=[
            {
                'role': 'system',
                'content': "You are an Evaluator, an agent specialized in assessing the correctness and completeness of the answers provided by a Reasoner. Your task is to: evaluate whether the concept explanations are clear and accurate, check if the problem decomposition into sub-questions is appropriate and whether the answers to those questions are correct, verify if the final reasoning is logical, complete, and leads to the correct answer, perform a counterfactual evaluation, examining alternative scenarios to see if the reasoning and answer remain robust, your assessment must be precise, well-justified, and based on a rigorous analysis.",
            },
            {
                'role': 'user',
                'content': f"Adesso tu devi valutare la risposta di un altro LLM: dato che il task era {prompt} e la risposta generata è stata '{response_text}', la risposta fornita è corretta? Rispondi solo con 'corretto' o 'sbagliato', esprimendoti esclusivamente in italiano.",
            },
        ])

        #print(response['message']['content'])

        with tab2:
            with st.chat_message("assistant"):     
                response_placeholder = st.empty()
                deepseek_text = ""
                for word in response['message']['content'].split():
                    deepseek_text += word + " "
                    response_placeholder.markdown(deepseek_text)
                    time.sleep(0.05)  # Simula il tempo di risposta
        
    
    #QUI AVVIENE IL FILTRAGGIO DELLA RISPOSTA DEL REASONER
    
    # Testo con la parte di thinking
    deepseek_response = deepseek_text
    # Rimuove tutto ciò che è tra <think> e </think>
    clean_response = re.sub(r"<think>.*?</think>", "", deepseek_response, flags=re.DOTALL).strip()

    # Definizione delle regex per "corretto" e "sbagliato"
    yes_pattern = re.compile(r"\b(corretto|certamente|ovviamente|assolutamente s[iì]|senz'altro)\b", re.IGNORECASE)
    no_pattern = re.compile(r"\b(sbagliato|negativo|assolutamente no|non credo|non penso)\b", re.IGNORECASE)

    # Verifica se la risposta contiene "sì" o "no": se sì, allora non c'è bisogno del rethinking, altrimenti sì
    if yes_pattern.search(clean_response):
        st.toast('La risposta è corretta', icon="✅")
    elif no_pattern.search(clean_response):
        st.toast('La risposta è sbagliata', icon="🚨")
        with tab1:
            st.markdown("**RISPOSTA RIFORMULATA**") 
            response_placeholder = st.empty()
            response_text = ""
            for word in st.session_state.agent.query(f"La seguente risposta {response_text} è sbagliata. Rispondi adeguatamente a questa domanda: {prompt}").split():
                response_text += word + " "
                response_placeholder.markdown(response_text)
                time.sleep(0.05)  # Simula il tempo di risposta
    else:
        st.toast('Hey, noccapito.', icon="⚠️")

    with tab1:
        # Salva la risposta completa nella sessione
        st.session_state.messages.append({"role": "assistant", "content": response_text})

    # Forza lo scroll in basso
    scroll_placeholder.markdown('<script>scrollToBottom();</script>', unsafe_allow_html=True)
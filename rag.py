import base64
import os
import streamlit as st
from pypdf import PdfReader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage

load_dotenv(override=True)

prompt_template = """
Answer the following question based only on the provided context:
<context>
    {context}
</context>
<question>
    {input}
</question>
"""


def image_to_data_url(file):
    data = file.read()
    file.seek(0)
    ext = "png"
    if hasattr(file, "name") and file.name:
        lower = file.name.lower()
        if lower.endswith(".jpg") or lower.endswith(".jpeg"):
            ext = "jpeg"
        elif lower.endswith(".png"):
            ext = "png"
        elif lower.endswith(".webp"):
            ext = "webp"
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:image/{ext};base64,{b64}"


def ensure_openai_key():
    # 1. Check Streamlit secrets
    if "openai" in st.secrets and "api_key" in st.secrets["openai"]:
        os.environ["OPENAI_API_KEY"] = st.secrets["openai"]["api_key"]
        return st.secrets["openai"]["api_key"]
    
    # 2. Check environment variables (including .env via load_dotenv)
    key = os.getenv("OPENAI_API_KEY", "")
    if key:
        return key
    
    st.error("OPENAI_API_KEY manquant. Ajoutez-la dans .streamlit/secrets.toml, dans un fichier .env ou via la barre latérale.")
    return None


def build_retriever_from_pdfs(pdf_files, embed_model_name: str = "text-embedding-3-small"):
    content = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            content += text
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=512, chunk_overlap=16)
    chunks = splitter.split_text(content)
    embedding_model = OpenAIEmbeddings(model=embed_model_name)
    try:
        vector_store = Chroma.from_texts(chunks, embedding_model, collection_name="data_collection")
    except Exception as e:
        st.error(f"Échec des embeddings: {e}")
        st.stop()
    return vector_store.as_retriever(search_kwargs={"k": 5})

def test_api_key(embed_model_name: str):
    try:
        OpenAIEmbeddings(model=embed_model_name).embed_query("ping")
        st.success("Clé OpenAI valide.")
    except Exception as e:
        st.error(f"Clé invalide ou accès refusé: {e}")


def main():
    st.set_page_config(page_title="RAG", layout="wide")
    st.subheader("Retrieval Augmented Generation", divider="blue")

    tabs = st.tabs(["RAG PDF", "RAG Multimodal"])

    with tabs[0]:
        with st.sidebar:
            st.sidebar.title("Données")
            try:
                st.image("rag.png")
            except Exception:
                pass
            # Vérification de la clé au démarrage de la sidebar
            key_from_config = ensure_openai_key()
            
            if key_from_config:
                 st.success(f"✅ Clé API chargée (secrets.toml / .env) [Termine par: {key_from_config[-4:]}]")
                 with st.expander("Modifier la clé API"):
                    api_key = st.text_input("Nouvelle OpenAI API Key", type="password", placeholder="sk-...")
                    if api_key:
                        os.environ["OPENAI_API_KEY"] = api_key.strip()
                        st.session_state["api_key_set"] = True
            else:
                api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
                if api_key:
                    os.environ["OPENAI_API_KEY"] = api_key.strip()
                    st.session_state["api_key_set"] = True
            embed_choice = st.selectbox("Modèle d'embeddings", ["text-embedding-3-small", "text-embedding-3-large"], index=0)
            if st.button("Vérifier la clé"):
                if not ensure_openai_key():
                    st.stop()
                test_api_key(embed_model_name=embed_choice)
            pdf_docs = st.file_uploader(label="Chargez des PDF", type=["pdf"], accept_multiple_files=True)
            if st.button("Indexer les PDF"):
                if not pdf_docs:
                    st.warning("Ajoutez au moins un PDF.")
                else:
                    if not ensure_openai_key():
                        st.stop()
                    with st.spinner("Indexation en cours..."):
                        retriever = build_retriever_from_pdfs(pdf_docs, embed_model_name=embed_choice)
                        st.session_state.retriever = retriever
                        st.success("Indexation terminée.")

        st.subheader("Chatbot")
        question = st.text_input("Posez votre question")
        if question:
            if "retriever" not in st.session_state:
                st.warning("Aucun index n'est disponible. Indexez d'abord des PDF.")
            else:
                if not ensure_openai_key():
                    st.stop()
                llm = ChatOpenAI(model="gpt-4o", temperature=0)
                docs = st.session_state.retriever.invoke(question)
                context_list = [d.page_content for d in docs]
                context_text = ". ".join(context_list)
                prompt = prompt_template.format(context=context_text, input=question)
                resp = llm.invoke(prompt)
                st.write(resp.content)

    with tabs[1]:
        st.subheader("Multimodal (Image + Texte)")
        image = st.file_uploader("Chargez une image", type=["png", "jpg", "jpeg", "webp"])
        q = st.text_input("Votre question sur l'image")
        if st.button("Analyser l'image") and image and q:
            if not ensure_openai_key():
                st.stop()
            llm = ChatOpenAI(model="gpt-4o", temperature=0)
            data_url = image_to_data_url(image)
            message = HumanMessage(content=[{"type": "text", "text": q}, {"type": "image_url", "image_url": {"url": data_url}}])
            resp = llm.invoke([message])
            st.write(resp.content)


if __name__ == "__main__":
    main()

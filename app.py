import streamlit as st
from PyPDF2 import PdfReader
import os
import io
import time
import hashlib
import re

# Imports de LangChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

# 1. Configuracion de la pagina web
st.set_page_config(page_title="Asistente Documental Corporativo", layout="centered")
st.title("Asistente de Documentos Corporativos (PoC)")
st.write("Sube un PDF y haz preguntas sobre su contenido. Desarrollado con LangChain.")

# Pide la API Key en la barra lateral
api_key = st.sidebar.text_input("Introduce tu Google Gemini API Key:", type="password")

if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

    # 2. Subida del archivo PDF
    pdf_docs = st.file_uploader("Sube tu archivo PDF aqui", accept_multiple_files=False, type=['pdf'])

    if pdf_docs is not None:
        pdf_bytes = pdf_docs.getvalue()
        pdf_hash = hashlib.md5(pdf_bytes).hexdigest()

        # Detectar si es un PDF nuevo para no re-embedir innecesariamente
        pdf_name = pdf_docs.name
        if "processed_pdf" not in st.session_state or st.session_state.processed_pdf != pdf_hash:
            st.session_state.question_count = 0

            with st.spinner("Procesando el PDF... (solo se hace una vez por archivo)"):
                # Intentar cargar el indice en disco (cache) para ahorrar cuota
                cache_dir = os.path.join("vector_cache", pdf_hash)
                
                embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

                if os.path.isdir(cache_dir):
                    vector_store = FAISS.load_local(
                        cache_dir, embeddings, allow_dangerous_deserialization=True
                    )
                else:
                    # Extraer texto del PDF
                    pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
                    raw_text = ""
                    for page in pdf_reader.pages:
                        text = page.extract_text()
                        if text:
                            raw_text += str(text)

                    # 3. Dividir el texto en trozos
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
                    chunks = text_splitter.split_text(raw_text)

                    # 4. Crear los Embeddings y guardarlos en FAISS (solo una vez)
                    vector_store = FAISS.from_texts(chunks, embedding=embeddings)

                    os.makedirs(cache_dir, exist_ok=True)
                    vector_store.save_local(cache_dir)

                # Guardar en session_state para no repetir el proceso
                st.session_state.vector_store = vector_store
                st.session_state.processed_pdf = pdf_hash

            st.success(f"PDF '{pdf_name}' procesado correctamente. Ya puedes hacer preguntas.")

        # Recuperar el vector store cacheado
        retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 2})

        # 5. Interfaz para hacer la pregunta (max 3 por PDF)
        question_count = st.session_state.get("question_count", 0)
        remaining = max(0, 3 - question_count)
        st.caption(f"Preguntas restantes para este PDF: {remaining}")
        user_question = st.text_input(
            "Que quieres saber sobre este documento?",
            disabled=question_count >= 3
        )

        if user_question and question_count < 3:
            def extract_retry_seconds(err_text):
                match = re.search(r"retry in ([0-9]+(?:\.[0-9]+)?)", err_text, re.IGNORECASE)
                if match:
                    return float(match.group(1))
                return None

            with st.spinner("Buscando respuesta..."):
                # 6. Configurar el LLM
                # gemini-2.0-flash-lite ya no esta disponible para nuevos usuarios
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

                # El Prompt Engineering
                prompt_template = """
                Responde a la pregunta de la forma mas detallada posible basandote UNICAMENTE en el contexto proporcionado.
                Si la respuesta no esta en el contexto, di simplemente "No puedo encontrar esta informacion en el documento proporcionado", no te inventes la respuesta.

                Contexto:
                {context}

                Pregunta:
                {question}

                Respuesta:
                """
                prompt = PromptTemplate.from_template(prompt_template)

                # Funcion auxiliar para unir los textos encontrados
                def format_docs(docs):
                    return "\n\n".join(doc.page_content for doc in docs)

                # 7. Construccion de la cadena con LCEL
                rag_chain = (
                    RunnableParallel(context=retriever | format_docs, question=RunnablePassthrough())
                    | prompt
                    | llm
                    | StrOutputParser()
                )

                # Ejecutar la cadena con la pregunta del usuario
                max_retries = 2
                attempt = 0
                while True:
                    try:
                        response = rag_chain.invoke(user_question)
                        break
                    except Exception as e:
                        err = str(e)
                        if "RESOURCE_EXHAUSTED" in err or "429" in err:
                            wait_s = extract_retry_seconds(err)
                            if wait_s is not None and attempt < max_retries:
                                st.info(f"Limite temporal alcanzado. Reintentando en {wait_s:.1f} segundos...")
                                time.sleep(wait_s)
                                attempt += 1
                                continue
                            st.error(
                                "Se ha agotado la cuota de la API gratuita (error 429). "
                                "Prueba mas tarde o habilita facturacion/paga un plan. "
                                "Si el error indica un tiempo de espera, reintenta pasado ese tiempo."
                            )
                        else:
                            st.error(f"Error al consultar el modelo: {err}")
                        st.stop()

            # Mostrar el resultado
            st.success("Respuesta:")
            st.write(response)
            st.session_state.question_count = question_count + 1
        elif question_count >= 3:
            st.info("Has alcanzado el limite de 3 preguntas para este PDF. Sube otro PDF para reiniciar.")
else:
    st.warning("Por favor, introduce tu API Key en el menu lateral para empezar.")

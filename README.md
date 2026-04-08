#  Asistente Documental Corporativo (PoC RAG)

Prueba de Concepto (PoC) desarrollada en Python para demostrar la integración de Inteligencia Artificial Generativa en casos de uso empresariales. 

Esta aplicación permite a los usuarios subir documentos corporativos (PDF) y realizar consultas en lenguaje natural sobre su contenido, utilizando la arquitectura RAG (Retrieval-Augmented Generation) para evitar alucinaciones y asegurar respuestas basadas en el contexto.

##  Tecnologías y Herramientas utilizadas
* **Lenguaje:** Python
* **Framework IA:** LangChain (Arquitectura moderna con LCEL - LangChain Expression Language)
* **Modelos Generativos:** Google Gemini 1.5 Flash (LLM) y Text-Embedding-004 (Embeddings)
* **Base de Datos Vectorial:** FAISS (Local)
* **Interfaz de Usuario:** Streamlit

##  Flujo de la Aplicación
1. **Extracción:** Lectura y procesamiento de archivos PDF (`PyPDF2`).
2. **Chunking:** División del texto en fragmentos semánticos (`RecursiveCharacterTextSplitter`).
3. **Vectorización:** Conversión de texto a embeddings usando la API de Google.
4. **Recuperación:** Búsqueda de similitud vectorial para encontrar el contexto exacto de la pregunta del usuario.
5. **Generación:** Construcción de un Prompt optimizado que obliga al modelo a responder única y exclusivamente basándose en los documentos proporcionados.

##  Cómo ejecutar el proyecto en local
1. Clona este repositorio.
2. Instala las dependencias: `pip install -r requirements.txt`
3. Ejecuta la aplicación: `streamlit run app.py`
4. Introduce tu API Key de Google Gemini en la barra lateral.

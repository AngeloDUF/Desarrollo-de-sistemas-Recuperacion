from dotenv import load_dotenv  # Carga variables de entorno desde un archivo .env
import os  # Interactúa con el sistema operativo
from PyPDF2 import PdfReader  # Lee archivos PDF
import streamlit as st  # Crea aplicaciones web interactivas
from langchain.text_splitter import CharacterTextSplitter  # Divide texto en caracteres
from langchain.embeddings.openai import OpenAIEmbeddings  # Genera incrustaciones de texto con OpenAI
from langchain_community.vectorstores import Chroma  # Realiza búsqueda de similitud
from langchain.chains.question_answering import load_qa_chain  # Carga cadenas de preguntas y respuestas
from langchain.llms import OpenAI  # Interactúa con modelos de lenguaje de OpenAI
from langchain.callbacks import get_openai_callback  # Obtiene realimentación de OpenAI
import langchain  # Importación general de langchain
from bs4 import BeautifulSoup  # Analiza contenido HTML
import xml.etree.ElementTree as ET  # Procesa XML
langchain.verbose = False  # Desactiva salida detallada de langchain

# Carga las variables de entorno desde el archivo .env
load_dotenv()

# Definición de función para procesar texto extraído de archivos
def process_text(text):
  # Configuración de cómo se dividirá el texto
  text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
  )

  # División del texto en trozos
  chunks = text_splitter.split_text(text)

  # Conversión de los trozos de texto en incrustaciones para análisis
  embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
  knowledge_base = Chroma.from_texts(chunks, embeddings)

  return knowledge_base

# Función principal del programa
# Asegúrate de que todas las importaciones necesarias estén al principio del archivo

def main():
    st.title("Preguntas sobre HTMLs, XMLs, PDFs y Texto Ingresado")
    # Configuración del cargador de archivos para aceptar archivos HTML, XML y PDF
    html_xml_pdf_files = st.file_uploader("Sube tus archivos HTML, XML o PDF", type=["html", "xml", "pdf"], accept_multiple_files=True)
    
    # Área para ingreso de texto manual por el usuario
    user_text = st.text_area("O ingresa tu texto aquí para hacerle preguntas", height=150)
    
    text = user_text + "\n"  # Inicia con el texto ingresado por el usuario, si lo hay

    if html_xml_pdf_files is not None and len(html_xml_pdf_files) > 0:
        for file in html_xml_pdf_files:
            if file.type == "application/pdf":
                # Procesamiento de archivos PDF
                reader = PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() + "\n\n"
            else:
                file_content = file.getvalue().decode("utf-8", errors='replace')
                # Procesamiento de archivos HTML
                if file.type == "text/html":
                    soup = BeautifulSoup(file_content, 'html.parser')
                    text += soup.get_text(separator="\n", strip=True) + "\n\n"
                # Procesamiento de archivos XML
                elif file.type == "text/xml":
                    root = ET.fromstring(file_content)
                    for elem in root.iter():
                        if elem.text:
                            text += elem.text.strip() + "\n"

    if text.strip():  # Procesa el texto si hay algo que procesar
        knowledgeBase = process_text(text)

    # Interfaz para la entrada de preguntas por parte del usuario
    query = st.text_input('Escribe tu pregunta sobre el texto ingresado o los archivos subidos...')
    cancel_button = st.button('Cancelar')  # Botón para cancelar operación

    if cancel_button:
        st.stop()  # Finaliza la ejecución

    if query and text.strip():  # Asegura que haya texto sobre el cual preguntar
        # Realiza búsqueda de similitud con la base de conocimientos
        docs = knowledgeBase.similarity_search(query)
      
        # Configuración del modelo de lenguaje y parámetros de respuesta
        model = "gpt-3.5-turbo-instruct"
        temperature = 0
        llm = OpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"), model_name=model, temperature=temperature)
      
        # Carga y ejecución de la cadena de preguntas y respuestas
        chain = load_qa_chain(llm, chain_type="stuff")
        with get_openai_callback() as cost:
            response = chain.invoke(input={"question": query, "input_documents": docs})
            st.write(response["output_text"])  # Muestra respuesta en la interfaz

if __name__== "__main__":
    main()

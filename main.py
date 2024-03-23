import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import random
import time
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from chat_models.google_gemini import GoogleGemini as GoogleGemini

def main():
    st.set_page_config(
    page_title="Chat Documents",
    page_icon="🗿",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
    })
    st.header('Tanya tentang Tugu Jogja 🗿', divider='rainbow')
    
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    
    if "llm_app" not in st.session_state:
        st.session_state["llm_app"] = GoogleGemini()
    
    if "chat_input" not in st.session_state:
        st.session_state["chat_input"] = True
    
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF/CSV Files and Click on the Submit & Process Button",
                                    accept_multiple_files=True)

        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = st.session_state["llm_app"].process_files(pdf_docs)
                    text_chunks = st.session_state["llm_app"].split_text_into_chunks(raw_text)
                    st.session_state["llm_app"].create_vector_store(text_chunks)
                st.success("Done")
                st.session_state["chat_input"] = False
            else:
                st.error("Please upload PDF files first.")

    for message in st.session_state["llm_app"].chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)
    # Accept user input
    if prompt := st.chat_input("Say something", disabled= st.session_state["chat_input"]):
        start_inference_time = time.time()  # Catat waktu awal inferensi
        with st.chat_message("user"):
            st.markdown(prompt)
        
        if pdf_docs:
            rag_chain = st.session_state["llm_app"].get_conversational_chain()
            ai_msg = rag_chain.invoke({"question": prompt, "chat_history": st.session_state["llm_app"].chat_history})

            end_inference_time = time.time()  # Catat waktu akhir inferensi
            inference_time = end_inference_time - start_inference_time  # Hitung waktu inferensi
        
            with st.chat_message("assistant"):
                response = st.write_stream(st.session_state["llm_app"].generate_response(ai_msg.content))
                st.info(f"Inference time: {inference_time:.2f} seconds.")  # Tampilkan waktu inferensi

            st.session_state["llm_app"].chat_history.extend([HumanMessage(content=prompt), ai_msg])
            
        else:
            st.error("Please upload PDF files first.")

if __name__ == "__main__":
    main()
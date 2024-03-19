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

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
        Answer the question as detailed as possible from the provided context, make sure to provide all the details\n\n
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(user_question, pdf_text):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(pdf_text)  # Use text from PDF as context

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    return response["output_text"]


def main():
    st.set_page_config("Chat PDF")
    st.title("Simple chat")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button",
                                    accept_multiple_files=True)

        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                st.success("Done")
            else:
                st.error("Please upload PDF files first.")

    # Accept user input
    if prompt := st.chat_input("Say something"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get text from uploaded PDF files if pdf_docs is not None
        if pdf_docs:
            # Get text from uploaded PDF files
            pdf_text = get_pdf_text(pdf_docs)
            # Pass user input and PDF text to user_input function
            answer = user_input(prompt, pdf_text)
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.write(answer)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            st.error("Please upload PDF files first.")

if __name__ == "__main__":
    main()
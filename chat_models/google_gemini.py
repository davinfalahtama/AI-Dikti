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
import random
import time
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory


class GoogleGemini:
    def __init__(self):
        self.chat_history = []
        self.vector_store = None
        self.model = ChatGoogleGenerativeAI(model="gemini-pro", 
                                            temperature=0.5, 
                                            convert_system_message_to_human=True,
                                            google_api_key="AIzaSyBYgcagyUPWzHFRyTZO3o8r85oZqmC25E8")

    def process_files(self, files):
        raw_text = ""
        for file in files:
            if file.name.endswith('.pdf'):
                raw_text += self.extract_text_from_pdf(file)
            elif file.name.endswith('.csv'):
                raw_text += self.extract_text_from_csv(file)
        return raw_text

    def extract_text_from_pdf(self, file):
        pdf_text = ""
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            pdf_text += page.extract_text()
        return pdf_text

    def extract_text_from_csv(self, file):
        df = pd.read_csv(file)
        return ' '.join(df['facts'])

    def split_text_into_chunks(self, text):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        return text_splitter.split_text(text)

    def create_vector_store(self, text_chunks):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key="AIzaSyBYgcagyUPWzHFRyTZO3o8r85oZqmC25E8")
        self.vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    def contextualized_question(self, input_data):
        if input_data.get("chat_history"):
            contextualized_q_chain = self.contextualize_system_prompt()
            return contextualized_q_chain
        else:
            return input_data["question"]

    def contextualize_system_prompt(self):
        system_prompt = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""
        prompt_template = ChatPromptTemplate.from_messages(
            [("system", system_prompt), MessagesPlaceholder(variable_name="chat_history"), ("human", "{question}")]
        )
        return prompt_template | self.model | StrOutputParser()

    def get_conversational_chain(self):
        prompt_template = """
            Answer the question as detailed as possible from the provided context, make sure to provide all the details\n
            If the question is about code, answer that you don't know the answer.\n
            If the user asks about your name, answer that your name is Bogu.\n
            Context:\n {context}?\n
            Question: \n{question}\n

            Answer:
        """
        retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 6})

        qa_prompt = ChatPromptTemplate.from_messages(
            [("system", prompt_template), MessagesPlaceholder(variable_name="chat_history"), ("human", "{question}")]
        )

        rag_chain = (RunnablePassthrough.assign(context=self.contextualized_question | retriever | self.format_docs)
                     | qa_prompt
                     | self.model)
        return rag_chain

    def generate_response(self, text):
        for word in text.split():
            yield word + " "
            time.sleep(0.05)
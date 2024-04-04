import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
import random
import time
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

class ChatOpenAi:
    def __init__(self,store):
        # Initialize class attributes
        self.chat_history = []  # List to store chat history
        self.vector_store = None  # Vector store for embeddings
        self.store = store
        # Initialize the AI model
        self.model = ChatOpenAI(temperature=0.5,openai_api_key="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")

    def process_files(self, files):
        """
        Process files to extract text.

        Args:
        - files: List of file objects.

        Returns:
        - raw_text: Concatenated text extracted from files.
        """
        raw_text = ""
        for file in files:
            # Extract text from PDF files
            if file.name.endswith('.pdf'):
                raw_text += self.extract_text_from_pdf(file)
            # Extract text from CSV files
            elif file.name.endswith('.csv'):
                raw_text += self.extract_text_from_csv(file)
        return raw_text

    def extract_text_from_pdf(self, file):
        """
        Extract text from a PDF file.

        Args:
        - file: PDF file object.

        Returns:
        - pdf_text: Text extracted from the PDF file.
        """
        pdf_text = ""
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            pdf_text += page.extract_text()
        return pdf_text

    def extract_text_from_csv(self, file):
        """
        Extract text from a CSV file.

        Args:
        - file: CSV file object.

        Returns:
        - text: Text extracted from the CSV file.
        """
        df = pd.read_csv(file)
        return ' '.join(df['facts'])

    def split_text_into_chunks(self, text):
        """
        Split text into smaller chunks.

        Args:
        - text: Input text to be split.

        Returns:
        - List of text chunks.
        """
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=1000)
        return text_splitter.split_text(text)

    def create_vector_store(self, text_chunks):
        """
        Create a vector store from text chunks.

        Args:
        - text_chunks: List of text chunks.

        Returns:
        - None
        """
        embeddings = OpenAIEmbeddings(openai_api_key="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")
        self.vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        

    def format_docs(self, docs):
        """
        Format documents.

        Args:
        - docs: List of documents.

        Returns:
        - Formatted document string.
        """
        return "\n\n".join(doc.page_content for doc in docs)
        
    def contextualize_system_prompt(self):
        contextualize_q_system_prompt = """Given a chat history and the latest user question \
            which might reference context in the chat history, formulate a standalone question \
            which can be understood without the chat history. Do NOT answer the question, \
            If theres no chat history before then return it as it \
            just reformulate it if needed and otherwise return it as is."""
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        retriever = self.vector_store.as_retriever(search_type="similarity")
        history_aware_retriever = create_history_aware_retriever(self.model, retriever, contextualize_q_prompt)
        return history_aware_retriever

    def get_conversational_chain(self):
        """
        Construct a conversational chain for generating responses.

        Returns:
        - Conversational chain for generating responses.
        """
        ### Answer question ###
        prompt_template = """
                            You are a friendly chatbot named Bogu that helps to answer question regarding Indonesian Culture\n
                            Answer the question as detailed as possible from the provided context, make sure to provide all the details\n
                            If the question is about code, answer that you don't know the answer.\n
                            Context:\n {context}?\n
                        """
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompt_template),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = self.contextualize_system_prompt()
        question_answer_chain = create_stuff_documents_chain(self.model, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        return rag_chain
    
    def get_session_history(self,session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]
    
    def run_invoke(self,input_question,session_id):
        get_conversational_chain = self.get_conversational_chain()

        conversational_rag_chain = RunnableWithMessageHistory(
            get_conversational_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        response = conversational_rag_chain.invoke(
                        {"input": input_question},
                        config={"configurable": {"session_id": session_id}},
                    )
        
        return response


    def generate_response(self, text):
        """
        Generate a response by yielding words from the input text.

        Args:
        - text: Input text.

        Yields:
        - Words from the input text with a small delay.
        """
        for word in text.split():
            yield word + " "
            time.sleep(0.05)
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


class ConversationalAI:
    def __init__(self):
        # Initialize class attributes
        self.chat_history = []  # List to store chat history
        self.vector_store = None  # Vector store for embeddings
        # Initialize the AI model
        self.model = ChatGoogleGenerativeAI(model="gemini-pro", 
                                            temperature=0.5, 
                                            convert_system_message_to_human=True,
                                            google_api_key="AIzaSyBYgcagyUPWzHFRyTZO3o8r85oZqmC25E8")

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
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        return text_splitter.split_text(text)

    def create_vector_store(self, text_chunks):
        """
        Create a vector store from text chunks.

        Args:
        - text_chunks: List of text chunks.

        Returns:
        - None
        """
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key="AIzaSyBYgcagyUPWzHFRyTZO3o8r85oZqmC25E8")
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
        
    def contextualized_question(self, input_data):
        """
        Generate a contextualized question based on input data.

        Args:
        - input_data: Input data dictionary containing chat history and question.

        Returns:
        - Contextualized question.
        """
        if input_data.get("chat_history"):
            contextualized_q_chain = self.contextualize_system_prompt()
            return contextualized_q_chain
        else:
            return input_data["question"]

    def contextualize_system_prompt(self):
        """
        Construct a system prompt for generating a standalone question.

        Returns:
        - System prompt for generating a standalone question.
        """
        system_prompt = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""
        prompt_template = ChatPromptTemplate.from_messages(
            [("system", system_prompt), MessagesPlaceholder(variable_name="chat_history"), ("human", "{question}")]
        )
        return prompt_template | self.model | StrOutputParser()

    def get_conversational_chain(self):
        """
        Construct a conversational chain for generating responses.

        Returns:
        - Conversational chain for generating responses.
        """
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
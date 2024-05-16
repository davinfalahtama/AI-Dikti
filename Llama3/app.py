import streamlit as st
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain import HuggingFacePipeline
from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM,pipeline
from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain import PromptTemplate
from langchain.chains import ConversationalRetrievalChain, StuffDocumentsChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory import ChatMessageHistory, ConversationEntityMemory, ConversationBufferMemory, ConversationSummaryMemory

def load_data():
    docs = WikipediaLoader(query="Candi Borobudur",lang='id',load_max_docs=2).load()

    allDocs = "\n\n".join(doc.page_content for doc in docs)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    all_splits = text_splitter.split_text(allDocs)

    embeddings = HuggingFaceEmbeddings(model_name="firqaaa/indo-sentence-bert-base")
    vectorstore = FAISS.from_texts(all_splits, embedding=embeddings)

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})

    return retriever


def main():
    st.set_page_config(
    page_title="Chat Documents",
    page_icon="🧊",
   # layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an extremely cool app!"
    })
    st.header(':sparkles: Mau nanya tentang PMB ITPLN :question:', divider='rainbow')
    st.subheader("Hallo, aku Elena. Temukan informasi seputar PMB ITPLN bersamaku.")
    with st.chat_message("assistant"):
                st.markdown("Kamu mau nanya apa?")

    if "hf" not in st.session_state:
        with st.spinner("Loading Model => Fetching from Hugginface"):
            compute_dtype = getattr(torch, "float16")

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=False,
            )


            model_id = "nvidia/Llama3-ChatQA-1.5-8B"

            tokenizer = AutoTokenizer.from_pretrained(model_id)

            model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map={"": 0},
                    quantization_config=quant_config
            )


            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,
                temperature=0.2,
                return_full_text= False,
                do_sample = True,
                num_return_sequences=1,
                top_k=10,
                eos_token_id=terminators,
            )

            st.session_state["hf"] = HuggingFacePipeline(pipeline=pipe)
    
    if "retriver" not in st.session_state:
        with st.spinner("Loading Data => Fetching from wikipedia"):
            st.session_state["retriver"] = load_data()


    if "chain" not in st.session_state:
        history = ChatMessageHistory()
        history.add_ai_message("Hai!")

        memory = ConversationBufferWindowMemory(k=2,
                                        input_key="question",
                                        memory_key="chat_history",
                                        chat_memory=history,
                                        ai_prefix="Assistant",
                                        output_key = "generated_question",
                                        human_prefix="User")
        
        ret ="""System: Anda adalah chatbot interaktif yang asik untuk menjawab pertanyaan. Kamu bisa mengambil potongan konteks yang diambil berikut ini untuk menjawab pertanyaan tidak apa untuk bilang tidak tahu. Buatlah jawaban yang ringkas 2 kalimat. Selalu berikan jawaban

          {context}
          {chat_history}
          User: {question}

          Assistant:
        """
        prompt_context = PromptTemplate(input_variables=["context", "chat_history", "question"], template=ret)

        condense_template ="""SYSTEM: Gabungkan riwayat obrolan dan pertanyaan lanjutan menjadi pertanyaan mandiri.
        CHAT_HISTORY : {chat_history}
        User: {question}
        Assistant:
        """

        condense_prompt = PromptTemplate(input_variables=["chat_history", "question"], template=condense_template)

        chain = ConversationalRetrievalChain.from_llm(st.session_state["hf"],
                                              retriever=st.session_state["retriver"],
                                              memory=memory,
                                              condense_question_prompt=condense_prompt,
                                              return_generated_question= True,
                                              combine_docs_chain_kwargs={'prompt': prompt_context},
                                              get_chat_history=lambda h : h, # fix support with memory
                                              )
        
        st.session_state["chain"] = chain
    
    if "response" not in st.session_state:
        st.session_state["response"] = None
    
    if "history" not in st.session_state:
        st.session_state["history"] = []

     #Display history with user messages as even elements and AI responses as odd elements
    for i, message in enumerate(st.session_state["history"]):
        if i % 2 == 0:
            with st.chat_message("user"):
                st.write(f"{message}")
        else:
            with st.chat_message("assistant"):
                st.write(f"{message}")

    # Accept user input
    if prompt := st.chat_input("Say something"):
        with st.chat_message("user"):
            st.markdown(prompt)
            st.session_state["history"].append(prompt)  # Append user prompt to history

        start_inference_time = time.time()  # Record inference start time

        # Perform inference using your pre-defined `chain` function
        
        #If statement
        if(True):
            pass
        response = st.session_state["chain"](prompt) #QNA borobudur
        st.session_state["response"] = response

        end_inference_time = time.time()  # Record inference end time
        inference_time = end_inference_time - start_inference_time  # Calculate inference time

        with st.chat_message("assistant"):
            st.write(response['answer'].strip())
            st.session_state["history"].append(response['answer'].strip())  # Append AI response to history
            st.info(f"Inference time: {inference_time:.2f} seconds.")

if __name__ == "__main__":
    main()

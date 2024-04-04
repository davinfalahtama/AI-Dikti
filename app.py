import streamlit as st
from chat_models.google_gemini import ChatGoogleGemini
from chat_models.openai import ChatOpenAi
from langchain_core.messages import HumanMessage
import time


def main():
    st.set_page_config(
        page_title="Chat Documents",
        page_icon="🗿",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://www.extremelycoolapp.com/help',
            'Report a bug': "https://www.extremelycoolapp.com/bug",
        })
    st.header('Ask your Docs')
    option = st.selectbox(
        'Choose Your Model',
        ('Google Gemini', 'OpenAI'),
        key='model_selectbox',  # Assign a unique key for customization
        help='Select the model you want to use for document retrieval'  # Add help text
    )
    st.divider()


    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = {}

    if "session_id" not in st.session_state:
        st.session_state["session_id"] = ''

    if "model" not in st.session_state:
        # Set the model based on user selection (avoiding redundant checks)
        if option == 'Google Gemini':
            st.session_state["model"] = ChatGoogleGemini(st.session_state["chat_history"])
        elif option == 'OpenAI':
            # Assuming ChatOpenAi is defined elsewhere
            st.session_state["model"] = ChatOpenAi(st.session_state["chat_history"])


    if "chat_input" not in st.session_state:
        st.session_state["chat_input"] = True

    with st.sidebar:
        st.title("Menu:")
        st.session_state["session_id"] = st.text_input('Your name / session')
        pdf_docs = st.file_uploader("Upload your PDF/CSV Files and Click on the Submit & Process Button",
                                    accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = st.session_state["model"].process_files(
                        pdf_docs)
                    text_chunks = st.session_state["model"].split_text_into_chunks(
                        raw_text)
                    st.session_state["model"].create_vector_store(text_chunks)
                st.success("Done")
                st.session_state["chat_input"] = False
            else:
                st.error("Please upload PDF files first.")

    if st.session_state["session_id"] in st.session_state["chat_history"]:
        user_session = st.session_state["chat_history"].get(
            st.session_state["session_id"])
        for messages in user_session.messages:
            if isinstance(messages, HumanMessage):
                with st.chat_message("user"):
                    st.markdown(messages.content)
            else:
                with st.chat_message("assistant"):
                    st.markdown(messages.content)

    # Accept user input
    if prompt := st.chat_input("Say something", disabled=st.session_state["chat_input"]):
        start_inference_time = time.time()  # Catat waktu awal inferensi
        with st.chat_message("user"):
            st.markdown(prompt)

        if pdf_docs:
            st.write(st.session_state["model"].vector_store)
            ai_msg = st.session_state["model"].run_invoke(
                prompt, st.session_state["session_id"])
            st.session_state["chat_history"] = st.session_state["model"].store
            end_inference_time = time.time()
            inference_time = end_inference_time - start_inference_time
            with st.chat_message("assistant"):
                response = st.write_stream(
                    st.session_state["model"].generate_response(ai_msg['answer']))
                st.info(f"Inference time: {inference_time:.2f} seconds.")

        else:
            st.error("Please upload PDF files first.")

if __name__ == "__main__":
    main()

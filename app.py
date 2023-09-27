import os

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
import tiktoken 

from HtmlTemplate import css, bot_template, user_template

def get_pdf_text(pdf_list):
    text = ""
    for pdf_file in pdf_list:
        with open(pdf_file, "rb") as file:
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
    return text

def get_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstores(chunks):
    embedings = OpenAIEmbeddings()
    vectorestore = FAISS.from_texts(texts=chunks, embedding=embedings)
    return vectorestore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain  = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever=vectorstore.as_retriever(),
        memory = memory
    )
    return conversation_chain


def handle_input(question):
    response = st.session_state.conversation({'question':question})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            
    
def main():
    load_dotenv()
    st.set_page_config(page_title="Chat PDF", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = None
    
    st.title(":books: Chat with multiple PDF's :books:")
    question = st.text_input("Ask a question about your documents")
    
    if question:
        handle_input(question)
    


    with st.sidebar:
        st.title("Enter your pdf files here")
        uploaded_files = st.file_uploader("Upload your pdf here and click process", accept_multiple_files=True)
        if st.button("Process"):
            if not uploaded_files:
                st.warning("Please upload at least one PDF file.")
                return

            pdf_paths = []  # Store the paths of uploaded PDFs
            for file in uploaded_files:
                with open(file.name, "wb") as pdf_file:
                    pdf_file.write(file.read())
                pdf_paths.append(file.name)

            with st.spinner("Reading your PDFs ;-)"):
                # Get pdf text
                raw_text = get_pdf_text(pdf_paths)
                
                # Get the text chunks
                text_chunks = get_chunks(raw_text)
                
                # Create vector stores
                vectorestore = get_vectorstores(text_chunks)
                
                #create a convo chain
                st.session_state.conversation = get_conversation_chain(vectorestore)

            # Clean up uploaded PDF files
            for pdf_path in pdf_paths:
                os.remove(pdf_path)

if __name__ == "__main__":
    main()

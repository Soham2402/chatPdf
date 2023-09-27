import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import os

def get_pdf_text(pdf_list):
    text = ""
    for pdf_file in pdf_list:
        with open(pdf_file, "rb") as file:
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
    return text

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat PDF", page_icon=":books:")
    st.title(":books: Chat with multiple PDF's :books:")

    question = st.text_input("Ask a question about your documents")

    with st.sidebar:
        st.subheader("Enter your pdf files here")
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
                st.write(raw_text)
                # Get the text chunks
                # Create vector stores

            # Clean up uploaded PDF files
            for pdf_path in pdf_paths:
                os.remove(pdf_path)

if __name__ == "__main__":
    main()

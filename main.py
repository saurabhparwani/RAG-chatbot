import os
import pickle
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from dotenv import load_dotenv

# Load OpenAI API key from .env file
load_dotenv()

# Set up OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("OpenAI API key not found. Please set it in the .env file.")
    st.stop()

# Set up the Streamlit title and sidebar
st.title("Bot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

# Allow the user to enter URLs in the sidebar
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i + 1}")
    if url:
        urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "chroma_store_openai"

main_placeholder = st.empty()

# Initialize OpenAI Embeddings for document processing
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Initialize LLM (Large Language Model)
llm = ChatOpenAI(temperature=0.9, max_tokens=500, openai_api_key=openai_api_key)  # Using ChatOpenAI

# Function to process URLs
if process_url_clicked:
    if not urls:
        st.warning("Please provide at least one URL to process.")
    else:
        # Start processing
        with st.spinner("Processing data... Please wait..."):
            try:
                # Load data from the URLs
                loader = UnstructuredURLLoader(urls=urls)
                data = loader.load()
                st.write("Data Loaded ✅")

                # Split the data into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    separators=['\n\n', '\n', '.', ','],
                    chunk_size=1000
                )
                docs = text_splitter.split_documents(data)
                st.write(f"Data Split into {len(docs)} chunks ✅")

                # Generate embeddings for the chunks and create a Chroma index
                vectorstore_openai = Chroma.from_documents(docs, embeddings, persist_directory=file_path)
                st.write("Embeddings Created ✅")

                # Save the Chroma index to a file
                vectorstore_openai.persist()
                st.write("Chroma Index Saved ✅")

            except Exception as e:
                st.error(f"An error occurred: {e}")

# Query input from the user
query = st.text_input("Ask a Question: ")

if query:
    if os.path.exists(file_path):
        try:
            # Load the Chroma index from the saved file
            vectorstore_openai = Chroma(persist_directory=file_path, embedding_function=embeddings)

            # Set up the retrieval-based question-answering chain
            retriever = vectorstore_openai.as_retriever(search_type="similarity", search_kwargs={"k": 3})
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)

            # Fetch the answer
            result = chain({"question": query}, return_only_outputs=True)

            # Display the answer to the user
            st.header("Answer")
            st.write(result["answer"])

            # Display the sources from where the answer was derived
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  # Split sources by newline
                for source in sources_list:
                    st.write(source)

        except Exception as e:
            st.error(f"An error occurred while fetching the answer: {e}")
    else:
        st.warning("Please process URLs first.")
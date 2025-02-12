import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os

# Load environment variables (API keys, etc.)
load_dotenv()

def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDFs."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""  # Avoid NoneType errors
    return text

def get_text_chunks(text):
    """Split text into manageable chunks for embeddings."""
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    return text_splitter.split_text(text)

def get_vectorstore(text_chunks):
    """Convert text chunks into a FAISS vector store."""
    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

def get_conversation_chain(vectorstore):
    """Create a conversational retrieval chain."""
    llm = ChatOpenAI(model_name="gpt-4", temperature=0.5)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)

def handle_userinput(user_question):
    """Process user input and generate a response."""
    if not user_question.strip():
        return  # Ignore empty input

    print(f"User Question: {user_question}")

    if "conversation" in st.session_state and st.session_state.conversation:
        print("Searching PDFs for an answer...")
        response = st.session_state.conversation({'question': user_question})
        chat_history = response.get("chat_history", [])

        bot_reply = chat_history[-1].content.strip() if chat_history else "I'm not sure, could you rephrase?"
    else:
        print("No PDFs uploaded. Using GPT-4 directly.")
        llm = ChatOpenAI(model_name="gpt-4", temperature=0.5)
        bot_reply = llm.predict(user_question)

    print(f"Bot Reply: {bot_reply}")

    # Store conversation history
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    st.session_state.chat_history.append({"role": "bot", "content": bot_reply})

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Chat with PDFs", page_icon="📚")

    # Initialize session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.title("Chat with Your PDFs 📄🤖")

    # File Upload
    pdf_docs = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

    if pdf_docs and st.button("Process PDFs"):
        with st.spinner("Processing PDFs..."):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            vectorstore = get_vectorstore(text_chunks)
            st.session_state.conversation = get_conversation_chain(vectorstore)
        st.success("PDFs processed successfully! You can now ask questions.")

    # Chat Interface
    user_question = st.text_input("Ask a question...")

    if user_question:
        handle_userinput(user_question)

    # Display chat history
    for chat in st.session_state.chat_history:
        role = "You" if chat["role"] == "user" else "AI"
        st.markdown(f"**{role}:** {chat['content']}")

if __name__ == "__main__":
    main()

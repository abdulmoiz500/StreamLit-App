import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS 
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain_community.llms import HuggingFaceHub
from langchain_community.chat_models import ChatOpenAI

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


# ============= INTRODUCE CHATGPT ================

def handle_userinput(user_question):
    print(f"User Question: {user_question}")  
    # handle_userinput in

    if "conversation" in st.session_state and st.session_state.conversation:
        print("Searching PDFs for an answer...")  
        response = st.session_state.conversation({'question': user_question})
        chat_history = response.get('chat_history', [])

        if chat_history and chat_history[-1].content.strip():
            bot_reply = chat_history[-1].content.strip()
            print(f"PDF Response Found: {bot_reply}")  
        else:
            print("No valid response from PDFs. Using GPT-4 fallback.")  
            llm = ChatOpenAI()
            bot_reply = llm.predict(user_question)  
    else:
        print("No PDFs uploaded. Using GPT-4 directly.")  
        llm = ChatOpenAI()
        bot_reply = llm.predict(user_question)  

    print(f"Bot Reply: {bot_reply}") 

    # Ensure chat history is stored correctly
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    st.session_state.chat_history.append({"role": "bot", "content": bot_reply})

    # Display only the latest user input and bot reply
    st.markdown(f"**You:** {user_question}")
    st.markdown(f"**AI:** {bot_reply}")

# ============= END CHATGPT ================



def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with PDFs", page_icon=":books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.write(css, unsafe_allow_html=True)

    # Display only the last few exchanges, avoiding duplicates
    if len(st.session_state.chat_history) >= 2:
        last_user_input = st.session_state.chat_history[-2]["content"]
        last_bot_reply = st.session_state.chat_history[-1]["content"]
        st.markdown(f"**You:** {last_user_input}")
        st.markdown(f"**AI:** {last_bot_reply}")

    user_question = st.text_input("Ask a question...")

    if user_question:
        print("User input detected, calling handle_userinput()")  # Debugging log
        handle_userinput(user_question)  # Call function explicitly
        st.rerun()  # Refresh UI




if __name__ == '__main__':
    main()

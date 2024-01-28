from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_models import ChatOpenAI

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain

conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True)

llm = ChatOpenAI(model='gpt-3.5-turbo-1106', temperature=0.0)
qa = load_qa_chain(llm=llm, chain_type='stuff')

# print(os.environ.get('OPENAI_API_KEY'))

if 'messages' not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("Contract Chat ⚖️✍🏽  ")
    st.markdown('''
        ## About
        This app answers questions about contracts.
        Was built using:
         - [Streamlit](https://streamlit.io)
         - [LangChain](https://langchain.io)
         - [OpenAI](https://openai.com)            
    ''')

    add_vertical_space(5)
    st.write("Made with ❤️ by [Innocent Kithinji](https://www.linkedin.com/in/ikithinji/)")


def show_chats():
    for i, message in enumerate(st.session_state.messages):
        if message['type'] == 'user':
            with st.chat_message('user'):
                st.write(message['content'])
        elif message['type'] == 'ai':
            with st.chat_message('ai'):
                st.markdown(message['content'])


def main():
    st.header("Contract Chat ⚖️✍🏽  ")
    st.write("This app answers questions about contracts.")

    pdf = st.file_uploader("Upload a Contract as a PDF file", type=["pdf"])

    if pdf:
        reader = PdfReader(pdf)
        text = ""

        for page in reader.pages:
            text += page.extract_text()

        spliter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=160, length_function=len)
        chunks = spliter.split_text(text)

        embeddings = OpenAIEmbeddings()

        db = FAISS.from_texts(chunks, embeddings)

        query = st.chat_input("Ask me about the contract")

        if query:
            st.session_state.messages.append({'type': 'user', 'content': query})
            docs = db.similarity_search(query, k=10)
            response = qa.run(input_documents=docs, question=query)
            st.session_state.messages.append({'type': 'ai', 'content': response})

    show_chats()


if __name__ == '__main__':
    main()

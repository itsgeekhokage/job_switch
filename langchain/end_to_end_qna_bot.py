from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite"
)

st.title("End-to-End Q&A Bot")

# Store conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ChatGPT-style input box
if query := st.chat_input("What's on your mind?"):

    # Display user message
    with st.chat_message("user"):
        st.markdown(query)

    # Store user message
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    query = st.session_state.messages

    # Get response from LLM
    response = llm.invoke(query)
    answer = response.content[0]["text"]

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(answer)

    # Store assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
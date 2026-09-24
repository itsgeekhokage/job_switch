
from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from langchain_groq import ChatGroq
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver


# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Aham Brahmasmi", page_icon="🕉️")

st.subheader("Aham Brahmasmi - Ask me anything!")


# -----------------------------
# Initialize memory
# -----------------------------
if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()
    st.session_state.history = []


# -----------------------------
# Initialize agent
# -----------------------------
if "agent" not in st.session_state:

    search = GoogleSerperAPIWrapper()

    llm = ChatGroq(
        model="openai/gpt-oss-20b"
    )

    tools = [search.run]

    st.session_state.agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            "You are a helpful assistant that can answer questions "
            "based on the information provided by the Google Serper API. "
            "Use the information from the search results to answer "
            "the user's question."
        ),
        checkpointer=st.session_state.memory,
    )


agent = st.session_state.agent


# -----------------------------
# Initialize UI chat history
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------
# Display previous messages
# -----------------------------
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# -----------------------------
# Chat input
# -----------------------------
query = st.chat_input("Enter your question here:")


if query:

    # Display user message
    st.chat_message("user").markdown(query)
    st.session_state.history.append({"role": "user", "content": query})

    # Store user message for UI
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })


    # Invoke LangGraph agent
    response = agent.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": query
                }
            ]
        },
        {
            "configurable": {
                "thread_id": "1"
            }
        },
        stream_mode="messages"
    )

    ai_container = st.chat_message("assistant")

    with ai_container:
        msg = st.empty()
        message = ""
        for chunk in response:
            message = message + chunk[0].content
            msg.markdown(message)

        st.session_state.history.append({
            "role": "assistant",
            "content": message
        })

    # Get latest assistant message
    # answer = response["messages"][-1].content


    # Display assistant response
    # st.chat_message("assistant").markdown(answer)


    # Store assistant message for UI
    # st.session_state.messages.append({
    #     "role": "assistant",
    #     "content": answer
    # })

    # st.session_state.history.append({
    #     "role": "assistant",
    #     "content": answer
    # })



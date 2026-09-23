from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

model = ChatGroq(model="openai/gpt-oss-20b")
search = GoogleSerperAPIWrapper()

saver = InMemorySaver()
agent = create_agent(
    model=model,
    tools=[search.run],
    system_prompt="You are a helpful assistant that can answer questions based on the information provided by the Google Serper API. Use the information from the search results to answer the user's question.",
    checkpointer=saver,
)

while True:
    query = input("Enter your question (or 'exit' to quit): ")
    if query.lower() == "exit":
        break

    response = agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
        {"configurable": {"thread_id": "1"}}
        )
    print("Answer:", response["messages"][-1].content)

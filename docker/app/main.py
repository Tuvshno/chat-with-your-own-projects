"""
Handles the API endpoint for processing and responding to user queries.
We create Retrieval-Augmented Generation using Pinecone, OpenAI, and LangChain.
"""

from typing import Union
from fastapi import FastAPI
from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)
from langchain.schema import (
    HumanMessage,
    AIMessage
)

# Load ENV Variables (For Local Running)
load_dotenv()

# Initialize the LLM
llm = ChatOpenAI()

# Name of the Pinecone Index
index_name="documentation" # Add your index name here

# Get OpenAI Embeddings
embeddings = OpenAIEmbeddings()

# Retrieve our Pinecone Vector Database
store = Pinecone.from_existing_index(index_name, embeddings)

# Create Information for our Vector Store
vectorstore_info = VectorStoreInfo(
  name="documentation for my resume",
  description="contains information about my projects",
  vectorstore=store
)

# Convert the document store into a langchain toolkit
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

# This is the System Message that deteremines how OpenAI will chat
system_message = """
    "Your name is {} and you are going to be explaining your projects to recruiters."
    "You are a Software Engineer."
    "You want to answer every question as if you are explaining a project to a recruiter."
    """

# Add the toolkit to an end-to-end LC
agent_executor = create_vectorstore_agent(
    llm=llm,
    prefix=system_message,
    toolkit=toolkit,
    verbose=True,
)

# Save Previous Messages for Memory 
messages = []

app = FastAPI()


"""
Handles the API endpoint for root.

Args:
    None.

Returns:
    dict: Returns basic response to know if the API is working.
"""
@app.get("/")
def read_root():
    return {"Hello": "World"}

"""
Handles the API endpoint for processing and responding to user queries.

Args:
    query (str): The user's query as a string.

Returns:
    dict: A response to the query from the RAG Pipelines.
"""
@app.get("/query/{query}")
def query_response(query):

    # Add the query to the messages array as a human message
    question = HumanMessage(content=query)
    messages.append(question)

    # Query the messages into the RAG Pipeline
    ai_response = ""
    for chunk in agent_executor.stream(messages):
        if "actions" in chunk:
            for action in chunk["actions"]:
                print(f"Calling Tool: `{action.tool}` with input `{action.tool_input}`")
        # Observation
        elif "steps" in chunk:
            for step in chunk["steps"]:
                print(f"Tool Result: `{step.observation}`")
        # Final result
        elif "output" in chunk:
            print(f'Final Output: {chunk["output"]}')
            ai_response = chunk["output"]
        else:
            raise ValueError()
        print("---")

        # Add the response to the messages for memory
        response = AIMessage(content=ai_response)
        messages.append(response)
        
    return {"message": messages[-1].content}


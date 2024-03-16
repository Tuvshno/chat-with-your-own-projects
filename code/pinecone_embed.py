"""
This handles embedding your own documents into your vector database so you can perform
RAG operations with your own data!
"""

from pinecone import Pinecone
from dotenv import load_dotenv
from pinecone import ServerlessSpec
import time
from langchain_openai import OpenAIEmbeddings
from tqdm.auto import tqdm  # for progress bar
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from dotenv import load_dotenv
from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone
from tqdm.auto import tqdm  
import pandas as pd
import openai

load_dotenv()

# ADD YOUR OWN INFORMATION HERE
cloud_provider = "aws"
cloud_region="us-west-2"
index_name = 'documentation'
sourcePath = "./myDocs.pdf" 

# ---------------------------------

def get_embeddings(text):
    response = client.embeddings.create(
        input=[text],  # The API expects a list of texts. You can input multiple texts if needed.
        model="text-embedding-ada-002"
    )
    # Extracting embeddings from the response.
    embeddings = response.data[0].embedding
    return embeddings

# Gets the Pinecone DB
pc = Pinecone() # <-- API KEY FROM ENVIRONMENT VARIABLES

# Creates a serverless instance
spec = ServerlessSpec(
    cloud=cloud_provider, region=cloud_region
)

# Get existing indexes from your Pinecone DB
existing_indexes = [
    index_info["name"] for index_info in pc.list_indexes()
]

# check if index already exists (it shouldn't if this is first time)
if index_name not in existing_indexes:
    # if does not exist, create index
    pc.create_index(
        index_name,
        dimension=1536,  # dimensionality of ada 002
        metric='cosine',
        spec=spec
    )
    # wait for index to be initialized
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

# connect to index
index = pc.Index(index_name)
time.sleep(1)
# view index stats
print(index.describe_index_stats())

from openai import OpenAI

client = OpenAI() # <- API KEY FROM ENV VARIABLE

batch_size = 100

# Load our documents and split them into chunks
loader = TextLoader(sourcePath) 
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# Add our vector embeddings to our vector database
for i in tqdm(range(0, len(docs))):
    for doc in docs:
        # generate unique ids for each chunk
        ids = [f"{i}-{sourcePath}"]
        # embed text
        texts = doc.page_content
        embeddings = [get_embeddings(texts)]
        # get metadata to store in Pinecone
        metadata = [
            {'text': texts,
            'source': sourcePath
            }
        ]
        # add to Pinecone
        index.upsert(vectors=zip(ids, embeddings, metadata))
    

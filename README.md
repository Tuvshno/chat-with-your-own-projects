# Deploy an LLM RAG with your data

This repository is a template to deploy a Retrieval Augmented Generation ChatBot with OpenAI and Pinecone using your data! It uses docker for deployment, FastAPI as the backend for your server, and OpenAI as your Large Language Model (LLM).

### What is Retrieval Augmented Generation (RAG)? 

RAG is the process of optimizing the output of a large language model by referencing a knowledge base outside of its training data sources before generating a response. This means that we can add our documentation to the LLM and get catered responses based on the documentation you inputted.
This is a common technique in utilizing LLMs to answer based on documents such as Bank Statements, Internal Documentation, etc.

As the user, you have three inputs that you have control over in a LLM: Input, Prompt, and Context. 
- Input: is the question that you ask an LLM
- Prompt: are the instructions, examples, or a specific setup that directs the model on how to answer the question or what style to use in its response. The prompt helps in setting the tone, format, and direction for the model's output. It acts as a template that tells the model how to structure its response in the context of the input question 
- Context: is the additional information or data provided to the LLM that it can use to make its responses more accurate, relevant, and personalized. 

The RAG Pipeline works by receiving the input from the user, sending it through a query encoder, and into a documents retriever. The document retriever will look for relevant context in your database of documents. After finding relevant context, it adds the context into the context input of the LLM.
As a result, you receive output that is tailored to the context relevant to your documentation.

### Purpose of this template

This template is a quick way to get up and running with a Chatbot that uses your data. It has a docker image that will run after inputting the relevant API Key and Vector Database Index Names. It also provides a solution for converting your documents into vector embeddings. 

### What Architecture does this project deploy?

I am glad you asked! This RAG Pipeline utilizes a slightly optimized version of the traditional architecture. Normally after receiving an input query in the document retriever, the retriever would conduct a similarity search in the vector database based on the input and return the top-K outputs.
This strategy is completely fine, but it tends to fall short due to its over-reliance on Initial Query Representation. 

This template follows a Reflection Strategy. Reflection is a prompting strategy used to improve the quality and success rate of document retrieval systems. It involves prompting an LLM to reflect on and critique its past actions, sometimes incorporating additional external information.
In this architecture, after the document retriever receives an input query, it will create an initial response and reflect on its response by creating observations and then requery with the new observations until it receives a suitable final answer.
This looped reflection improves the performance and success rate of the LLM greatly.

In addition, the template allows you to change the prompt input. This will allow you to add additional instructions on how you want the LLM to respond to your inputs.

### How to use the template

`code` folder contains `pinecone_embed.py` that allows you to create vector embeddings for your documents and insert them into a Pinecone Vector Database Index. 
`docker` folder contains the code to run the code both locally or deploy using docker.

There are detailed instruction on how to use each inside the code.

1. Clone the GitHub
2. Input relevant API Keys inside `.env` and `dockerfile`
3. Run `uvicorn app.main:app --host 0.0.0.0 --port 80` to run the code locally
4. Use the `dockerfile` to deploy to a cloud service provider

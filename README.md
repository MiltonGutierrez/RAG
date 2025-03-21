### Escuela Colombiana de Ingeniería

### Arquitectura Empresarial - AREP

# Build a Retrieval Augmented Generation (RAG)
One of the most powerful applications enabled by LLMs is sophisticated question-answering (Q&A) chatbots. These are applications that can answer questions about specific source information. These applications use a technique known as Retrieval Augmented Generation, or RAG.

## Pinecon 
Pinecone is a vector database with broad functionality.
## Setup

To use the `PineconeVectorStore`, you first need to install the partner package, as well as the other packages used throughout this notebook.

```sh
pip install -qU langchain-pinecone pinecone-notebooks
```

## Credentials

Create a new Pinecone account, or sign into your existing one, and create an API key to use in this notebook.

```python
import getpass
import os
from pinecone import Pinecone, ServerlessSpec

if not os.getenv("PINECONE_API_KEY"):
    os.environ["PINECONE_API_KEY"] = getpass.getpass("Enter your Pinecone API key: ")

pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)
```

---

## Initialization

Before initializing our vector store, let's connect to a Pinecone index. If one named `index_name` doesn't exist, it will be created.

```python
import time

index_name = "langchain-test-index"  # change if desired

existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)
```

![alt text](image.png)

Now that our Pinecone index is set up, we can initialize our vector store.

```python
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = PineconeVectorStore(index=index, embedding=embeddings)
```

---

## Manage Vector Store

Once you have created your vector store, you can interact with it by adding and deleting different items.

### Add Items to Vector Store

We can add items to our vector store by using the `add_documents` function.

## Query Vector Store

Once your vector store has been created and the relevant documents have been added, you will most likely wish to query it during the running of your chain or agent.

# RAG creation


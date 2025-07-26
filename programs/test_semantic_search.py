import os
import glob
from dotenv import load_dotenv

import gradio as gr

from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain_community.llms import Ollama
# from langchain_community.embeddings import OllamaEmbeddings

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline
import torch
from langchain.embeddings import HuggingFaceEmbeddings


LLM_MODEL = "Qwen/Qwen3-4B"
# "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-4B"
# "sentence-transformers/all-MiniLM-L6-v2"
# "Qwen/Qwen3-Embedding-0.6B"
# "Qwen/Qwen3-Embedding-4B"
# "Qwen/Qwen3-Embedding-8B"
# "BAAI/bge-small-en"
# "intfloat/multilingual-e5-small"
# "thenlper/gte-small"

# db_name = "dbs/fixwing_vector_db"
# path = "/home/hoda/Documents/Hooma/Fixed-wing/my_papers/*"

VECTOR_DB_NAME = "dbs/2-Qwen3-Embedding-4B"
INPUT_PATH = "semantic_search_inputs/2/*"
DELETE_VECTOR_DB = False

# Load environment variables in a file called .env
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "your-key-if-not-using-env")
print(os.environ["HF_TOKEN"])

# Read in documents using LangChain's loaders
# Take everything in all the sub-folders of our knowledgebase
docs = glob.glob(INPUT_PATH)

# With thanks to CG and Jon R, students on the course, for this fix needed for some users
text_loader_kwargs = {"encoding": "utf-8"}
# If that doesn't work, some Windows users might need to uncomment the next line instead
# text_loader_kwargs={'autodetect_encoding': True}

# documents = []
# for i, doc in enumerate(docs):
#     loader = PyPDFLoader(doc)
#     texts = loader.load()
#     for text in texts:
#         text.metadata["doc_type"] = str(i)
#         documents.append(text)

documents = []
for i, doc in enumerate(docs):
    loader = TextLoader(doc)
    texts = loader.load()
    for text in texts:
        text.metadata["doc_type"] = str(i)
        documents.append(text)

print(f"Len docs: {len(documents)}")

print(f"Doc0: {documents[0]}")

# split the text into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

print(f"Num chunks: {len(chunks)}")

doc_types = set(chunk.metadata["doc_type"] for chunk in chunks)
print(f"Document types found: {', '.join(doc_types)}")

print(EMBEDDING_MODEL)
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
# embeddings = OllamaEmbeddings(model=MODEL)


# Check if a Chroma Datastore already exists - if so, use that
if DELETE_VECTOR_DB is True:
    if os.path.exists(VECTOR_DB_NAME):
        Chroma(
            persist_directory=VECTOR_DB_NAME, embedding_function=embeddings
        ).delete_collection()
    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=VECTOR_DB_NAME
    )
else:
    if os.path.exists(VECTOR_DB_NAME):
        vectorstore = Chroma(
            persist_directory=VECTOR_DB_NAME, embedding_function=embeddings
        )
        # vector
    else:
        # Create our Chroma vectorstore!
        vectorstore = Chroma.from_documents(
            documents=chunks, embedding=embeddings, persist_directory=VECTOR_DB_NAME
        )
print(f"Vectorstore created with {vectorstore._collection.count()} documents")

# Get one vector and find how many dimensions it has
collection = vectorstore._collection
sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
dimensions = len(sample_embedding)
print(f"The vectors have {dimensions:,} dimensions")


# create a new Chat

# llm = ChatOpenAI(temperature=0.7, model_name=MODEL, base_url='http://localhost:11434')

# llm = Ollama(model=MODEL)


tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
# device = torch.device("cpu")

model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
)
# .to(device)

print("HI")

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1,
    return_full_text=False,
)

llm = HuggingFacePipeline(pipeline=pipe)

# the retriever is an abstraction over the VectorStore that will be used during RAG
retriever = vectorstore.as_retriever()

# set up a new conversation memory for the chat
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# putting it together: set up the conversation chain with the GPT 4o-mini LLM, the vector store and memory
conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm, retriever=retriever, memory=memory
)


def chat(message, history):
    result = conversation_chain.invoke({"question": message})
    return result["answer"]


if __name__ == "__main__":

    """
    # set up the conversation memory for the chat
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # putting it together: set up the conversation chain with the GPT 4o-mini LLM, the vector store and memory
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=retriever, memory=memory
    )

    query = "How to use reinforcement learning for fixed wing landing"
    result = conversation_chain.invoke({"question": query})
    print(result["answer"])
    """

    view = gr.ChatInterface(chat, type="messages").launch(inbrowser=True)
    # , share=True

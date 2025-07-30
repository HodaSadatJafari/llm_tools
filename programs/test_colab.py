from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import glob
from langchain.text_splitter import CharacterTextSplitter
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os
import torch
from langchain.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import gradio as gr

db_name = "dbs/2"
path = "semantic_search_inputs/2/*"

docs = glob.glob(path)

# With thanks to CG and Jon R, students on the course, for this fix needed for some users
text_loader_kwargs = {'encoding': 'utf-8'}

documents = []
for i, doc in enumerate(docs):
    loader = TextLoader(doc)
    texts = loader.load()
    for text in texts:
        text.metadata["doc_type"] = str(i)
        documents.append(text)

# split the text into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)


from langchain.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# Check if a Chroma Datastore already exists - if so, use that

if os.path.exists(db_name):
    Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()
    # vectorstore = Chroma(persist_directory=db_name, embedding_function=embeddings)
# else:
    # Create our Chroma vectorstore!
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)
print(f"Vectorstore created with {vectorstore._collection.count()} documents")


# Get one vector and find how many dimensions it has

collection = vectorstore._collection
sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
dimensions = len(sample_embedding)
print(f"The vectors have {dimensions:,} dimensions")


# create a new Chat


model_id = "Qwen/Qwen3-4B"
# "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# "TheBloke/TinyLLama-1.1B-Chat-GGUF"
# "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
# # device = torch.device("cpu")

model = AutoModelForCausalLM.from_pretrained(
    model_id,
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
    return_full_text=False
)

llm = HuggingFacePipeline(pipeline=pipe)



# set up the conversation memory for the chat
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

# the retriever is an abstraction over the VectorStore that will be used during RAG
retriever = vectorstore.as_retriever()

# putting it together: set up the conversation chain with the GPT 4o-mini LLM, the vector store and memory
conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)


# Wrapping in a function - note that history isn't used, as the memory is in the conversation_chain

def chat(message, history):
    result = conversation_chain.invoke({"question": message})
    return result["answer"]


# And in Gradio:

view = gr.ChatInterface(chat, type="messages").launch(inbrowser=True, share=True, debug=True)
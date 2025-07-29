import os
import torch
import gradio as gr
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.prompts import PromptTemplate


class DocumentChatbot:
    def __init__(
        self,
        document_directory,
        llm_model="Qwen/Qwen3-4B",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    ):
        # Initialize key components
        self.LLM_MODEL = llm_model
        self.EMBEDDING_MODEL = embedding_model

        self.embeddings = self.create_embeddings()
        self.db_name = (
            document_directory.split("/")[-1]
            + "_llm_"
            + self.LLM_MODEL.split("/")[-1]
            + "_embedding_"
            + self.EMBEDDING_MODEL.split("/")[-1]
        )
        self.documents = self.load_documents(document_directory)
        self.vector_store = self.create_vector_store()
        self.llm = self.initialize_qwen_model()
        self.rag_pipeline = self.create_rag_pipeline()

    def create_embeddings(self):
        """Create embeddings for document processing"""
        return HuggingFaceEmbeddings(
            model_name=self.EMBEDDING_MODEL,
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        )

    def load_documents(self, directory):
        """Load documents from a directory"""
        # Support multiple file types
        loader = DirectoryLoader(
            directory,
            glob="**/*",  # Match all files
            loader_cls=TextLoader,
            show_progress=True,
        )
        documents = loader.load()

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=100
        )
        return text_splitter.split_documents(documents)

    def create_vector_store(self):
        """Create Chroma vector store"""
        # Persist directory for caching embeddings
        persist_directory = f"./dbs/{self.db_name}"
        os.makedirs(persist_directory, exist_ok=True)

        return Chroma.from_documents(
            documents=self.documents,
            embedding=self.embeddings,
            persist_directory=persist_directory,
        )

    def initialize_qwen_model(self):
        """Initialize Qwen3 model"""
        llm = LLM(
            model=self.LLM_MODEL,
            # max_new_tokens=500,
            enforce_eager=True,
            dtype="bfloat16",
            gpu_memory_utilization=0.8,
        )

        return llm

    def create_rag_pipeline(self):
        retriever = self.vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        template = """
                    You are an assistant for question-answering tasks.
                Use the following pieces of retrieved context to answer the 
                question.
                    If you don't know the answer, just say that you don't know.
                    Please provide the answer in the English language.

                    Question: {question}
                    Context: {context}

                    Answer:
                """

        prompt = PromptTemplate.from_template(template)

        qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True,
        )
        return qa_chain

    def display_response_details(self, response):
        print("Query:", response["query"])
        print("Result:", response["result"])

        # Extract metadata from the first source document
        source_doc = response["source_documents"][0].metadata
        print("Source:", source_doc["source"])
        print("Description:", source_doc["description"])
        print("Title:", source_doc["title"])

        return response["result"]

    def chat_with_documents(self, query, history):
        """Process user query and return response"""
        try:
            # Run RAG pipeline
            result = self.rag_pipeline.invoke({"query": question, "context": retriever})
            return self.display_response_details(result)

        except Exception as e:
            return f"An error occurred: {str(e)}"

    def launch_gradio_interface(self):
        """Create Gradio interface for document chatbot"""
        iface = gr.ChatInterface(
            fn=self.chat_with_documents,
            title="Document Chat with Qwen3 RAG",
            description="Chat with your documents using Retrieval-Augmented Generation",
            theme="soft",
            examples=[
                "موضوعات اصلی در این مستندها چیست؟",
                "نکات مهم را خلاصه سازی کن.",
                "مهم ترین اطلاعات را استخراج کن.",
            ],
            type="messages",
        )

        # Launch the interface
        iface.launch(debug=True)


def main():
    # Specify directory containing documents
    DOCUMENT_DIRECTORY = "./semantic_search_inputs/3"

    # Initialize and launch chatbot
    chatbot = DocumentChatbot(
        document_directory=DOCUMENT_DIRECTORY,
        llm_model="Qwen/Qwen3-4B",
        embedding_model="Qwen/Qwen3-Embedding-0.6B",
        # "Qwen/Qwen3-Embedding-4B",
    )
    chatbot.launch_gradio_interface()


if __name__ == "__main__":
    main()

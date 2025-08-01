import os
import torch
import gradio as gr
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv


load_dotenv(override=True)

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')
print(os.environ['OPENAI_API_KEY'])

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
            + "_embedding_"
            + self.EMBEDDING_MODEL.split("/")[-1]
        )
        self.documents = self.load_documents(document_directory)
        self.vector_store = self.create_vector_store()
        self.llm = self.initialize_model()
        self.rag_pipeline = self.create_rag_pipeline()

    def create_embeddings(self):
        """Create embeddings for document processing"""
        if self.EMBEDDING_MODEL == "openai":
            return OpenAIEmbeddings()
        else:
            return HuggingFaceEmbeddings(
                model_name=self.EMBEDDING_MODEL,
                model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            )            

    def load_documents(self, directory):
        """Load documents from a directory"""
        
        text_loader_kwargs = {'encoding': 'utf-8'}

        # Support multiple file types
        loader = DirectoryLoader(
            directory,
            glob="**/*",  # Match all files
            loader_cls=TextLoader,
            show_progress=True,
        )
        documents = loader.load()
        print(f"####Loaded {len(documents)} documents from {directory}")
        print(f"Document metadata: {documents[0].metadata}")
        print(f"Document metadata: {documents[0]}")

        # Split documents into chunks
        # text_splitter = RecursiveCharacterTextSplitter(
        #     chunk_size=500, chunk_overlap=100
        # )
        text_splitter = CharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
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

    def initialize_model(self):
        """Initialize model"""

        tokenizer = AutoTokenizer.from_pretrained(self.LLM_MODEL)
        model = AutoModelForCausalLM.from_pretrained(
            self.LLM_MODEL, 
            torch_dtype="auto", #torch.float16, 
            device_map="cuda",
        )
        tokenizer = AutoTokenizer.from_pretrained(self.LLM_MODEL)

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            return_full_text=False
            
            # max_new_tokens=100,
            # top_k=50,
            # temperature=0.1,

        )
        llm = HuggingFacePipeline(pipeline=pipe)

        return llm

    def create_rag_pipeline(self):
        retriever = self.vector_store.as_retriever(
            # search_kwargs={
                # "k": 3,  # Top 3 most relevant documents
                # "search_type": "mmr"  # Maximal Marginal Relevance
            # }
        )
        """Create Retrieval-Augmented Generation pipeline"""
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            # input_key="question",  # Key for user input
            # output_key="answer",  # Key for model response
        )

        # putting it together: set up the conversation chain with the GPT 4o-mini LLM, the vector store and memory
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=memory,
            # return_source_documents=True,
            # response_if_no_docs_found="نمیدانم. لطفا سوال دیگری بپرسید.",
        )
        return conversation_chain

    def chat_with_documents(self, query, history):
        """Process user query and return response"""
        try:
            # Run RAG pipeline
            # result = self.rag_pipeline({"query": query})
            result = self.rag_pipeline.invoke({"question": query})
            print(f"****Result: {result}")

            # Extract answer and sources
            answer = result["answer"]
            # result['result']
            # sources = result["source_documents"]

            # Format source information
            # source_info = "\n\nSources:\n" + "\n".join(
            #     [f"- {doc.metadata.get('source', 'Unknown Source')}" for doc in sources]
            # )

            # Combine answer with source information
            # full_response = answer + source_info

            # return full_response
            return answer

        except Exception as e:
            return f"An error occurred: {str(e)}"

    def launch_gradio_interface(self):
        """Create Gradio interface for document chatbot"""
        iface = gr.ChatInterface(
            fn=self.chat_with_documents,
            title="RAG based Document Chat",
            description="Chat with your documents using Retrieval-Augmented Generation",
            theme="soft",
            # examples=[
            #     "موضوعات اصلی در این مستندها چیست؟",
            #     "نکات مهم را خلاصه سازی کن.",
            #     "مهم ترین اطلاعات را استخراج کن.",
            # ],
            type="messages",
        )

        # Launch the interface
        iface.launch(debug=True)


def main():
    # Specify directory containing documents
    DOCUMENT_DIRECTORY = "./semantic_search_inputs/2"

    # Initialize and launch chatbot
    chatbot = DocumentChatbot(
        document_directory=DOCUMENT_DIRECTORY,
        llm_model="Qwen/Qwen3-4B",
        # "google/gemma-3n-E2B-it",
        # "Qwen/Qwen3-4B",
        embedding_model="Qwen/Qwen3-Embedding-4B"
        # "sentence-transformers/all-MiniLM-L6-v2"
        # "Qwen/Qwen3-Embedding-4B",
        # "Qwen/Qwen3-Embedding-0.6B",
        # "BAAI/bge-m3"
        
    )
    chatbot.launch_gradio_interface()


if __name__ == "__main__":
    main()

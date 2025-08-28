"""
Customer Support Chatbot using Local LLM + RAG + Gradio.

This script builds a Retrieval-Augmented Generation (RAG) application:
1. Upload a PDF document.
2. Ask questions in natural language.
3. The app retrieves relevant chunks from the document and queries a local LLM.
"""
import gradio as gr
import re
import warnings
warnings.filterwarnings('ignore')

# --- LangChain & RAG Components ---
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnableLambda



# Local LLM configuration
from configure_llm import configured_llm
llm = RunnableLambda(lambda x: configured_llm.invoke(x)) # Wrpping with RunnableLambda to make it usable in chaining

# --- Embedding Model (BGE Small) ---
from transformers import AutoTokenizer, AutoModel
import torch
from langchain_core.embeddings import Embeddings

## Load BGE Small
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
model = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5")

# ---- Wrapper for LangChain ----
class HFEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)  # mean pooling
        return embeddings[0].numpy().tolist()

# --- Document Loading & Preprocessing ---
def document_loader(file):
    """Load PDF file into LangChain documents."""
    loader = PyPDFLoader(file.name)
    loaded_document = loader.load()
    return loaded_document


def text_splitter(data):
    """Split long documents into smaller chunks for embedding & retrieval."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 50,
        length_function = len,
    )
    chunks = text_splitter.split_documents(data)
    return chunks

# Vector Store
def vector_database(chunks):
    """Create a Chroma vector database from document chunks."""
    embedding_model = HFEmbeddings()
    vectorDB = Chroma.from_documents(chunks, embedding_model)
    return vectorDB

# Define Retriever
def retriever(file):
    """Convert a PDF file into a retriever for relevant document search."""
    splits = document_loader(file)
    chunks = text_splitter(splits)
    vectordb = vector_database(chunks)
    retriever = vectordb.as_retriever()
    return retriever

# --- Retriever + LLM Question Answering ---
def retriever_qa(file, query):
    retriever_obj = retriever(file)
    
    # Get relavant documents from retriever
    docs = retriever_obj.get_relevant_documents(query)
    context = "\n\n".join([d.page_content for d in docs])
    
    prompt = ChatPromptTemplate.from_template(
        "Answer the question using the context below:\n\n{context}\n\nQuestion: {question}"
    )
    # Retriever → format prompt → LLM
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"context": context, "question" : query})
    response = re.sub(r"\s+", " ", response).strip() # Clean up spaces
    return response

# --- Gradio UI ---
rag_application = gr.Interface(
    fn= retriever_qa,
    allow_flagging="never",
    inputs=[
        gr.File(label="Upload PDF file", file_count="single", file_types=['.pdf'], type="filepath"), # Drag and drop file upload
        gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here...")
    ],
    outputs=gr.Textbox(label='Output'),
    title = 'RAG Application',
    description="Upload a PDF document and ask any question. The chatbot will try to answer using the provided document."
)   

# Launch the interface
rag_application.launch(server_name="127.0.0.1", server_port= 7860)
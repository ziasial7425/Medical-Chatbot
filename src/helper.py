from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Function to load PDF files from a directory
def load_pdf_files(pdf_directory):
    loader = DirectoryLoader(pdf_directory, glob="*.pdf", loader_cls=PyPDFLoader)  # ✅ Use the parameter
    documents = loader.load()
    return documents

# filter to only include page content and metadata
def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """Given a list of documents, filter them to only include the page content and metadata."""
    minimal_docs = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src} if src else {}
            )
        )
    
    return minimal_docs

# split the documents into smaller chunks
def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    texts_chunk = text_splitter.split_documents(minimal_docs)
    return texts_chunk  

# function to download the HuggingFace embeddings model and return the embeddings object
def download_embeddings():
    """Download the HuggingFace embeddings model and return the embeddings object."""
    
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings

embeddings = download_embeddings()
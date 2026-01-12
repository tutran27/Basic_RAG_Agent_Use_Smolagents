from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# 3. Vectorize document
vector_store = Chroma.from_documents(
    documents=splitted_docs, 
    embedding=embeddings, 
    persist_directory="./chroma_database", # Đặt tên thư mục mới
    collection_name="tu_tuong_hcm_384_dim"  # <--- QUAN TRỌNG: Đổi tên collection khác tên cũ
)
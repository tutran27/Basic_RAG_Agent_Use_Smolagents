from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# 1. Load document
loader=PyPDFLoader(r"tu_Tuong_HCM.pdf")
docs=loader.load()

# 2. Split document
text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=128,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)
splitted_docs=text_splitter.split_documents(docs)

# 3. Vectorize document
embedding_model='huyydangg/DEk21_hcmute_embedding'
embeddings=HuggingFaceEmbeddings(model_name=embedding_model,
                                model_kwargs={"device": "cpu"},
                                encode_kwargs={"normalize_embeddings": True})
                                
try:
    vector_store=Chroma.from_documents(splitted_docs, 
                                    embeddings, 
                                    persist_directory=r"chroma_database",
                                    collection_name="tu_tuong_hcm_384_dim"
                                    )
except Exception as e:
    print(f"Lỗi: {str(e)}")
    exit()

print("Vector store đã được tạo thành công")

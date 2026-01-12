from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from smolagents import CodeAgent, InferenceClientModel, Tool
import os

model=InferenceClientModel("Qwen/Qwen3-4B-Instruct-2507", api_key=os.getenv("HF_API_KEY"))

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
vector_store=Chroma.from_documents(splitted_docs, 
                                    embeddings, 
                                    # persist_directory=r"chroma_db"
                                    )

# 4. Create retriever
retriever=vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}
)

# query="Tại sao Hồ Chí Minh lại xem nông nghiệp là mặt trận hàng đầu"
# search=retriever.invoke(query)
# out="\n".join([doc.page_content for doc in search])
# print("----------------- RETRIEVER OUTPUT ------------------")
# print(out)
# print("----------------- END RETRIEVER OUTPUT ------------------")



# ---------------------- Build Agent ----------------------

# Create Tool
class Retriever_Tool(Tool):
    name="Retriever"
    description="Sử dụng tool này, lấy thông tin từ vector store đưa vào agent để trả lời câu hỏi"
    inputs={
        "query": {
            "type": "string",
            "description": "Câu hỏi cần trả lời",
            "language": "vi"
        }
    }
    output_type="string"
    
    def __init__(self, vector_store : Chroma):
        super().__init__()
        self.vector_store=vector_store
    
    def forward(self, query: str) -> str:
        search=self.vector_store.as_retriever().invoke(query, k=2)
        return "\n".join([doc.page_content for doc in search])



# 5. Create agent
agent=CodeAgent(tools=[Retriever_Tool(vector_store)], model=model)

# 6. Run agent
query="Nguyên tắc quan trọng nhất trong xây dựng chỉnh đốn Đảng là gì"

prompt = f"""
Quy tắc:
1) BẮT BUỘC gọi tool retrieve trước khi trả lời.
2) Trả lời tiếng Việt, có trích dẫn dạng [chunk_i] cho mỗi ý chính.
3) Nếu tool trả NO_MATCH thì nói "Không có trong tài liệu".

Câu hỏi: {query}
"""
print(agent.run(prompt))
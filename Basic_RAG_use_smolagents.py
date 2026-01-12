from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from smolagents import CodeAgent, InferenceClientModel, Tool
from dotenv import load_dotenv
import os

load_dotenv()

model=InferenceClientModel("Qwen/Qwen3-4B-Instruct-2507", api_key=os.getenv("HF_API_KEY"))

# 3. Call vector store
embedding_model='huyydangg/DEk21_hcmute_embedding'
embeddings=HuggingFaceEmbeddings(model_name=embedding_model,
                                model_kwargs={"device": "cpu"},
                                encode_kwargs={"normalize_embeddings": True})
vector_store=Chroma(embedding_function=embeddings, 
                    persist_directory=r"chroma_database", 
                    collection_name="tu_tuong_hcm_384_dim"
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
        try:
            search=self.vector_store.as_retriever().invoke(query, k=1)
            if not search:
                return "NO_MATCH"

            # Format kết quả: [chunk_1]: nội dung... [chunk_2]: nội dung...
            formatted_docs = []
            for i, doc in enumerate(search):
                content = doc.page_content.replace("\n", " ") # Xóa xuống dòng thừa
                formatted_docs.append(f"[chunk_{i+1}]: {content}")
            
            # Trả về chuỗi đã format để AI hiểu và trích dẫn
            return "\n\n".join(formatted_docs)

        except Exception as e:
            return f"LỖI: {str(e)}"
        
# 5. Create agent
agent=CodeAgent(tools=[Retriever_Tool(vector_store)], model=model)

# # 6. Run agent
# query="Nguyên tắc quan trọng nhất trong xây dựng chỉnh đốn Đảng là gì"

# prompt = f"""
# Quy tắc:
# 1) BẮT BUỘC gọi tool retrieve trước khi trả lời.
# 2) Trả lời tiếng Việt.
# 3) Nếu tool trả NO_MATCH thì nói "Không có trong tài liệu".

# Câu hỏi: {query}
# """
# print(agent.run(prompt))

# Build function for API
def chat(query: str):
    prompt = f"""
        Quy tắc:
        1) BẮT BUỘC gọi tool retrieve trước khi trả lời.
        2) Trả lời tiếng Việt, có trích dẫn dạng [chunk_i] cho mỗi ý chính.
        3) Nếu tool trả NO_MATCH thì nói "Không có trong tài liệu".

        Câu hỏi: {query}
        """
    return agent.run(prompt)    
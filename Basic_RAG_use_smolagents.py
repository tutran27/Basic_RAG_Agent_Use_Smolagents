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
        
import json
class GroundingTool(Tool):
    

    name="Grounding"
    description="""Kiểm tra xem câu trả lời có bám theo nguồn đã retrieve không
                    Trả JSON String {CHECK: bool, PROBLEMS: [....]}"""
    inputs={
        "answer": {
            "type": "string",
            "description": "Câu trả lời cần kiểm tra",
            "language": "vi"
        },
        "retrieved_docs": {
            "type": "string",
            "description": "Các đoạn văn đã retrieve",
        }
    }
    output_type="string"
    
    def forward(self, answer: str, retrieved_docs: str) -> str:
        problems=[]
        if "[" not in answer or "]" not in answer:
            problems.append("Thiếu citations dạng [chunk_i]")
        elif "NO_MATCH" in retrieved_docs and "Không tìm thấy tài liệu" not in answer:
            problems.append("Không có tài liệu liên quan nhưng answer vẫn cho ra câu trả lời") 
        ok=len(problems)==0
        return json.dumps({"CHECK": ok, "PROBLEMS": problems})

# 5. Create agent
agent=CodeAgent(tools=[Retriever_Tool(vector_store), GroundingTool()], model=model)

question="Mối quan hệ giữa đạo đức và pháp luật trong xây dựng nhà nước dân chủ??"
prompt = f"""
Bạn là trợ lý hỗ trợ nội bộ. Quy tắc BẮT BUỘC:
1) Luôn gọi tool retrieve trước khi trả lời.
2) Trả lời tiếng Việt, gạch đầu dòng, mỗi ý phải có trích dẫn [chunk_i].
3) Không được bịa. Nếu không có thông tin trong tài liệu: trả "Không có trong tài liệu".
4) Sau khi viết draft, luôn gọi GroundingTool(answer, retrieved_docs). Nếu ok=false thì sửa.


Câu hỏi: {question}
"""


output=agent.run(prompt)
print(output)

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
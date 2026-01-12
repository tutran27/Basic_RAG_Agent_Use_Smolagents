from fastapi import FastAPI
from pydantic import BaseModel, Field
# Giả sử file cũ tên là Basic_RAG_use_smolagents.py
from Basic_RAG_use_smolagents import chat 

app = FastAPI(title="HCM Thought RAG API")

# 1. Tách model Request và Response cho rõ ràng
class QueryRequest(BaseModel):
    message: str = Field(..., description="Câu hỏi của người dùng", example="Nguyên tắc xây dựng Đảng là gì?")

class QueryResponse(BaseModel):
    answer: str = Field(..., description="Câu trả lời từ AI")

# 2. Bỏ 'async' ở đây để FastAPI tự động chạy function này trong ThreadPool
# Lý do: agent.run() là blocking code, nếu để async server sẽ bị đơ.
@app.post("/chat", response_model=QueryResponse)
def chat_endpoint(request: QueryRequest):
    # Gọi hàm chat từ logic RAG
    # Lưu ý: Cần đảm bảo hàm chat() bên file kia trả về string
    ai_response = chat(request.message) 
    
    # Trả về đúng định dạng object JSON
    return QueryResponse(answer=ai_response)


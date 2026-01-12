# RAG tiếng Việt với SmolAgents + Chroma + FastAPI

Repo này triển khai một pipeline RAG (Retrieval-Augmented Generation) đơn giản để hỏi–đáp tiếng Việt dựa trên tài liệu PDF.

## Chức năng
- Đọc PDF, chia nhỏ văn bản (chunking)
- Tạo embeddings và lưu vào Chroma (vector database)
- Truy vấn theo ngữ nghĩa (top-k) để lấy đoạn liên quan
- Agent (smolagents) bắt buộc gọi tool retriever trước khi trả lời
- Cung cấp API FastAPI qua endpoint `/chat`

## Cấu trúc dự án
- `vector_store.py`: build Chroma DB từ PDF
- `Basic_RAG_use_smolagents.py`: định nghĩa tool Retriever và hàm `chat()`
- `main.py`: FastAPI server (POST `/chat`)

## Yêu cầu
- Python 3.10+
- Thư viện chính: `fastapi`, `uvicorn`, `chromadb`, `langchain`, `langchain-community`, `smolagents`,
  `transformers`/`sentence-transformers`, `pypdf`

> Tên package có thể thay đổi theo phiên bản LangChain. Nếu gặp lỗi import, hãy kiểm tra version và cập nhật lại requirements.



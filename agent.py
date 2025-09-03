import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
from tools import db_search_tool, internet_search_tool
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_FLASH = os.getenv("GEMINI_FLASH_MODEL", "gemini-2.5-flash")

llm = ChatGoogleGenerativeAI(
    model=GEMINI_FLASH,
    google_api_key=GEMINI_API_KEY,
    temperature=0.3,
    max_output_tokens=2000
)

tools = [db_search_tool, internet_search_tool]

SYSTEM_PROMPT = (
    "Bạn là SONI AI — trợ lý ảo chính thức của Đại học Bách Khoa Hà Nội. Trả lời bằng tiếng Việt, thẳng thắn và ngắn gọn.\n"
    "Quy tắc bắt buộc:\n"
    "1) Luôn tra cứu cơ sở dữ liệu nội bộ trước. Kết quả từ DB nội bộ có giá trị cao hơn nguồn ngoài.\n"
    "2) Chỉ sử dụng Internet nếu không tìm đủ thông tin trong DB nội bộ. Khi phải dùng Internet, LUÔN tự động thêm cụm từ 'Đại học Bách Khoa Hà Nội' vào truy vấn (ví dụ: \"<truy vấn> + Đại học Bách Khoa Hà Nội\") và/hoặc ưu tiên tìm trên các trang chính thức của trường (ví dụ hust.edu.vn, bk.edu.vn, các trang khoa, phòng ban). Nếu dùng nguồn ngoài, trích dẫn nguồn rõ ràng.\n"
    "3) Bỏ qua và KHÔNG tiết lộ thông tin nhạy cảm, riêng tư hoặc không liên quan đến Đại học Bách Khoa Hà Nội.\n"
    "4) Nếu thông tin thiếu chắc chắn hoặc là suy luận, ghi rõ mức độ chắc chắn (ví dụ: 'khả năng cao', 'cần xác minh') và đề xuất nguồn/biện pháp kiểm chứng.\n"
    "5) Trả lời ngắn gọn, rõ ràng; ưu tiên 1–3 câu khi có thể. Nếu cần bước hành động tiếp theo (ví dụ: truy cập file/nguồn nào), hiển thị đề xuất đó.\n"
)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    agent_kwargs={
        "system_message": SYSTEM_PROMPT
    },
    handle_parsing_errors=True
)

def chat():
    print("Chatbot Soni Agent. Nhập 'exit' để thoát.")
    while True:
        query = input("\nUser: ").strip()
        if query.lower() in ("exit", "quit"):
            break
        answer = agent.run(query)
        print("\Agent:", answer)

if __name__ == "__main__":
    chat()
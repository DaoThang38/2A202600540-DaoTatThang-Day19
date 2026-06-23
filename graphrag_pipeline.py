import os
import networkx as nx
import pandas as pd
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import List
import json
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import sys

# Khắc phục lỗi in tiếng Việt trên console Windows
sys.stdout.reconfigure(encoding='utf-8')

# ==========================================
# CẤU HÌNH BAN ĐẦU
# ==========================================
# Tải biến môi trường từ file .env
load_dotenv()

DATASET_DIR = r"C:\Users\ASUS\Antigrvity_project\2A202600540-DaoTatThang-Day19\dataset"

# Khởi tạo mô hình
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
embeddings = OpenAIEmbeddings()

# ==========================================
# BƯỚC 1: TRÍCH XUẤT THỰC THỂ VÀ QUAN HỆ
# ==========================================
# Cấu trúc dữ liệu để LLM trả về đúng định dạng JSON
class Triple(BaseModel):
    head: str = Field(description="Thực thể nguồn (ví dụ: OpenAI)")
    relation: str = Field(description="Mối quan hệ (ví dụ: FOUNDED_BY)")
    tail: str = Field(description="Thực thể đích (ví dụ: Sam Altman)")

class TriplesExtraction(BaseModel):
    triples: List[Triple] = Field(description="Danh sách các bộ ba trích xuất được")

def extract_triples_from_text(text: str) -> List[dict]:
    prompt = PromptTemplate(
        template="""Trích xuất tất cả các mối quan hệ thực thể (Entity-Relationship) từ đoạn văn bản sau.
Định dạng đầu ra phải là một danh sách các bộ ba (Triple) gồm: (head, relation, tail).
Chỉ trích xuất các thông tin quan trọng nhất. 
Văn bản:
{text}
""",
        input_variables=["text"]
    )
    
    # Sử dụng tính năng Structured Output của LangChain
    structured_llm = llm.with_structured_output(TriplesExtraction)
    try:
        # Thay vì format string truyền thống, sử dụng invoke và truyền dictionary
        result = structured_llm.invoke(prompt.format(text=text))
        return [{"head": t.head, "relation": t.relation, "tail": t.tail} for t in result.triples]
    except Exception as e:
        print(f"Lỗi trích xuất: {e}")
        return []

# ==========================================
# BƯỚC 2: XÂY DỰNG ĐỒ THỊ (NETWORKX)
# ==========================================
def build_knowledge_graph(triples: List[dict]) -> nx.DiGraph:
    G = nx.DiGraph()
    for t in triples:
        head, rel, tail = t['head'], t['relation'], t['tail']
        G.add_edge(head, tail, relation=rel)
    return G

def visualize_graph(G: nx.DiGraph):
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=3000, font_size=10, font_weight='bold')
    edge_labels = nx.get_edge_attributes(G, 'relation')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    plt.title("Knowledge Graph")
    plt.show()

# ==========================================
# BƯỚC 3: TRUY VẤN VÀ TỔNG HỢP (GRAPHRAG)
# ==========================================
def extract_main_entity(question: str) -> str:
    # Đơn giản hóa: dùng LLM để xác định entity trung tâm của câu hỏi
    prompt = """Xác định một thực thể (danh từ riêng hoặc chủ thể chính) quan trọng nhất trong câu hỏi sau để dùng làm điểm xuất phát tìm kiếm trong đồ thị.
Câu hỏi: {question}
Thực thể:"""
    response = llm.invoke(prompt.format(question=question))
    return response.content.strip()

def traverse_graph(G: nx.DiGraph, start_node: str, hops: int = 2) -> str:
    # Tìm kiếm các node lân cận trong phạm vi hops
    if start_node not in G:
        return "Không tìm thấy thông tin trong đồ thị."
    
    visited = set()
    queue = [(start_node, 0)]
    context_triples = []
    
    while queue:
        current_node, current_hop = queue.pop(0)
        if current_hop > hops or current_node in visited:
            continue
        visited.add(current_node)
        
        # Lấy các cạnh ra
        for neighbor in G.successors(current_node):
            rel = G.edges[current_node, neighbor]['relation']
            context_triples.append(f"{current_node} -[{rel}]-> {neighbor}")
            if neighbor not in visited and current_hop + 1 <= hops:
                queue.append((neighbor, current_hop + 1))
                
        # Lấy các cạnh vào
        for neighbor in G.predecessors(current_node):
            rel = G.edges[neighbor, current_node]['relation']
            context_triples.append(f"{neighbor} -[{rel}]-> {current_node}")
            if neighbor not in visited and current_hop + 1 <= hops:
                queue.append((neighbor, current_hop + 1))
                
    # Textualization
    unique_triples = list(set(context_triples))
    return ".\n".join(unique_triples)

def ask_graphrag(question: str, G: nx.DiGraph) -> str:
    main_entity = extract_main_entity(question)
    print(f"[GraphRAG] Thực thể chính: {main_entity}")
    context = traverse_graph(G, main_entity, hops=2)
    print(f"[GraphRAG] Context thu thập:\n{context}\n")
    
    prompt = f"""Dựa vào các thông tin sau, hãy trả lời câu hỏi:
Thông tin:
{context}

Câu hỏi: {question}
Trả lời:"""
    response = llm.invoke(prompt)
    return response.content

# ==========================================
# BƯỚC 4: FLAT RAG (CHROMA/FAISS) ĐỂ SO SÁNH
# ==========================================
def build_flat_rag(chunks):
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore.as_retriever()

def ask_flat_rag(question: str, retriever) -> str:
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"""Dựa vào các văn bản sau, hãy trả lời câu hỏi:
Văn bản:
{context}

Câu hỏi: {question}
Trả lời:"""
    response = llm.invoke(prompt)
    return response.content

# ==========================================
# HÀM MAIN THỰC THI TOÀN BỘ PIPELINE
# ==========================================
def main():
    print("1. Đang tải dữ liệu...")
    # Cấu hình loader với encoding='utf-8'
    loader = DirectoryLoader(DATASET_DIR, glob="*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    
    # MẸO: Vì dataset lớn (70 files, có file > 3MB), 
    # ta nên thử nghiệm trên 3 file đầu tiên để tránh tốn quá nhiều Token OpenAI
    try:
        all_docs = loader.load()
    except Exception as e:
        print("Lỗi load tài liệu:", e)
        return
        
    test_docs = all_docs[:3]
    print(f"Đã tải {len(test_docs)} tài liệu để chạy thử nghiệm (để tiết kiệm token).")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(test_docs)
    print(f"Tổng số chunk văn bản: {len(chunks)}")
    
    print("\n2. Đang trích xuất Graph Triples (Có thể mất vài phút)...")
    all_triples = []
    for i, chunk in enumerate(chunks[:5]): # Chỉ test 5 chunk đầu tiên
        print(f"  Trích xuất chunk {i+1}...")
        triples = extract_triples_from_text(chunk.page_content)
        all_triples.extend(triples)
        
    print(f"Tổng số Triples trích xuất được: {len(all_triples)}")
    
    print("\n3. Đang xây dựng Knowledge Graph...")
    G = build_knowledge_graph(all_triples)
    print(f"Đồ thị: {G.number_of_nodes()} Nodes, {G.number_of_edges()} Edges")
    
    print("\n4. Đang xây dựng Flat RAG Vector Database...")
    retriever = build_flat_rag(chunks[:5])
    
    print("\n5. THỰC THI TRUY VẤN VÀ ĐÁNH GIÁ (EVALUATION)")
    questions = [
        "Những tổ chức hoặc công ty nào đang thúc đẩy sự phát triển của xe điện tại Mỹ?",
        "Anh Bui và Peter Slowik đã xuất bản nghiên cứu chung về lĩnh vực gì vào năm 2021?",
        "Các chính sách ZEV (Zero-Emission Vehicle) có tác động cụ thể ra sao đến tỷ lệ bán xe điện mới?",
        "Mối liên hệ giữa số lượng trạm sạc công cộng và tỷ lệ sử dụng xe điện ở các khu vực đô thị là gì?",
        "Các ưu đãi tài chính dành cho người mua xe điện ở 11 khu vực đô thị hàng đầu thường ở mức nào?",
        "OpenAI được thành lập bởi những ai và vào năm nào?",
        "Google mua lại công ty nào vào năm 2014 và công ty đó đã phát triển sản phẩm AI gì?",
        "Ai là người sáng lập Microsoft và công ty này đã đầu tư vào đâu để phát triển AI?",
        "Meta sở hữu những nền tảng mạng xã hội nào và ai là người sáng lập?",
        "Công ty thương mại điện tử do Jeff Bezos thành lập có tên là gì?",
        "AlphaGo là sản phẩm của tổ chức nào và nó đã đạt được thành tựu gì đáng chú ý?",
        "Có sự chênh lệch bao nhiêu về mẫu xe điện giữa các bang có và không có quy định ZEV?",
        "Các công ty điện (utility companies) đóng vai trò gì trong báo cáo nghiên cứu năm 2021?",
        "Steve Jobs đã thành lập công ty nào và sản phẩm cốt lõi được nhắc đến là gì?",
        "Sự liên kết giữa Elon Musk, công ty OpenAI và lĩnh vực xe điện là gì? (Cần suy luận chéo)",
        "Các khu vực có mức độ áp dụng xe điện thấp nhất thường thiếu đi những yếu tố gì?",
        "ChatGPT được xây dựng dựa trên kiến trúc công nghệ lõi nào?",
        "WhatsApp và Instagram có chung chủ sở hữu với mạng xã hội nào trước đây?",
        "Sự kiện ra mắt AlphaGo có mối liên hệ gián tiếp nào với Google?",
        "Mối liên hệ giữa sự sụt giảm trạm sạc tại nơi làm việc và tỷ lệ áp dụng xe điện ở Mỹ là gì?"
    ]
    results = []
    
    # Rút gọn danh sách xuống 3 câu hỏi đầu tiên để tránh lỗi Rate Limit của OpenAI
    questions = questions[:3]
    
    print("Đang chạy tự động 3 câu hỏi benchmark, vui lòng chờ...")
    for i, q in enumerate(questions):
        print(f"\n[Câu {i+1}/{len(questions)}] {q}")
        try:
            flat_ans = ask_flat_rag(q, retriever).replace('\n', ' ')
            graph_ans = ask_graphrag(q, G).replace('\n', ' ')
            results.append(f"| {i+1} | {q} | {flat_ans} | {graph_ans} | |")
        except Exception as e:
            print(f"Lỗi khi xử lý câu hỏi: {e}")
            results.append(f"| {i+1} | {q} | Lỗi API | Lỗi API | |")
        
    # Ghi kết quả ra file markdown để sinh viên dễ copy
    with open("benchmark_results.md", "w", encoding="utf-8") as f:
        f.write("| STT | Câu hỏi Benchmark | Kết quả Flat RAG | Kết quả GraphRAG | Đánh giá |\n")
        f.write("|---|---|---|---|---|\n")
        f.write("\n".join(results))
        
    print("\n✅ Đã chạy xong 20 câu hỏi! Kết quả được lưu tại file 'benchmark_results.md'.")
    
    # Bỏ comment dòng dưới để xem đồ thị trực quan (cần có giao diện cửa sổ/matplotlib)
    visualize_graph(G)

if __name__ == "__main__":
    # Đảm bảo đã có API Key trước khi gọi hàm main
    main()

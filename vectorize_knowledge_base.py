"""
RAG 向量化脚本 - Qwen 全家桶版本
将 Zino's Petrel 文献库向量化并存储到 ChromaDB

使用方法:
    python vectorize_knowledge_base.py

功能:
    - 批量处理 PDF 文件
    - 优化的文档分割（chunk_overlap=200）
    - Qwen Embedding (text-embedding-v3)
    - 进度追踪和错误处理
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma

# 加载环境变量
load_dotenv()

# 配置
PDF_FOLDER = "Mediterranean Monk Seal"
VECTOR_DB_PATH = "db7_qwen"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200  # 20% 重叠，保持上下文连续性
# 从环境变量读取 Embedding 模型（与 rag_utils.py 保持一致）
EMBEDDING_MODEL = os.getenv("QWEN_EMBEDDING_MODEL", "text-embedding-v3")

def get_pdf_files(folder_path):
    """获取文件夹中所有 PDF 文件"""
    pdf_path = Path(folder_path)
    if not pdf_path.exists():
        print(f"❌ 错误: 文件夹 '{folder_path}' 不存在")
        sys.exit(1)
    
    pdf_files = list(pdf_path.glob("*.pdf"))
    if not pdf_files:
        print(f"⚠️  警告: 文件夹 '{folder_path}' 中没有 PDF 文件")
        sys.exit(1)
    
    return pdf_files

def load_and_split_pdf(pdf_path, text_splitter):
    """加载并分割单个 PDF 文件"""
    try:
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        
        # 为每个文档添加元数据
        for i, page in enumerate(pages):
            page.metadata.update({
                "source_file": pdf_path.name,
                "page": i + 1,
                "total_pages": len(pages)
            })
        
        # 分割文档
        chunks = text_splitter.split_documents(pages)
        return chunks, None
    
    except Exception as e:
        return None, str(e)

def vectorize_documents(pdf_files, embeddings, text_splitter):
    """向量化所有文档"""
    all_chunks = []
    failed_files = []
    
    print(f"\n📚 开始处理 {len(pdf_files)} 个 PDF 文件...\n")
    
    # 使用 tqdm 显示进度
    for pdf_file in tqdm(pdf_files, desc="处理 PDF", unit="文件"):
        chunks, error = load_and_split_pdf(pdf_file, text_splitter)
        
        if error:
            failed_files.append((pdf_file.name, error))
            tqdm.write(f"❌ 失败: {pdf_file.name} - {error}")
        else:
            all_chunks.extend(chunks)
            tqdm.write(f"✅ 成功: {pdf_file.name} ({len(chunks)} chunks)")
    
    print(f"\n📊 统计:")
    print(f"  - 成功: {len(pdf_files) - len(failed_files)} 个文件")
    print(f"  - 失败: {len(failed_files)} 个文件")
    print(f"  - 总块数: {len(all_chunks)} chunks")
    
    if failed_files:
        print(f"\n⚠️  失败文件列表:")
        for filename, error in failed_files:
            print(f"  - {filename}: {error}")
    
    return all_chunks

def create_vector_store(chunks, embeddings, persist_directory):
    """创建并持久化向量数据库"""
    print(f"\n🔄 创建向量数据库...")
    print(f"  - 向量库路径: {persist_directory}")
    print(f"  - 嵌入模型: {EMBEDDING_MODEL}")
    print(f"  - 文档块数量: {len(chunks)}")
    
    try:
        # 清空旧数据库（如果存在）
        if Path(persist_directory).exists():
            import shutil
            shutil.rmtree(persist_directory)
            print(f"  - 已清空旧数据库")
        
        # 分批处理向量化（DashScope 限制：batch_size ≤ 10）
        batch_size = 10
        vectordb = None
        
        for i in tqdm(range(0, len(chunks), batch_size), desc="向量化", unit="批次"):
            batch = chunks[i:i + batch_size]
            
            if vectordb is None:
                # 首次创建
                vectordb = Chroma.from_documents(
                    documents=batch,
                    embedding=embeddings,
                    persist_directory=persist_directory,
                    collection_name="zinos_petrel_knowledge"
                )
            else:
                # 追加到现有数据库
                vectordb.add_documents(batch)
        
        print(f"\n✅ 向量数据库创建成功!")
        return vectordb
    
    except Exception as e:
        print(f"\n❌ 向量数据库创建失败: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def test_retrieval(vectordb):
    """测试检索功能"""
    print(f"\n🧪 测试检索功能...")
    
    test_queries = [
        "What is Zino's Petrel?",
        "Where does Zino's Petrel nest?",
        "What does Zino's Petrel eat?"
    ]
    
    for query in test_queries:
        print(f"\n📝 查询: '{query}'")
        results = vectordb.similarity_search(query, k=2)
        
        for i, doc in enumerate(results, 1):
            print(f"\n  结果 {i}:")
            print(f"    - 来源: {doc.metadata.get('source_file', 'Unknown')}")
            print(f"    - 页码: {doc.metadata.get('page', 'N/A')}")
            print(f"    - 内容预览: {doc.page_content[:150]}...")

def main():
    """主函数"""
    print("=" * 60)
    print("📚 RAG 向量化脚本 - Qwen 全家桶版本")
    print("=" * 60)
    
    # 1. 检查 API Key
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("❌ 错误: 未找到 DASHSCOPE_API_KEY")
        print("请在 .env 文件中配置 API Key")
        sys.exit(1)
    
    print(f"✅ API Key 已配置")
    
    # 2. 获取 PDF 文件列表
    pdf_files = get_pdf_files(PDF_FOLDER)
    print(f"✅ 找到 {len(pdf_files)} 个 PDF 文件")
    
    # 3. 初始化 Embeddings
    print(f"\n🔧 初始化 Embedding 模型...")
    embeddings = DashScopeEmbeddings(
        model=EMBEDDING_MODEL,
        dashscope_api_key=api_key
    )
    print(f"✅ 使用模型: {EMBEDDING_MODEL}")
    
    # 4. 初始化文本分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    print(f"✅ 文本分割配置: chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
    
    # 5. 向量化文档
    chunks = vectorize_documents(pdf_files, embeddings, text_splitter)
    
    if not chunks:
        print("❌ 没有成功处理任何文档")
        sys.exit(1)
    
    # 6. 创建向量数据库
    vectordb = create_vector_store(chunks, embeddings, VECTOR_DB_PATH)
    
    # 7. 测试检索
    test_retrieval(vectordb)
    
    print("\n" + "=" * 60)
    print("🎉 向量化完成!")
    print("=" * 60)
    print(f"\n📁 向量库位置: {VECTOR_DB_PATH}")
    print(f"📊 总文档块数: {len(chunks)}")
    print(f"\n下一步: 运行 'streamlit run main.py' 开始使用!")

if __name__ == "__main__":
    main()


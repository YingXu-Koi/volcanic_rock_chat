"""
RAG 质量测试脚本
验证向量化结果和检索性能

使用方法:
    python test_rag_quality.py
"""

import os
import time
from dotenv import load_dotenv
from rag_utils import get_rag_instance

# 加载环境变量
load_dotenv()

def test_vectordb_stats():
    """测试向量库统计信息"""
    print("=" * 60)
    print("📊 向量库统计信息")
    print("=" * 60)
    
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("❌ 错误: 未找到 DASHSCOPE_API_KEY")
        return
    
    rag = get_rag_instance("db5_qwen", api_key)
    stats = rag.get_stats()
    
    print(f"\n✅ 向量库路径: {stats['persist_directory']}")
    print(f"✅ 嵌入模型: {stats['embedding_model']}")
    print(f"✅ 文档数量: {stats['total_documents']}")
    print()

def test_retrieval_quality(lambda_mult=0.3):
    """测试检索质量 - 基础场景"""
    print("=" * 60)
    print(f"🧪 检索质量测试 - 基础场景 (lambda_mult={lambda_mult})")
    print("=" * 60)
    
    api_key = os.getenv("DASHSCOPE_API_KEY")
    rag = get_rag_instance("db5_qwen", api_key)
    
    # 测试查询列表
    test_queries = [
        {
            "query": "What is Zino's Petrel?",
            "expected_keywords": ["petrel", "bird", "seabird", "Pterodroma"],
            "complexity": "simple"
        },
        {
            "query": "Where does Zino's Petrel nest and what is its habitat?",
            "expected_keywords": ["nest", "habitat", "mountains", "Madeira"],
            "complexity": "medium"
        },
        {
            "query": "Describe the conservation status and main threats to Zino's Petrel, and what actions are being taken to protect the species?",
            "expected_keywords": ["conservation", "endangered", "threats", "protection"],
            "complexity": "complex"
        }
    ]
    
    for i, test in enumerate(test_queries, 1):
        print(f"\n{'=' * 60}")
        print(f"测试 {i}: {test['complexity'].upper()} 查询")
        print(f"{'=' * 60}")
        print(f"📝 查询: '{test['query']}'")
        print(f"🎯 预期关键词: {', '.join(test['expected_keywords'])}")
        
        # 计时
        start_time = time.time()
        docs = rag.retrieve(test['query'], lambda_mult=lambda_mult)
        elapsed_time = time.time() - start_time
        
        print(f"\n⏱️  检索耗时: {elapsed_time:.3f} 秒")
        print(f"📄 返回文档数: {len(docs)}")
        
        # 检查关键词覆盖
        all_content = " ".join([doc.page_content.lower() for doc in docs])
        found_keywords = [kw for kw in test['expected_keywords'] if kw.lower() in all_content]
        coverage = len(found_keywords) / len(test['expected_keywords']) * 100
        
        print(f"✅ 关键词覆盖率: {coverage:.1f}% ({len(found_keywords)}/{len(test['expected_keywords'])})")
        print(f"   找到: {', '.join(found_keywords) if found_keywords else '无'}")
        
        # 显示文档来源
        print(f"\n📚 文档来源:")
        for j, doc in enumerate(docs, 1):
            source = doc.metadata.get('source_file', 'Unknown')
            page = doc.metadata.get('page', 'N/A')
            preview = doc.page_content[:100].replace('\n', ' ')
            print(f"   {j}. {source} (页 {page})")
            print(f"      预览: {preview}...")
        
        # 质量评估
        if coverage >= 75:
            print(f"\n✅ 质量评估: 优秀（覆盖率 ≥75%）")
        elif coverage >= 50:
            print(f"\n⚠️  质量评估: 良好（覆盖率 ≥50%）")
        else:
            print(f"\n❌ 质量评估: 需改进（覆盖率 <50%）")

def test_user_scenarios(lambda_mult=0.3):
    """测试用户实际场景"""
    print("\n" + "=" * 60)
    print(f"👥 用户实际场景测试 (lambda_mult={lambda_mult})")
    print("=" * 60)
    print("模拟真实用户对话，测试 RAG 系统的实际表现")
    print()
    
    api_key = os.getenv("DASHSCOPE_API_KEY")
    rag = get_rag_instance("db5_qwen", api_key)
    
    # 用户实际测试问题
    user_tests = [
        {
            "id": 1,
            "query": "Hi, how are you doing today?",
            "category": "问候",
            "expected_keywords": ["petrel", "bird", "fine", "good"],
            "expected_sticker": None,
            "expected_score_change": "+1 (empathy)"
        },
        {
            "id": 2,
            "query": "Where do you usually have your nesting areas?",
            "category": "栖息地",
            "expected_keywords": ["nest", "Madeira", "mountains", "cliffs", "caves"],
            "expected_sticker": "🏡 Home",
            "expected_score_change": "+1 (knowledge)"
        },
        {
            "id": 3,
            "query": "How long do you live approximately?",
            "category": "寿命",
            "expected_keywords": ["years", "lifespan", "live", "age"],
            "expected_sticker": None,
            "expected_score_change": "+1 (deep_interaction)"
        },
        {
            "id": 4,
            "query": "Why do you need to abort sometimes to protect your species, that's a very sad thing and I don't quite understand how does it help you",
            "category": "保护策略",
            "expected_keywords": ["conservation", "protection", "breeding", "survival", "predators"],
            "expected_sticker": "🌱 Helper (可能)",
            "expected_score_change": "+1 (conservation_action/empathy)"
        },
        {
            "id": 5,
            "query": "How long do you sleep?",
            "category": "日常习惯",
            "expected_keywords": ["sleep", "rest", "night", "day", "active"],
            "expected_sticker": "🌙 Routine (可能)",
            "expected_score_change": "+1 (knowledge)"
        },
        {
            "id": 6,
            "query": "How do I find you?",
            "category": "观察指南",
            "expected_keywords": ["Madeira", "mountains", "sea", "observation", "location"],
            "expected_sticker": "🏡 Home (如未触发)",
            "expected_score_change": "+1 (personal_engagement)"
        },
        {
            "id": 7,
            "query": "Do you have a friend?",
            "category": "社交",
            "expected_keywords": ["mate", "colony", "pair", "social", "alone"],
            "expected_sticker": None,
            "expected_score_change": "+1 (personal_engagement)"
        },
        {
            "id": 8,
            "query": "What do you eat for food and how do you catch it?",
            "category": "饮食",
            "expected_keywords": ["fish", "squid", "food", "catch", "hunt", "sea"],
            "expected_sticker": "🍽️ Food",
            "expected_score_change": "+1 (knowledge)"
        },
        {
            "id": 9,
            "query": "How can I help you and your species thrive?",
            "category": "保护行动",
            "expected_keywords": ["help", "protect", "conservation", "support", "habitat"],
            "expected_sticker": "🌱 Helper",
            "expected_score_change": "+1 (conservation_action)"
        }
    ]
    
    total_coverage = 0
    successful_tests = 0
    
    for test in user_tests:
        print(f"\n{'=' * 60}")
        print(f"测试 {test['id']}: {test['category']} - {test['expected_sticker'] or '无贴纸'}")
        print(f"{'=' * 60}")
        print(f"📝 问题: '{test['query']}'")
        print(f"🎯 预期关键词: {', '.join(test['expected_keywords'])}")
        print(f"🎁 预期贴纸: {test['expected_sticker'] or '无'}")
        print(f"❤️  预期评分: {test['expected_score_change']}")
        
        # 计时
        start_time = time.time()
        docs = rag.retrieve(test['query'], lambda_mult=lambda_mult)
        elapsed_time = time.time() - start_time
        
        print(f"\n⏱️  检索耗时: {elapsed_time:.3f} 秒")
        print(f"📄 返回文档数: {len(docs)}")
        
        # 检查关键词覆盖
        all_content = " ".join([doc.page_content.lower() for doc in docs])
        found_keywords = [kw for kw in test['expected_keywords'] if kw.lower() in all_content]
        coverage = len(found_keywords) / len(test['expected_keywords']) * 100 if test['expected_keywords'] else 0
        total_coverage += coverage
        
        print(f"✅ 关键词覆盖率: {coverage:.1f}% ({len(found_keywords)}/{len(test['expected_keywords'])})")
        if found_keywords:
            print(f"   找到: {', '.join(found_keywords)}")
        else:
            print(f"   找到: 无")
        
        # 显示文档来源（最多显示 2 个）
        print(f"\n📚 文档来源:")
        for i, doc in enumerate(docs[:2], 1):
            source = doc.metadata.get('source_file', 'Unknown')
            page = doc.metadata.get('page', 'N/A')
            preview = doc.page_content[:80].replace('\n', ' ')
            print(f"   {i}. {source} (页 {page})")
            print(f"      预览: {preview}...")
        
        # 质量评估
        if coverage >= 60:
            print(f"\n✅ 检索质量: 优秀（覆盖率 ≥60%）")
            successful_tests += 1
        elif coverage >= 40:
            print(f"\n⚠️  检索质量: 良好（覆盖率 ≥40%）")
            successful_tests += 1
        else:
            print(f"\n❌ 检索质量: 需改进（覆盖率 <40%）")
    
    # 总结
    print(f"\n{'=' * 60}")
    print(f"📊 测试总结")
    print(f"{'=' * 60}")
    print(f"✅ 成功测试: {successful_tests}/{len(user_tests)} ({successful_tests/len(user_tests)*100:.1f}%)")
    print(f"📈 平均关键词覆盖率: {total_coverage/len(user_tests):.1f}%")
    
    if successful_tests >= 7:
        print(f"\n🎉 整体评估: 优秀！RAG 系统表现出色")
    elif successful_tests >= 5:
        print(f"\n👍 整体评估: 良好，基本满足需求")
    else:
        print(f"\n⚠️  整体评估: 需要优化，建议调整检索参数")

def test_performance():
    """测试性能（缓存效果）"""
    print("\n" + "=" * 60)
    print("⚡ 性能测试（缓存效果）")
    print("=" * 60)
    
    api_key = os.getenv("DASHSCOPE_API_KEY")
    test_query = "What is Zino's Petrel?"
    
    # 首次查询（冷启动）
    print(f"\n🔵 首次查询（冷启动）...")
    start_time = time.time()
    rag1 = get_rag_instance("db7_qwen", api_key)
    docs1 = rag1.retrieve(test_query)
    cold_time = time.time() - start_time
    print(f"   ⏱️  耗时: {cold_time:.3f} 秒")
    
    # 第二次查询（缓存命中）
    print(f"\n🟢 第二次查询（缓存命中）...")
    start_time = time.time()
    rag2 = get_rag_instance("db5_qwen", api_key)
    docs2 = rag2.retrieve(test_query)
    hot_time = time.time() - start_time
    print(f"   ⏱️  耗时: {hot_time:.3f} 秒")
    
    # 性能提升
    speedup = cold_time / hot_time if hot_time > 0 else float('inf')
    print(f"\n📊 性能提升: {speedup:.1f}x")
    print(f"   🔹 冷启动: {cold_time:.3f} 秒")
    print(f"   🔹 缓存命中: {hot_time:.3f} 秒")

def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("🧪 RAG 质量测试套件")
    print("=" * 60)
    print()
    
    # 1. 统计信息
    test_vectordb_stats()
    
    # 2. 基础检索质量测试
    test_retrieval_quality()
    
    # 3. 用户实际场景测试（新增）
    test_user_scenarios()
    
    # 4. 性能测试
    test_performance()
    
    print("\n" + "=" * 60)
    print("✅ 测试完成!")
    print("=" * 60)
    print("\n💡 提示:")
    print("   - 如果关键词覆盖率 <40%，建议调整 lambda_mult 参数")
    print("   - 如果检索速度 >3秒，检查网络连接或 API 配额")
    print("   - 运行 'streamlit run main.py' 进行实际测试")
    print()

if __name__ == "__main__":
    main()


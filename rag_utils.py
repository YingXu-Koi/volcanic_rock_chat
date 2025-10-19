"""
RAG 检索优化工具
包含向量库缓存、动态 k 值调整、相关性过滤等优化策略
"""

import os
from functools import lru_cache
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma

class OptimizedRAG:
    """优化的 RAG 检索器"""
    
    def __init__(self, persist_directory, dashscope_api_key):
        self.persist_directory = persist_directory
        self.dashscope_api_key = dashscope_api_key
        self._vectordb = None
        
    @property
    def vectordb(self):
        """延迟加载并缓存向量数据库"""
        if self._vectordb is None:
            print(f"[RAG] 加载向量数据库: {self.persist_directory}")
            embeddings = DashScopeEmbeddings(
                model=os.getenv("QWEN_EMBEDDING_MODEL", "text-embedding-v3"),
                dashscope_api_key=self.dashscope_api_key
            )
            self._vectordb = Chroma(
                embedding_function=embeddings,
                persist_directory=self.persist_directory,
                collection_name="volcanic_rock_knowledge"  # 与向量化脚本保持一致
            )
            print(f"[RAG] ✅ 向量数据库已加载")
        return self._vectordb
    
    def retrieve(self, query, k=None, fetch_k=None, lambda_mult=0.7, 
                 relevance_threshold=None):
        """
        智能检索文档
        
        Args:
            query: 查询文本
            k: 返回文档数量（None 则自动调整）
            fetch_k: MMR 候选池大小（None 则自动调整）
            lambda_mult: MMR 多样性参数（0-1，越大越相关，越小越多样）
            relevance_threshold: 相关性阈值（0-1，过滤低质量文档）
        
        Returns:
            list: 检索到的文档列表
        """
        # 动态调整 k 值（基于查询复杂度）
        if k is None:
            k = self._estimate_k(query)
        
        # 动态调整 fetch_k
        if fetch_k is None:
            fetch_k = k * 3  # 候选池是返回数量的3倍
        
        print(f"[RAG] 检索参数: k={k}, fetch_k={fetch_k}, lambda_mult={lambda_mult}")
        
        # 使用 MMR（最大边际相关性）检索
        docs = self.vectordb.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult
        )
        
        # 相关性过滤（如果设置了阈值）
        if relevance_threshold is not None:
            filtered_docs = self._filter_by_relevance(
                query, docs, threshold=relevance_threshold
            )
            print(f"[RAG] 相关性过滤: {len(docs)} -> {len(filtered_docs)} 文档")
            return filtered_docs
        
        return docs
    
    def _estimate_k(self, query):
        """
        基于查询复杂度估算 k 值
        
        简单启发式规则：
        - 短问题（<20词）: k=2
        - 中等问题（20-50词）: k=3
        - 复杂问题（>50词）: k=4
        """
        word_count = len(query.split())
        
        if word_count < 20:
            return 2
        elif word_count < 50:
            return 3
        else:
            return 4
    
    def _filter_by_relevance(self, query, docs, threshold=0.6):
        """
        基于相关性分数过滤文档
        
        Args:
            query: 查询文本
            docs: 文档列表
            threshold: 相关性阈值（0-1）
        
        Returns:
            list: 过滤后的文档
        """
        # 使用 similarity_search_with_score 获取相关性分数
        docs_with_scores = self.vectordb.similarity_search_with_score(
            query, 
            k=len(docs)
        )
        
        # 过滤低于阈值的文档
        # 注意：ChromaDB 的距离越小越相关（L2距离），需要转换为相似度
        filtered = [
            doc for doc, score in docs_with_scores 
            if score < (1 - threshold)  # 距离阈值转换
        ]
        
        return filtered if filtered else docs[:1]  # 至少返回1个文档
    
    def get_stats(self):
        """获取向量库统计信息"""
        collection = self.vectordb._collection
        count = collection.count()
        return {
            "total_documents": count,
            "persist_directory": self.persist_directory,
            "embedding_model": os.getenv("QWEN_EMBEDDING_MODEL", "text-embedding-v3")
        }


# 全局 RAG 实例缓存（避免重复加载）
_rag_instances = {}

def get_rag_instance(persist_directory, dashscope_api_key):
    """
    获取 RAG 实例（带缓存）
    
    Args:
        persist_directory: 向量库路径
        dashscope_api_key: DashScope API Key
    
    Returns:
        OptimizedRAG: RAG 实例
    """
    if persist_directory not in _rag_instances:
        _rag_instances[persist_directory] = OptimizedRAG(
            persist_directory, 
            dashscope_api_key
        )
    return _rag_instances[persist_directory]


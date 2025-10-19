"""
Fact-Check 工具 - 智能摘要和验证
结合知识库检索和可选的网络搜索，生成总结性文本
"""

import os
from langchain_community.llms import Tongyi
from dotenv import load_dotenv

load_dotenv()

def get_friendly_filename(source_file):
    """
    Convert technical source file names to user-friendly names
    """
    filename_mapping = {
        # Your Excel mappings
    'Rohiman_2019_J._Phys.__Conf._Ser._1204_012070.pdf': 'Rohiman_et_al_2019_J._Phys._Conf._Ser._Geochemical_Characteristics_of_Volcanic_Rocks_from_Mt_Masurai_Caldera_Indonesia.pdf',
    'mineralization-study-of-volcanic-rocks-in-colo-volcano-tojo-12p5xwo6nl.pdf': 'Asrafil_et_al_2020_Jurnal_Geomine_Mineralization_Study_of_Volcanic_Rocks_in_Colo_Volcano_Indonesia.pdf',
    'e3sconf_icosmed2023_01011.pdf': 'Permana_et_al_2023_E3S_Web_of_Conferences_Characteristics_of_Volcanic_Rock_In_The_Bualemo_Area_North_Gorontalo_Indonesia.pdf',
    '978-3-642-74864-6_2.pdf': 'Fisher_et_al_1984_Pyroclastic_Rocks_Chapter_2_Volcanoes_Volcanic_Rocks_and_Magma_Chambers.pdf',
    'ngm2016_article_Abstr_no_137revised.pdf': 'Foged_et_al_2016_Proceedings_17th_Nordic_Geotechnical_Meeting_Strength_and_deformation_properties_of_volcanic_rocks_in_Iceland.pdf',
    '1-s2.0-S2949736125000065-main.pdf': 'Yang_et_al_2025_Volcanic_rocks_in_the_21st_century_Multifaceted_applications_for_sustainable_development.pdf',
    'Navelot_JVGR_2018-CorrectionV4.pdf': 'Navelot_et_al_2018_Petrophysical_properties_of_volcanic_rocks_and_impacts_of_hydrothermal_alteration_in_the_Guadeloupe_Archipelago.pdf',
    '2018-JVGR-complex conductivity volcanoes.pdf': 'Ghorbani_et_al_2018_Complex_conductivity_of_volcanic_rocks_and_the_geophysical_mapping_of_alteration_in_volcanoes.pdf',
    'ca19527404c0dd71b7d37e10339b3ab09862.pdf': 'Shaaban_et_al_2020_Classification_of_Volcanic_Rocks_based_on_Rough_Set_Theory.pdf',
    'geokniga-volcanic-textures-guide-interpretation-textures-volcanic-rocks.pdf': 'McPhie_et_al_1993_Volcanic_Textures_A_Guide_to_the_Interpretation_of_Textures_in_Volcanic_Rocks.pdf',
    'Week10-ilesaintehelene.pdf': 'Nov_2018_EPSC_240_Classification_and_Texture_of_Volcanic_Rocks.pdf',
    '289.pdf': 'Middlemost_1972_A_Simple_Chemical_Classification_of_Volcanic_Rocks.pdf',
    'The_Life_of_Volcanic_Rocks_During_and_After_an_Eru.pdf': 'Brennan_et_al_2021_The_Life_of_Volcanic_Rocks_During_and_After_an_Eruption.pdf',
        
        # Default fallback
        'unknown': 'Unknown Document'
    }
    
    base_name = os.path.basename(source_file) if source_file else 'unknown'
    return filename_mapping.get(base_name, base_name.replace('_', ' ').replace('-', ' ').title())


def summarize_fact_check(question, retrieved_docs, ai_answer, language="English"):
    """
    对 Fact-Check 内容进行智能摘要
    
    Args:
        question: 用户问题
        retrieved_docs: 检索到的文档列表
        ai_answer: AI 的回答
        language: 语言（English/Portuguese）
    
    Returns:
        str: 总结性文本
    """
    # 提取文档内容
    doc_contents = []
    sources = []
    
    for i, doc in enumerate(retrieved_docs[:3], 1):  # 最多使用3个文档
        content = doc.page_content[:500]  # 每个文档最多500字符
        source = doc.metadata.get('source_file', 'Unknown')
        page = doc.metadata.get('page', 'N/A')

        friendly_name = get_friendly_filename(source)
        
        doc_contents.append(f"[Source {i}: {friendly_name}, Page {page}]\n{content}")
        sources.append(f"{friendly_name} (p.{page})")
    
    combined_docs = "\n\n".join(doc_contents)
    
    # 构建摘要 Prompt
    if language == "Portuguese":
        prompt = f"""
        Tu és um verificador de factos científico. Com base nos documentos fornecidos, cria um resumo claro e conciso.

        **Pergunta do utilizador:** {question}

        **Resposta da IA:** {ai_answer}

        **Documentos de referência:**
        {combined_docs}

        **Tua tarefa:**
        1. Resume os pontos-chave dos documentos que apoiam a resposta
        2. Menciona dados específicos (números, locais, datas) se disponíveis
        3. Mantém o resumo abaixo de 100 palavras
        4. Usa linguagem simples e clara
        5. Se os documentos não apoiam a resposta, indica isso

        **Resumo factual:**
        """
    else:
        prompt = f"""
        You are a scientific fact-checker. Based on the provided documents, create a clear and concise summary.

        **User's Question:** {question}

        **AI's Answer:** {ai_answer}

        **Reference Documents:**
        {combined_docs}

        **Your Task:**
        1. Summarize key points from the documents that support the answer
        2. Mention specific data (numbers, locations, dates) if available
        3. Keep the summary under 100 words
        4. Use simple, clear language
        5. If documents don't support the answer, indicate that

        **Factual Summary:**
        """
    
    # 使用 Qwen LLM 生成摘要
    try:
        api_key = os.getenv("DASHSCOPE_API_KEY")
        llm = Tongyi(
            model_name=os.getenv("QWEN_MODEL_NAME", "qwen-turbo"),
            temperature=0.3,  # 较低温度，确保事实性
            dashscope_api_key=api_key
        )
        
        summary = llm.invoke(prompt)
        
        # 添加来源引用
        if language == "Portuguese":
            source_text = f"\n\n📚 **Fontes:** {', '.join(sources)}"
        else:
            source_text = f"\n\n📚 **Sources:** {', '.join(sources)}"
        
        return summary.strip() + source_text
    
    except Exception as e:
        print(f"[Fact-Check] 摘要生成失败: {str(e)}")
        # 降级：返回简化的文档内容
        source = retrieved_docs[0].metadata.get('source_file', 'Unknown')
        page = retrieved_docs[0].metadata.get('page', 'N/A')
        friendly_name = get_friendly_filename(source)
        
        if language == "Portuguese":
            return f"📄 Informação extraída dos documentos:\n\n{retrieved_docs[0].page_content[:200]}...\n\n📚 Fonte: {friendly_name} (p.{page})"
        else:
            return f"📄 Information from documents:\n\n{retrieved_docs[0].page_content[:200]}...\n\n📚 Source: {friendly_name} (p.{page})"


def optimize_search_query(question, retrieved_docs):
    """
    Optimize search query based on user question and RAG retrieval for volcanic rock character
    
    Args:
        question: User's original question
        retrieved_docs: RAG retrieved documents
    
    Returns:
        str: Optimized search query
    """
    # Extract key concepts from RAG documents
    rag_keywords = set()
    for doc in retrieved_docs[:2]:  # Only look at top 2 most relevant documents
        content = doc.page_content.lower()
        # Extract key geology/volcanology related vocabulary
        geo_keywords = ['volcanic rock', 'basalt', 'magma', 'lava', 'igneous', 
                       'madeira', 'portugal', 'atlantic', 'geology', 'volcanology',
                       'eruption', 'volcano', 'crust', 'mantle', 'mineral',
                       'formation', 'geological', 'rock cycle', 'plate tectonics',
                       'archipelago', 'macaronesia', 'azores', 'canary islands',
                       'oceanic island', 'hotspot', 'seamount', 'pillow lava',
                       'vesicular', 'porphyritic', 'crystallization', 'solidification',
                       'quaternary', 'miocene', 'pliocene', 'cenozoic']
        for keyword in geo_keywords:
            if keyword in content:
                rag_keywords.add(keyword)
    
    # Build precise search query
    base_query = "Madeira volcanic rock geology"
    
    # Add relevant context keywords
    if 'formation' in rag_keywords or 'geological' in rag_keywords:
        base_query += " formation process age"
    elif 'eruption' in rag_keywords or 'volcano' in rag_keywords:
        base_query += " volcanic eruption history timeline"
    elif 'basalt' in rag_keywords or 'lava' in rag_keywords:
        base_query += " basalt composition mineralogy"
    elif 'macaronesia' in rag_keywords or 'archipelago' in rag_keywords:
        base_query += " Macaronesian islands geological origin"
    elif 'madeira' in rag_keywords or 'portugal' in rag_keywords:
        base_query += " Madeira island geology landscape"
    else:
        base_query += " volcanic rock characteristics properties"
    
    # Add English keywords to ensure search quality
    base_query += " igneous rock geology scientific"
    
    return base_query

def filter_search_results(results, question):
    """
    Intelligently filter search results, excluding irrelevant content for volcanic geology
    
    Args:
        results: Raw search results list
        question: User question
    
    Returns:
        list: Filtered relevant results
    """
    filtered = []
    
    # Relevant keywords (geology/volcanology related)
    relevant_keywords = [
        'volcanic rock', 'basalt', 'magma', 'lava', 'igneous', 'madeira', 
        'portugal', 'geology', 'volcanology', 'eruption', 'volcano', 'crust', 
        'mantle', 'mineral', 'formation', 'geological', 'rock cycle', 
        'plate tectonics', 'archipelago', 'macaronesia', 'azores', 
        'canary islands', 'oceanic island', 'hotspot', 'seamount', 
        'pillow lava', 'vesicular', 'porphyritic', 'crystallization', 
        'solidification', 'quaternary', 'miocene', 'pliocene', 'cenozoic',
        'ignimbrite', 'tuff', 'pyroclastic', 'stratovolcano', 'shield volcano',
        'fumarole', 'geothermal', 'petrology', 'lithology', 'bedrock',
        'weathering', 'erosion', 'sedimentary', 'metamorphic'
    ]
    
    # Irrelevant keywords (other contexts)
    irrelevant_keywords = [
        'music', 'band', 'song', 'album', 'rock music', 'concert',
        'mining', 'quarry', 'construction', 'building material', 'gravel',
        'video game', 'game character', 'fictional', 'fantasy',
        'climbing', 'bouldering', 'rock climbing', 'sports',
        'gemstone', 'jewelry', 'precious stone', 'decoration',
        'programming', 'software', 'code', 'framework', 'development'
    ]
    
    for result in results:
        title = result.get('title', '').lower()
        body = result.get('body', '').lower()
        combined = title + ' ' + body
        
        # Check for irrelevant keywords
        has_irrelevant = any(keyword in combined for keyword in irrelevant_keywords)
        if has_irrelevant:
            print(f"[Fact-Check] Filtered irrelevant result: {result.get('title', 'Unknown')[:50]}...")
            continue
        
        # Check for relevant keywords
        has_relevant = any(keyword in combined for keyword in relevant_keywords)
        if has_relevant:
            filtered.append(result)
        else:
            # Additional check: if title contains key geological terms, keep it
            title_lower = title.lower()
            if any(term in title_lower for term in ['madeira geology', 'volcanic rock', 'basalt formation', 'magma', 'igneous rock']):
                filtered.append(result)
    
    return filtered


def web_search_supplement(question, retrieved_docs=None, language="English"):
    """
    智能网络搜索补充信息
    支持 DuckDuckGo（免费）和 Tavily（需 API Key）
    
    Args:
        question: 用户问题
        retrieved_docs: RAG 检索到的文档（用于优化搜索查询）
        language: 语言
    
    Returns:
        str: 网络搜索结果摘要（如果启用）
    """
    # 检查是否启用网络搜索
    use_web_search = os.getenv("USE_WEB_SEARCH", "false").lower() == "true"
    
    if not use_web_search:
        return None
    
    # 优化搜索查询（基于 RAG 上下文）
    if retrieved_docs and len(retrieved_docs) > 0:
        optimized_query = optimize_search_query(question, retrieved_docs)
        print(f"[Fact-Check] 优化搜索查询: {optimized_query}")
    else:
        optimized_query = f"Mediterranean monk seal {question} marine mammal"
    
    # 获取搜索提供商（默认 duckduckgo）
    provider = os.getenv("WEB_SEARCH_PROVIDER", "duckduckgo").lower()
    
    # 方案 1: DuckDuckGo（完全免费，无需 API Key）
    results = []  # 初始化 results 变量
    
    if provider == "duckduckgo":
        try:
            from ddgs import DDGS
            
            # 使用新版 API（无需 context manager）
            ddgs = DDGS()
            # 新版 API：参数名是 query 而不是 keywords
            raw_results = list(ddgs.text(
                query=optimized_query,
                max_results=5  # 多获取一些结果，后续过滤
            ))
            
            # 智能过滤结果
            results = filter_search_results(raw_results, question)
            print(f"[Fact-Check] 原始结果: {len(raw_results)} → 过滤后: {len(results)}")
            
            if results:
                if language == "Portuguese":
                    summary = "🌐 **Informação da Internet:**\n\n"
                else:
                    summary = "🌐 **Internet Information:**\n\n"
                
                # 只显示前2个最相关的结果
                for i, result in enumerate(results[:2], 1):
                    title = result.get('title', 'Unknown')
                    body = result.get('body', '')[:150]
                    url = result.get('href', '')
                    
                    summary += f"{i}. **{title}**\n   {body}...\n   🔗 {url}\n\n"
                
                return summary.strip()
        
        except ImportError:
            print("[Fact-Check] DDGS 未安装，运行: pip install ddgs")
        except Exception as e:
            print(f"[Fact-Check] DuckDuckGo 搜索失败: {str(e)}")
            print(f"[Fact-Check] 尝试降级到 Tavily...")
    
    # 方案 2: Tavily（需要 API Key，1000 次/月免费）
    # 如果 DuckDuckGo 失败或提供商设置为 tavily，尝试 Tavily
    if provider == "tavily" or (provider == "duckduckgo" and len(results) == 0):
        try:
            tavily_key = os.getenv("TAVILY_API_KEY")
            if tavily_key and tavily_key != "tvly-your-api-key":
                from tavily import TavilyClient
                
                client = TavilyClient(api_key=tavily_key)
                response = client.search(
                    query=f"Mediterranean monk seal {question}",
                    max_results=2,
                    search_depth="basic"
                )
                
                if response and 'results' in response:
                    results = response['results'][:2]
                    
                    if language == "Portuguese":
                        summary = "🌐 **Informação da Internet:**\n\n"
                    else:
                        summary = "🌐 **Internet Information:**\n\n"
                    
                    for i, result in enumerate(results, 1):
                        title = result.get('title', 'Unknown')
                        content = result.get('content', '')[:150]
                        url = result.get('url', '')
                        
                        summary += f"{i}. **{title}**\n   {content}...\n   🔗 {url}\n\n"
                    
                    return summary.strip()
        
        except ImportError:
            print("[Fact-Check] Tavily 未安装，运行: pip install tavily-python")
        except Exception as e:
            print(f"[Fact-Check] Tavily 搜索失败: {str(e)}")
    
    return None


def generate_fact_check_content(question, retrieved_docs, ai_answer, language="English"):
    """
    生成完整的 Fact-Check 内容（智能优化版）
    
    Args:
        question: 用户问题
        retrieved_docs: 检索到的文档
        ai_answer: AI 回答
        language: 语言
    
    Returns:
        str: HTML 格式的 Fact-Check 内容
    """
    # 1. 生成知识库摘要
    kb_summary = summarize_fact_check(question, retrieved_docs, ai_answer, language)
    
    # 2. 可选：智能网络搜索补充（传递 RAG 文档用于优化搜索查询）
    web_summary = web_search_supplement(
        question=question, 
        retrieved_docs=retrieved_docs,  # 传递 RAG 上下文优化搜索
        language=language
    )
    
    # 3. 组合内容
    if language == "Portuguese":
        header = "📋 **Verificação de Factos Baseada em Conhecimento Científico**\n\n"
    else:
        header = "📋 **Fact-Check Based on Scientific Knowledge**\n\n"
    
    content = header + kb_summary
    
    if web_summary:
        content += f"\n\n---\n\n{web_summary}"
    
    return content


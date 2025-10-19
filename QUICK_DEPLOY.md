# 🚀 Zino's Chat - 快速部署指南

**3 种部署方式，10 分钟上线！**

---

## 📋 部署前准备

### 必需的 API Keys

| API | 用途 | 获取地址 | 费用 |
|-----|------|----------|------|
| **Qwen API** | LLM + TTS + Embeddings | [DashScope](https://dashscope.aliyun.com/) | 免费额度可用 |
| **Supabase** | 交互记录存储 | [Supabase](https://supabase.com/) | 免费计划足够 |

### 可选的 API Keys

| API | 用途 | 获取地址 | 费用 |
|-----|------|----------|------|
| **Tavily** | 高质量网络搜索 | [Tavily](https://tavily.com/) | 1000次/月免费 |

---

## 🎯 方式 1: 本地部署（Windows）⭐ 推荐新手

### 步骤 1: 克隆项目
```bash
git clone https://github.com/你的用户名/zinos-chat.git
cd zinos-chat
```

### 步骤 2: 安装依赖
```bash
# 使用 pip
pip install -r requirements.txt
```

### 步骤 3: 配置环境变量
```bash
# 1. 复制配置模板
copy config.env.template .env

# 2. 编辑 .env 文件，填入你的 API Keys
notepad .env
```

**必需配置：**
```env
DASHSCOPE_API_KEY=sk-你的Qwen密钥
SUPABASE_URL=https://你的项目.supabase.co
SUPABASE_KEY=你的Supabase密钥
```

**可选配置：**
```env
USE_WEB_SEARCH=true
WEB_SEARCH_PROVIDER=duckduckgo
TAVILY_API_KEY=tvly-你的密钥（可选）
```

### 步骤 4: 设置 RAG 知识库
```bash
# 一键设置（推荐）
setup_rag_system.bat

# 或手动执行
pip install tqdm
python vectorize_knowledge_base.py
```

**等待 5-10 分钟**，完成后应看到：
```
✅ 向量数据库创建成功！
📊 统计信息:
   - 文档数量: 1298 blocks
   - 嵌入模型: text-embedding-v3
```

### 步骤 5: 启用智能网络搜索（可选）
```bash
# 已包含在 requirements.txt，无需额外操作
# 网络搜索功能将自动启用
```

### 步骤 6: 运行应用
```bash
streamlit run main.py
```

**访问**: http://localhost:8501

---

## 🌐 方式 2: Streamlit Cloud 部署（在线访问）

### 步骤 1: 准备 GitHub 仓库
```bash
# 1. Fork 或推送项目到你的 GitHub
git add .
git commit -m "Initial commit"
git push origin main
```

### 步骤 2: 部署到 Streamlit Cloud

1. 访问 [Streamlit Cloud](https://share.streamlit.io/)
2. 点击 "New app"
3. 选择你的 GitHub 仓库
4. 配置：
   - **Main file path**: `main.py`
   - **Python version**: 3.11

### 步骤 3: 配置 Secrets

在 Streamlit Cloud 设置页面，添加以下 Secrets：

```toml
# .streamlit/secrets.toml

# 必需配置
DASHSCOPE_API_KEY = "sk-你的Qwen密钥"
SUPABASE_URL = "https://你的项目.supabase.co"
SUPABASE_KEY = "你的Supabase密钥"

# 可选配置
USE_WEB_SEARCH = "true"
WEB_SEARCH_PROVIDER = "duckduckgo"
TAVILY_API_KEY = "tvly-你的密钥"

# 模型配置
QWEN_MODEL_NAME = "qwen-turbo"
QWEN_EMBEDDING_MODEL = "text-embedding-v3"
QWEN_TTS_MODEL = "qwen3-tts-flash"
QWEN_TTS_VOICE = "Marcus"
```

### 步骤 4: 部署并测试

1. 点击 "Deploy"
2. 等待部署完成（约 2-3 分钟）
3. 访问你的应用链接

---

## 🐧 方式 3: Linux/Mac 部署

### 步骤 1: 克隆项目
```bash
git clone https://github.com/你的用户名/zinos-chat.git
cd zinos-chat
```

### 步骤 2: 创建虚拟环境（推荐）
```bash
# Python venv
python3 -m venv venv
source venv/bin/activate

# 或使用 conda
conda create -n zinos python=3.11
conda activate zinos
```

### 步骤 3: 安装依赖
```bash
pip install -r requirements.txt
```

### 步骤 4: 配置环境变量
```bash
# 复制配置模板
cp config.env.template .env

# 编辑配置
nano .env
# 或 vim .env
```

### 步骤 5: 设置 RAG 知识库
```bash
pip install tqdm
python vectorize_knowledge_base.py
```

### 步骤 6: 启用网络搜索（可选）
```bash
pip install ddgs
```

在 `.env` 中添加：
```env
USE_WEB_SEARCH=true
WEB_SEARCH_PROVIDER=duckduckgo
```

### 步骤 7: 运行应用
```bash
streamlit run main.py
```

---

## 🧪 部署后测试

### 1. 基础功能测试

访问应用后：
1. ✅ 选择语言（English/Português）
2. ✅ 输入问题："Hi, how are you?"
3. ✅ 检查 AI 回复
4. ✅ 检查语音播放

### 2. RAG 质量测试
```bash
# 完整测试
python test_rag_quality.py

# 快速测试
python test_user_questions.py
```

**期望结果：**
```
✅ 向量库路径: db5_qwen
✅ 文档数量: 1298
✅ 检索质量: 优秀（覆盖率 ≥75%）
```

### 3. 网络搜索测试
```bash
python test_smart_search.py
```

**期望结果：**
```
✅ 搜索查询优化正常
✅ 结果过滤正常（无技术/编程内容）
✅ 全部测试通过
```

---

## 🔧 常见部署问题

### 问题 1: DDGS 包错误

**错误：**
```
DDGS.text() missing 1 required positional argument: 'query'
```

**解决：**
```bash
# 卸载旧包，安装新包
pip uninstall duckduckgo-search -y
pip install ddgs
```

---

### 问题 2: 向量数据库为空

**错误：**
```
文档数量: 0
```

**解决：**
```bash
# 确保嵌入模型配置正确
# 在 .env 中：
QWEN_EMBEDDING_MODEL=text-embedding-v3

# 重新向量化
python vectorize_knowledge_base.py
```

---

### 问题 3: Streamlit Cloud 部署失败

**错误：**
```
ModuleNotFoundError: No module named 'ddgs'
```

**解决：**
确保 `requirements.txt` 包含：
```
ddgs
tavily-python
```

---

### 问题 4: Supabase 连接失败

**错误：**
```
Connection to Supabase failed
```

**解决：**
1. 检查 Supabase URL 和 Key 是否正确
2. 确保数据库表已创建：
   - 运行 `create_table_interactions.sql`
   - 或在 Supabase Dashboard 手动创建

---

## 📊 部署检查清单

### 环境配置 ✅

- [ ] Python 3.11+ 已安装
- [ ] 所有依赖已安装（`pip install -r requirements.txt`）
- [ ] `.env` 文件已配置
- [ ] Qwen API Key 有效
- [ ] Supabase URL 和 Key 有效

### RAG 系统 ✅

- [ ] 向量数据库已创建（`db5_qwen/`）
- [ ] 文档数量 = 1298
- [ ] 嵌入模型 = text-embedding-v3
- [ ] RAG 测试通过（`test_rag_quality.py`）

### 网络搜索 ✅

- [ ] DDGS 包已正确安装
- [ ] `USE_WEB_SEARCH=true` 已配置
- [ ] 搜索测试通过（`test_smart_search.py`）

### 应用功能 ✅

- [ ] 应用可正常访问
- [ ] 对话功能正常
- [ ] 语音合成正常
- [ ] Fact-Check 功能正常
- [ ] 双语切换正常

---

## 🚀 快速命令参考

### Windows
```bash
# 完整部署流程
git clone <repo>
cd zinos-chat
pip install -r requirements.txt
copy config.env.template .env
# 编辑 .env 填入 API Keys
setup_rag_system.bat
streamlit run main.py
```

### Linux/Mac
```bash
# 完整部署流程
git clone <repo>
cd zinos-chat
pip install -r requirements.txt
cp config.env.template .env
# 编辑 .env 填入 API Keys
pip install tqdm ddgs
python vectorize_knowledge_base.py
streamlit run main.py
```

### Streamlit Cloud
```bash
# 1. 推送到 GitHub
git push origin main

# 2. 访问 share.streamlit.io
# 3. 连接仓库并配置 Secrets
# 4. 点击 Deploy
```

---

## 📚 下一步

部署成功后：

1. **体验核心功能**: 与 Zino's Petrel 对话
2. **测试 RAG 质量**: 运行 `test_rag_quality.py`
3. **测试智能搜索**: 运行 `test_smart_search.py`
4. **阅读完整文档**: [docs/COMPLETE_GUIDE.md](docs/COMPLETE_GUIDE.md)
5. **自定义配置**: 调整 `.env` 中的参数

---

## 🆘 获取帮助

遇到问题？

1. **查看文档**: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)
2. **运行测试**: `python test_*.py` 诊断问题
3. **查看日志**: 控制台输出中的 `[Fact-Check]`, `[RAG]` 等信息
4. **提交 Issue**: [GitHub Issues](https://github.com/你的用户名/zinos-chat/issues)

---

**祝部署顺利！** 🎉

[⬆ 返回顶部](#-zinos-chat---快速部署指南)


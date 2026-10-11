# Private RAG System

私有知识库RAG系统，基于 Qwen2-7B + Qdrant + Ollama + FastAPI，支持混合检索、重排序、多轮对话、流式输出与量化评测。

## 技术栈

- **大模型**: Qwen2-7B-Instruct (Ollama)
- **向量库**: Qdrant (本地嵌入模式)
- **嵌入模型**: bge-m3 (Ollama)
- **检索**: BM25 + 向量检索 RRF融合（混合检索） + CrossEncoder Reranker精排
- **对话**: 支持基于session_id的多轮对话历史
- **输出**: 支持一次性返回 / SSE流式输出两种模式
- **框架**: LangChain LCEL (Runnable组合)
- **API**: FastAPI
- **评测**: 自研轻量RAG评测harness（检索命中率 / 回答命中率 / LLM裁判忠实度打分）
- **部署**: Docker Compose 一键起 Ollama + API，数据不出域

## 架构说明

```
                        ┌───────────────┐
用户问题 ──────────────▶│  FastAPI层     │
                        └──────┬────────┘
                               │
                   ┌───────────▼────────────┐
                   │   混合检索 (Ensemble)    │
                   │  BM25检索 ─┐            │
                   │  向量检索 ─┴─▶ RRF融合   │
                   └───────────┬────────────┘
                               │ Top-K候选
                   ┌───────────▼────────────┐
                   │ CrossEncoder Reranker  │  (可关闭/自动降级)
                   │      精排取Top-N        │
                   └───────────┬────────────┘
                               │ 最终上下文
                   ┌───────────▼────────────┐
                   │  LCEL Chain            │
                   │  Prompt(含chat_history) │
                   │  → Qwen2-7B (Ollama)   │
                   │  → invoke() / stream() │
                   └────────────────────────┘
```

**为什么要做混合检索**：纯向量检索对同义改写敏感度高，但对精确的专有名词、编号、缩写等关键词召回较弱；BM25正好互补。两路结果用倒数排名融合（RRF）合并后，再用一个专门训练过的CrossEncoder对Query-Doc pair做精排，比"只看向量相似度分数"排序更准。

**为什么不用 `EnsembleRetriever` + `ContextualCompressionRetriever`**：这两个组件拿不到可解释的分数，也无法在"没有任何相关分片"时返回空结果——无关chunk照样会被塞进prompt，模型就会绕开上下文、用自己的通用知识作答（表现为"不看文档也能答"）。所以检索层在 `app/rag/retriever.py` 里自己实现了 `RRF融合 → CrossEncoder精排 → 阈值过滤 → 去重 → 按字符数截断`，保证要么给出真正相关的上下文，要么返回空让上层回复固定话术 `NO_ANSWER_TEXT`。

**中文BM25分词**：`BM25Retriever` 默认分词是 `text.split()`，中文整段话会被当成一个token，BM25基本失效。这里用「英文按词 + 中文单字/双字」的分词器（`tokenize_for_bm25`，无需额外依赖）。

**降级与稳健性**：

- Reranker权重首次运行要联网下载，下载/加载失败会自动降级为「混合检索 + 相似度阈值」，不会导致服务不可用；加载在后台线程预热，不会把第一个请求卡住。
- `SIMILARITY_THRESHOLD` 在Qdrant侧过滤不相关分片，`MAX_CONTEXT_CHARS` 限制拼给LLM的上下文长度，两者配合 `N_CTX`（默认8192）避免prompt被Ollama静默截断——中文1个汉字≈1个token，窗口不够时截断掉的正是检索到的上下文，模型自然就开始自由发挥。
- 上传时按正文内容去重：同一份PDF重复上传不会重复入库，否则向量检索的Top-K会被完全相同的分片占满。
- BM25语料是进程内内存态的，服务重启时会通过 `scroll_all_documents` 从Qdrant里把已有chunk回灌回来，避免重启后混合检索"失忆"退化成纯向量检索。

## 项目结构

```
rag_project/
├── app/
│   ├── main.py               # FastAPI入口：问答/流式问答/上传/清空/会话管理
│   ├── config.py             # 配置管理
│   └── rag/
│       ├── chain.py          # LCEL RAG链构建（含多轮历史 + 流式）
│       ├── loader.py         # PDF文档加载
│       ├── splitter.py       # 文本分割
│       ├── vectorstore.py    # Qdrant向量库操作 + 全量回灌
│       ├── retriever.py      # 混合检索(BM25+向量) + Reranker
│       └── memory.py         # 多轮对话历史管理
├── eval/
│   ├── eval_dataset.json     # 评测问题集（需替换成你自己的文档对应的问题）
│   └── evaluate.py           # 评测脚本
├── tests/                    # pytest单元测试
├── models/                   # 本地Embedding模型（如有）
├── qdrant_data/               # Qdrant本地数据
├── Dockerfile
├── docker-compose.yml         # 一键起 Ollama + API 服务
├── .env / .env.example
├── requirements.txt
└── README.md
```

## 快速开始

### 方式一：本地直接跑

1. 安装依赖

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

2. 启动Ollama并拉取模型

```bash
ollama pull qwen2:7b-instruct
ollama pull bge-m3
```

3. 复制配置文件

```bash
cp .env.example .env
```

4. 启动服务

```bash
python -m app.main
```

5. 打开 http://localhost:8000/docs 用Swagger UI测试

### 方式二：Docker Compose 一键起

```bash
cp .env.example .env
docker compose up -d --build
# 首次启动后，进容器给ollama拉模型（或改成启动脚本自动拉取）
docker exec -it rag_ollama ollama pull qwen2:7b-instruct
docker exec -it rag_ollama ollama pull bge-m3
```

## 前端页面

`frontend/index.html` 是一个不依赖任何构建工具的单文件前端，覆盖了后端全部接口：

- 左侧栏：拖拽/选择PDF上传（显示分片数）、清空知识库、会话切换/新建、清空当前会话历史、后端健康状态指示灯、可编辑的API地址
- 主区域：问答对话（气泡式），每条回答可展开查看引用来源（命中的文档片段+页码），支持"流式"/"标准"两种应答模式切换
- 会话历史和界面设置保存在浏览器 `localStorage` 里，仅存在本地，不会上传到任何地方

**推荐用法**：直接 `python -m app.main` 启动后打开 http://localhost:8000/ ——项目已经把 `frontend/` 挂载在根路径下，同源访问不需要处理CORS。如果你把前端单独部署到别的地址/端口，左下角的"API地址"输入框可以改指向后端实际地址（后端已加了 `CORSMiddleware` 允许跨域）。


## API接口（Swagger http://localhost:8000/docs）

1. `POST /api/document/upload` —— 上传PDF文档，解析切片入库（同时更新BM25索引）
2. `POST /api/rag/query` —— RAG问答（一次性返回），支持 `session_id` 做多轮对话
   ```json
   {"question": "你的问题", "session_id": "user-001"}
   ```
3. `POST /api/rag/query/stream` —— RAG问答（SSE流式返回token）
4. `DELETE /api/session/{session_id}` —— 清空指定会话的对话历史
5. `DELETE /api/knowledge/clear` —— 清空知识库集合
6. `GET /api/health` —— 健康检查

## 运行测试

```bash
pytest -v
```

目前覆盖：文本分割的边界情况、PDF加载的异常处理、多轮对话历史管理逻辑、检索层（中文BM25分词/重复分片去重/RRF融合与元数据/字符预算截断/检索全失败返回空）。检索层用替身检索器注入固定分片，不依赖Ollama与Qdrant，可以离线跑；真实端到端效果仍走 `eval/evaluate.py`。

## 评测

```bash
# 先启动服务，再把 eval/eval_dataset.json 换成你实际文档对应的问题+关键词
python -m eval.evaluate
```

会输出：
- **检索命中率**：检索到的上下文是否包含预期关键词（context recall代理指标）
- **回答关键词命中率**：最终回答是否包含预期关键词（answer relevancy代理指标）
- **平均忠实度评分**：用同一个本地LLM当裁判，给"回答是否忠于检索上下文"打1-5分（衡量幻觉程度）

结果写入 `eval/eval_results.json`
提高：

## 后续可以继续做的方向,目前来不急

- 把BM25/向量语料从进程内内存换成持久化存储，支持多副本部署
- 多会话历史存储换成Redis，支撑水平扩展
- 接入LangSmith或自建trace日志，记录每次检索到的chunk、耗时、token消耗
- Query Rewriting / HyDE，处理口语化、指代不清的提问
- 简单前端(Streamlit/Gradio)，替代目前只有Swagger UI的交互方式

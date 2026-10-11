import logging
import os

from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda

from app.config import settings
from app.rag.retriever import hybrid_retriever_manager

logger = logging.getLogger(__name__)


def load_llm():
    """通过Ollama加载Qwen本地模型"""
    llm = ChatOllama(
        model=settings.LLM_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=settings.TEMPERATURE,
        num_ctx=settings.N_CTX,
        num_predict=settings.MAX_TOKENS,
    )
    logger.info(
        f"Ollama LLM加载完成: {settings.LLM_MODEL} "
        f"(temperature={settings.TEMPERATURE}, num_ctx={settings.N_CTX})"
    )
    return llm


def format_docs(docs) -> str:
    """
    把检索结果拼成「带编号 + 来源 + 页码」的上下文。
    编号让模型可以按 [1][2] 引用，前端展开的引用来源也能和正文对得上。
    """
    if not docs:
        return ""
    blocks = []
    for idx, doc in enumerate(docs, start=1):
        source = os.path.basename(str(doc.metadata.get("source") or "未知来源"))
        page = doc.metadata.get("page")
        location = f"{source} 第{int(page) + 1}页" if isinstance(page, (int, float)) else source
        blocks.append(f"[{idx}] 来源：{location}\n{doc.page_content.strip()}")
    return "\n\n".join(blocks)


SYSTEM_PROMPT = """
你是私有知识库问答助手，只能依据下面的【检索上下文】作答。

必须遵守：
1. 只允许使用【检索上下文】中明确写出的信息，禁止用你自己的通用知识补充、举例、扩展或推测。
2. 【检索上下文】为空，或其中没有足以回答该问题的信息时，只输出：{no_answer}
3. 回答要简短直接，优先使用上下文里的原文表述，并在引用处标注来源编号，例如 [1]。
4. 不要输出“根据上下文”“检索到”之类的套话，也不要复述这些规则。
5. 可以结合对话历史理解指代（如“它”“这个”指什么），但答案依据必须来自【检索上下文】。

【检索上下文】
{context}
""".strip()


RAG_SYSTEM_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{question}"),
])


def build_lcel_rag_chain(vector_db, llm):
    """
    LCEL RAG链路：
    - 检索层由 hybrid_retriever_manager 提供（BM25 + 向量 RRF融合 -> CrossEncoder重排 -> 阈值过滤）
    - 支持多轮对话历史（chat_history 可选注入）
    - 支持一次性 .invoke() 与流式 .stream()
    输入格式统一为 dict：
        {"question": str, "chat_history": List[BaseMessage] (可选), "documents": List[Document] (可选)}
    documents 用于复用上层已经检索好的结果（避免检索两次导致引用来源与实际上下文不一致）。
    """
    retriever = hybrid_retriever_manager.build_retriever(vector_db)

    def _prepare_prompt_inputs(inputs: dict) -> dict:
        documents = inputs.get("documents")
        if documents is None:
            documents = retriever.invoke(inputs["question"])
        return {
            "context": format_docs(documents),
            "question": inputs["question"],
            "chat_history": inputs.get("chat_history") or [],
            "no_answer": settings.NO_ANSWER_TEXT,
        }

    rag_chain = (
        RunnableLambda(_prepare_prompt_inputs)
        | RAG_SYSTEM_PROMPT
        | llm
        | StrOutputParser()
    )
    logger.info("LCEL RAG链路构建完成（混合检索 + Reranker + 阈值过滤 + 多轮历史 + 流式支持）")
    return rag_chain, retriever

import os
import json
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uuid

from app.config import settings
from app.rag.loader import load_pdf_documents
from app.rag.splitter import split_documents
from app.rag.vectorstore import get_qdrant_client, get_embedding, get_vector_store, scroll_all_documents
from app.rag.retriever import hybrid_retriever_manager
from app.rag.memory import session_memory

from app.rag.chain import load_llm, build_lcel_rag_chain

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 全局变量
rag_chain = None
retriever = None
qdrant_client = None
vector_db = None
llm = None


def _rebuild_chain():
    """重新用当前 vector_db 构建检索器 + RAG链，供上传/清空文档后刷新使用"""
    global rag_chain, retriever
    rag_chain, retriever = build_lcel_rag_chain(vector_db, llm)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务启动初始化模型、向量库，不再预加载PDF"""
    global rag_chain, retriever, qdrant_client, vector_db, llm
    try:
        logger.info("正在初始化RAG系统...")
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        qdrant_client = get_qdrant_client()
        embeddings = get_embedding()
        # 只初始化向量库，不加载文档；集合不存在则创建空集合
        vector_db = get_vector_store(qdrant_client, embeddings, split_docs=None)

        # 用Qdrant里已有的chunk回灌内存态BM25索引，保证重启后混合检索依然可用
        existing_docs = scroll_all_documents(qdrant_client, settings.QDRANT_COLLECTION_NAME)
        hybrid_retriever_manager.seed(existing_docs)

        llm = load_llm()
        _rebuild_chain()
        # Reranker后台预热：避免第一次问答时才下载权重把请求卡住
        hybrid_retriever_manager.warm_up_reranker()
        logger.info("RAG系统初始化完成，等待上传文档")
    except Exception as e:
        logger.critical(f"RAG系统初始化失败: {str(e)}")
        raise SystemExit(1)
    yield
    logger.info("服务关闭")


app = FastAPI(
    title="Private-RAG-Qwen-7B-Qdrant",
    version="2.0.0",
    description="企业级私有知识库RAG系统，Qwen2-7B + Qdrant + Ollama + 混合检索 + Reranker",
    lifespan=lifespan
)

# 允许前端页面跨域调用（本地开发场景；生产环境建议收窄为具体域名）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default"


@app.post("/api/rag/query", summary="RAG问答接口（支持多轮对话）")
def rag_query(req: QueryRequest):
    """RAG问答接口"""
    try:
        history = session_memory.get_history(req.session_id)

        # 只检索一次：检索结果既作为LLM的上下文，也作为返回给前端的引用来源
        source_docs = retriever.invoke(req.question)
        if len(source_docs) == 0:
            answer = settings.NO_ANSWER_TEXT
            session_memory.add_turn(req.session_id, req.question, answer)
            return {
                "code": 0,
                "msg": "success",
                "data": {
                    "question": req.question,
                    "answer": answer,
                    "source_documents": []
                }
            }

        answer = rag_chain.invoke({
            "question": req.question,
            "chat_history": history,
            "documents": source_docs,
        })
        session_memory.add_turn(req.session_id, req.question, answer)

        source_out = [
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in source_docs
        ]
        return {
            "code": 0,
            "msg": "success",
            "data": {
                "question": req.question,
                "answer": answer,
                "source_documents": source_out
            }
        }
    except Exception as e:
        logger.error(f"问答接口异常: {str(e)}")
        raise HTTPException(status_code=500, detail=f"服务内部异常: {str(e)}")


@app.post("/api/rag/query/stream", summary="RAG问答接口（SSE流式输出）")
def rag_query_stream(req: QueryRequest):
    """
    流式返回LLM生成的token，前端可用EventSource/fetch+ReadableStream逐字渲染。
    事件格式: data: {"type": "token", "content": "..."}  以及最后一条 {"type": "done"}
    """
    history = session_memory.get_history(req.session_id)

    def event_generator():
        collected = ""
        try:
            source_docs = retriever.invoke(req.question)
            if len(source_docs) == 0:
                answer = settings.NO_ANSWER_TEXT
                session_memory.add_turn(req.session_id, req.question, answer)
                yield f"data: {json.dumps({'type': 'token', 'content': answer}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
                return

            for chunk in rag_chain.stream({
                "question": req.question,
                "chat_history": history,
                "documents": source_docs,
            }):
                collected += chunk
                yield f"data: {json.dumps({'type': 'token', 'content': chunk}, ensure_ascii=False)}\n\n"

            session_memory.add_turn(req.session_id, req.question, collected)
            yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.error(f"流式问答异常: {str(e)}")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.delete("/api/session/{session_id}", summary="清空指定会话的多轮对话历史")
def clear_session(session_id: str):
    session_memory.clear(session_id)
    return {"code": 0, "msg": f"会话 {session_id} 历史已清空"}


@app.post("/api/document/upload", summary="上传PDF文档解析入库")
async def upload_document(file: UploadFile = File(...)):
    """上传PDF，解析、分片、写入向量库+BM25索引；swagger页面直接上传文件测试"""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="仅支持pdf文件")
    try:
        uid = str(uuid.uuid4())
        save_path = os.path.join(settings.UPLOAD_DIR, f"{uid}_{file.filename}")
        with open(save_path, "wb") as f:
            f.write(await file.read())
        logger.info(f"文件已保存: {save_path}")

        docs = load_pdf_documents(save_path)
        chunks = split_documents(docs)

        # 按正文内容去重：同一份PDF重复上传时不再重复入库，
        # 否则向量检索的Top-K会被完全相同的分片占满，挤掉其它有效内容
        new_chunks = hybrid_retriever_manager.select_new_documents(chunks)
        skipped = len(chunks) - len(new_chunks)
        if not new_chunks:
            logger.info(f"《{file.filename}》内容与知识库已有分片完全重复，跳过入库")
            return {
                "code": 0,
                "msg": "duplicated",
                "data": {
                    "filename": file.filename,
                    "chunk_count": 0,
                    "skipped_chunk_count": skipped,
                    "temp_file": save_path
                }
            }

        # 追加写入向量库 + 同步刷新BM25语料，然后重建检索器/链路
        vector_db.add_documents(new_chunks)
        hybrid_retriever_manager.add_documents(new_chunks)
        _rebuild_chain()

        logger.info(f"成功向向量库追加 {len(new_chunks)} 个分片（跳过重复 {skipped} 个）")
        return {
            "code": 0,
            "msg": "upload success",
            "data": {
                "filename": file.filename,
                "chunk_count": len(new_chunks),
                "skipped_chunk_count": skipped,
                "temp_file": save_path
            }
        }
    except Exception as e:
        logger.error(f"文档上传解析失败 {str(e)}")
        raise HTTPException(status_code=500, detail=f"解析失败:{str(e)}")


@app.delete("/api/knowledge/clear", summary="清空当前知识库集合")
def clear_knowledge():
    global vector_db
    try:
        coll_name = settings.QDRANT_COLLECTION_NAME
        # 如果集合存在才删除
        if qdrant_client.collection_exists(coll_name):
            qdrant_client.delete_collection(coll_name)
            logger.info(f"已删除集合 {coll_name}")
        # 删除完成，立刻重新初始化向量库，新建空集合，刷新全局对象
        embeddings = get_embedding()
        vector_db = get_vector_store(qdrant_client, embeddings, split_docs=None)
        # 同步清空BM25语料，重建检索器/链路
        hybrid_retriever_manager.clear()
        _rebuild_chain()
        logger.info(f"已重建空集合 {coll_name}")
        return {"code": 0, "msg": "知识库已清空并重建空集合"}
    except Exception as e:
        logger.error(f"清空知识库失败 {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health", summary="健康检查接口")
def health_check():
    return {"code": 0, "msg": "success", "data": {"status": "running"}}


# 挂载前端页面：必须放在所有 /api/... 路由之后注册，
# 否则这个通配的 "/" 挂载会先于具体路由匹配，把API请求也当成静态文件处理。
# 启动服务后直接打开 http://localhost:8000/ 即可使用，同源访问，无需处理跨域。
_frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.isdir(_frontend_dir):
    app.mount("/", StaticFiles(directory=_frontend_dir, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

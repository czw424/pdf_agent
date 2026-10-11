import logging
from typing import List

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http import models
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document

from app.config import settings

logger = logging.getLogger(__name__)


def get_embedding():
    embeddings = OllamaEmbeddings(
        model=settings.EMBED_MODEL_NAME,
        base_url=settings.OLLAMA_BASE_URL
    )
    logger.info(f"Ollama Embedding加载完成: {settings.EMBED_MODEL_NAME}")
    return embeddings


def get_qdrant_client() -> QdrantClient:
    client = QdrantClient(path=settings.QDRANT_PATH)
    logger.info(f"Qdrant本地模式启动成功，数据存储路径: {settings.QDRANT_PATH}")
    return client


def get_vector_store(client: QdrantClient, embeddings, split_docs=None):
    # 尝试加载已有集合
    try:
        client.get_collection(settings.QDRANT_COLLECTION_NAME)
        logger.info(f"检测到已存在集合 {settings.QDRANT_COLLECTION_NAME}，直接加载")
        db = Qdrant(
            client=client,
            collection_name=settings.QDRANT_COLLECTION_NAME,
            embeddings=embeddings
        )
    except (UnexpectedResponse, Exception):
        logger.info(f"集合不存在，新建Qdrant集合 {settings.QDRANT_COLLECTION_NAME}【空集合，等待后续上传文档】")
        # 获取嵌入向量维度
        test_embedding = embeddings.embed_query("test")
        vector_size = len(test_embedding)
        # 创建空集合
        client.create_collection(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE
            )
        )
        db = Qdrant(
            client=client,
            collection_name=settings.QDRANT_COLLECTION_NAME,
            embeddings=embeddings
        )
        # 如果传入分片文档才写入，没有就留空
        if split_docs is not None and len(split_docs) > 0:
            db.add_documents(split_docs)
            logger.info(f"成功添加 {len(split_docs)} 个文档到向量库")
        else:
            logger.info("空集合创建完成，等待接口上传文档")
    return db


def scroll_all_documents(client: QdrantClient, collection_name: str, batch_size: int = 256) -> List[Document]:
    """
    翻页拉取集合内全部已有chunk，用于服务重启后重建内存态的BM25索引。
    Qdrant + LangChain默认把正文/元数据存在payload的 page_content / metadata 字段中。
    """
    docs: List[Document] = []
    if not client.collection_exists(collection_name):
        return docs

    next_offset = None
    while True:
        records, next_offset = client.scroll(
            collection_name=collection_name,
            limit=batch_size,
            offset=next_offset,
            with_payload=True,
            with_vectors=False,
        )
        for record in records:
            payload = record.payload or {}
            content = payload.get("page_content", "")
            metadata = payload.get("metadata", {}) or {}
            if content:
                docs.append(Document(page_content=content, metadata=metadata))
        if next_offset is None:
            break

    logger.info(f"从已有集合恢复了 {len(docs)} 个chunk，用于重建BM25索引")
    return docs

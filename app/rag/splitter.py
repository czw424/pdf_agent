import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import settings

logger = logging.getLogger(__name__)


def split_documents(docs):
    """文档递归字符分片，生产常用参数"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "，", " ", ""]
    )
    split_docs = splitter.split_documents(docs)
    logger.info(f"文档分片完成，分片数量: {len(split_docs)}")
    return split_docs
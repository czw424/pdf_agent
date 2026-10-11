import os
import logging
from langchain_community.document_loaders import PyPDFLoader

logger = logging.getLogger(__name__)


def load_pdf_documents(file_path: str):
    """加载PDF文档"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF文件不存在: {file_path}")

    loader = PyPDFLoader(file_path)
    docs = loader.load()
    logger.info(f"PDF加载完成，总页数: {len(docs)}")
    return docs
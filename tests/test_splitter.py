from langchain_core.documents import Document
from app.rag.splitter import split_documents


def test_split_documents_respects_chunk_size():
    long_text = "这是一段测试文本。" * 200
    docs = [Document(page_content=long_text, metadata={"source": "test.pdf", "page": 0})]

    chunks = split_documents(docs)

    assert len(chunks) > 1
    for chunk in chunks:
        # 允许分隔符导致的少量超出，但不应远超配置的chunk_size
        assert len(chunk.page_content) <= 1000


def test_split_documents_preserves_metadata():
    docs = [Document(page_content="短文本，不需要切分。", metadata={"source": "a.pdf", "page": 1})]

    chunks = split_documents(docs)

    assert len(chunks) == 1
    assert chunks[0].metadata["source"] == "a.pdf"
    assert chunks[0].metadata["page"] == 1


def test_split_documents_empty_input():
    assert split_documents([]) == []

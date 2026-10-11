from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from app.config import settings
from app.rag.retriever import HybridRetrieverManager, tokenize_for_bm25


class _StubVectorRetriever:
    """替身向量检索器：单测里不依赖Ollama/Qdrant，直接返回给定文档。"""

    def __init__(self, docs):
        self._docs = list(docs)

    def invoke(self, question):
        return list(self._docs)


def _doc(text, source="test.pdf", page=0):
    return Document(page_content=text, metadata={"source": source, "page": page})


# ---------- 中文分词 ----------

def test_tokenize_for_bm25_keeps_ascii_words_and_cjk_units():
    tokens = tokenize_for_bm25("MySQL日志")

    assert "mysql" in tokens          # 英文单词整体保留
    assert "日" in tokens             # 中文单字
    assert "日志" in tokens           # 中文双字，保证精度
    assert "sql日" not in tokens      # 不应跨语言边界组合


def test_tokenize_for_bm25_handles_empty_text():
    assert tokenize_for_bm25("") == []


def test_bm25_with_custom_tokenizer_recalls_chinese_doc():
    docs = [
        _doc("Python 基础：列表、字典、循环与函数的使用方法。", source="python.pdf"),
        _doc("MySQL 日志：错误日志、二进制日志、慢查询日志的位置与查看方式。", source="mysql.pdf"),
    ]
    retriever = BM25Retriever.from_documents(docs, preprocess_func=tokenize_for_bm25)
    retriever.k = 1

    hits = retriever.invoke("MySQL的日志有哪几种")

    assert hits[0].metadata["source"] == "mysql.pdf"


# ---------- 重复分片去重 ----------

def test_select_new_documents_filters_cross_batch_duplicates():
    manager = HybridRetrieverManager()
    manager.seed([_doc("第一段内容"), _doc("第二段内容")])

    new_docs = manager.select_new_documents([_doc("第二段内容"), _doc("第三段内容")])

    assert [d.page_content for d in new_docs] == ["第三段内容"]


def test_select_new_documents_filters_duplicates_inside_one_batch():
    manager = HybridRetrieverManager()

    new_docs = manager.select_new_documents([_doc("同一段内容"), _doc("同一段内容")])

    assert len(new_docs) == 1


def test_add_documents_does_not_duplicate_corpus():
    manager = HybridRetrieverManager()
    manager.add_documents([_doc("内容A")])
    manager.add_documents([_doc("内容A")])

    assert len(manager._bm25_docs) == 1


# ---------- 融合检索 ----------

def test_retrieve_marks_fusion_metadata_and_dedups(monkeypatch):
    monkeypatch.setattr(settings, "USE_RERANKER", False, raising=False)
    manager = HybridRetrieverManager()
    shared = _doc("MySQL 错误日志记录了服务启动与停止时的信息。", source="mysql.pdf")
    python_doc = _doc("Python 列表推导式可以生成新的列表。", source="python.pdf")
    manager.seed([shared, python_doc])

    results = manager.retrieve("MySQL错误日志", _StubVectorRetriever([shared, python_doc]))

    contents = [d.page_content for d in results]
    assert len(contents) == len(set(contents))  # 去重后不出现完全相同的分片
    assert len(results) <= settings.RETRIEVER_K
    assert all("fusion_score" in d.metadata for d in results)
    assert all("retrieval_source" in d.metadata for d in results)
    # 两路检索都命中的分片应标注为 bm25+vector
    assert any(d.metadata["retrieval_source"] == "bm25+vector" for d in results)


def test_retrieve_returns_empty_when_both_retrievers_fail(monkeypatch):
    monkeypatch.setattr(settings, "USE_HYBRID_SEARCH", False, raising=False)
    manager = HybridRetrieverManager()

    class _BrokenRetriever:
        def invoke(self, question):
            raise RuntimeError("qdrant down")

    assert manager.retrieve("任意问题", _BrokenRetriever()) == []


def test_trim_respects_top_k_and_char_budget():
    docs = [_doc("一" * 100), _doc("二" * 100), _doc("三" * 100)]

    assert len(HybridRetrieverManager._trim(docs, max_chars=10_000, top_k=2)) == 2
    kept = HybridRetrieverManager._trim(docs, max_chars=250, top_k=3)
    assert [d.page_content[0] for d in kept] == ["一", "二"]

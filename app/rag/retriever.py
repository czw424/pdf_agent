import hashlib
import logging
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

# 先导入 settings：它会在导入任何 HF/sentence-transformers 相关库之前写好 HF_ENDPOINT
from app.config import settings

from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

logger = logging.getLogger(__name__)

_ASCII_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")


def tokenize_for_bm25(text: str) -> List[str]:
    """
    中文友好的 BM25 分词器：英文/数字按单词切，中文按「单字 + 相邻双字」切。

    为什么必须自定义：BM25Retriever 默认分词是 text.split()，中文句子没有空格，
    整段文字会被当成一个 token，词频统计基本失效（实测问“MySQL的日志有哪几种”
    会返回完全不相关的 Python 章节）。这里不引入 jieba 等额外依赖，
    用「单字 + 双字」的组合已经能让 BM25 在中文上正常工作。
    """
    tokens: List[str] = [w.lower() for w in _ASCII_TOKEN_RE.findall(text)]
    for run in _CJK_RUN_RE.findall(text):
        tokens.extend(run)  # 单字，保证召回
        tokens.extend(run[i:i + 2] for i in range(len(run) - 1))  # 双字，保证精度
    return tokens


class HybridRetrieverManager:
    """
    维护一份内存态的chunk语料用于BM25关键词检索，并与向量检索做RRF融合，
    最后可选接一层CrossEncoder Reranker做精排，并按阈值过滤。

    设计取舍：
    - BM25需要全量语料在内存里才能算词频统计（Qdrant只存向量+payload），
      所以这里用进程内list维护语料，新增文档时增量重建BM25索引；
      服务重启时通过 vectorstore.scroll_all_documents 从Qdrant回灌语料。
    - 不再直接使用 EnsembleRetriever + ContextualCompressionRetriever：
      这两个组件拿不到可解释的分数，也无法在“没有任何相关分片”时返回空结果，
      结果就是无关chunk照样塞给LLM，模型于是绕开上下文、用自己的通用知识回答。
      这里自己实现 RRF 融合 -> 重排 -> 阈值过滤 -> 去重 -> 截断，
      保证最终要么给出真正相关的上下文，要么返回空让上层回复“未找到相关信息”。
    """

    def __init__(self) -> None:
        self._bm25_docs: List[Document] = []
        self._bm25_retriever: Optional[BM25Retriever] = None
        self._content_hashes: set = set()
        self._cross_encoder: Any = None
        # idle / loading / ready / unavailable
        self._reranker_state = "idle"
        self._reranker_lock = threading.Lock()

    # ---------- 语料维护 ----------
    @staticmethod
    def content_hash(doc: Document) -> str:
        """按正文内容算指纹，用于跨文件/跨次上传的去重。"""
        return hashlib.sha1(doc.page_content.strip().encode("utf-8")).hexdigest()

    def _register(self, doc: Document) -> bool:
        """把分片登记进去重表；已存在返回False。"""
        digest = self.content_hash(doc)
        if digest in self._content_hashes:
            return False
        self._content_hashes.add(digest)
        return True

    def seed(self, docs: List[Document]) -> None:
        """服务启动时用Qdrant里已有的chunk回灌BM25语料与去重表"""
        self._bm25_docs = []
        self._content_hashes = set()
        for doc in docs:
            if self._register(doc):
                self._bm25_docs.append(doc)
        self._rebuild_bm25()

    def select_new_documents(self, docs: List[Document]) -> List[Document]:
        """
        筛掉与已有语料（或本批内部）正文重复的分片。
        同一份PDF重复上传时，否则会写入多份完全相同的chunk，
        向量检索的Top-K会被这些重复分片占满，挤掉其它真正有用的内容。
        """
        new_docs: List[Document] = []
        batch_hashes: set = set()
        for doc in docs:
            digest = self.content_hash(doc)
            if digest in self._content_hashes or digest in batch_hashes:
                continue
            batch_hashes.add(digest)
            new_docs.append(doc)
        return new_docs

    def add_documents(self, docs: List[Document]) -> None:
        """有新文档入库时增量更新BM25语料（调用前应先用 select_new_documents 去重）"""
        if not docs:
            return
        added = 0
        for doc in docs:
            if self._register(doc):
                self._bm25_docs.append(doc)
                added += 1
        if added:
            self._rebuild_bm25()
        else:
            logger.info("本批分片与已有语料重复，BM25语料未变化")

    def clear(self) -> None:
        self._bm25_docs = []
        self._bm25_retriever = None
        self._content_hashes = set()

    def _rebuild_bm25(self) -> None:
        if not self._bm25_docs:
            self._bm25_retriever = None
            return
        self._bm25_retriever = BM25Retriever.from_documents(
            self._bm25_docs,
            preprocess_func=tokenize_for_bm25,
        )
        self._bm25_retriever.k = settings.RETRIEVE_INIT_K
        logger.info(f"BM25索引重建完成（中文分词），当前语料量: {len(self._bm25_docs)}")

    # ---------- Reranker（懒加载，失败自动降级） ----------
    def _get_cross_encoder(self, wait: bool = False):
        """
        加载CrossEncoder。首次运行需要联网下载权重（约1.1GB），
        所以用状态机 + 锁保证只加载一次，且加载期间其它请求直接走降级路径。
        """
        with self._reranker_lock:
            if self._cross_encoder is not None:
                return self._cross_encoder
            if self._reranker_state == "unavailable":
                return None  # 已经确定不可用，不再重复触发下载
            if self._reranker_state == "loading":
                return None  # 正在加载中，本次请求先走降级路径
            self._reranker_state = "loading"

        encoder = None
        try:
            encoder = HuggingFaceCrossEncoder(model_name=settings.RERANK_MODEL_PATH)
            logger.info(f"Reranker加载完成: {settings.RERANK_MODEL_PATH}")
        except Exception as e:
            logger.warning(
                f"Reranker加载失败（{settings.RERANK_MODEL_PATH}），"
                f"降级为「混合检索 + 相似度阈值」: {e}"
            )

        with self._reranker_lock:
            self._cross_encoder = encoder
            self._reranker_state = "ready" if encoder is not None else "unavailable"
        return encoder

    def warm_up_reranker(self) -> None:
        """
        后台线程预热Reranker。
        否则第一次问答时才去下载/加载模型，那个请求会卡住几十秒到几分钟，
        看起来就像服务卡死；预热期间请求自动走「混合检索+阈值」的降级路径。
        """
        if not settings.USE_RERANKER or self._reranker_state != "idle":
            return
        thread = threading.Thread(
            target=self._get_cross_encoder,
            kwargs={"wait": True},
            name="reranker-warmup",
            daemon=True,
        )
        thread.start()
        logger.info("Reranker开始后台预热（首次运行需下载权重，失败会自动降级）")

    # ---------- RRF融合 ----------
    def _fuse(self, ranked_lists: List[Tuple[List[Document], float, str]]) -> List[Document]:
        """倒数排名融合(RRF) + 按正文去重，返回按融合分降序排列的分片。"""
        pool: Dict[str, Document] = {}
        fusion: Dict[str, float] = {}
        hit_sources: Dict[str, set] = {}

        for docs, weight, name in ranked_lists:
            for rank, doc in enumerate(docs, start=1):
                digest = self.content_hash(doc)
                # 复制一份，避免把打分写回内存语料里的原始Document
                pool.setdefault(
                    digest,
                    Document(page_content=doc.page_content, metadata=dict(doc.metadata)),
                )
                fusion[digest] = fusion.get(digest, 0.0) + weight / (settings.RRF_K + rank)
                hit_sources.setdefault(digest, set()).add(name)

        ordered_keys = sorted(fusion, key=lambda key: fusion[key], reverse=True)
        for key in ordered_keys:
            metadata = pool[key].metadata
            metadata["fusion_score"] = round(fusion[key], 6)
            metadata["retrieval_source"] = "+".join(sorted(hit_sources[key]))
            metadata["relevance_score"] = metadata["fusion_score"]
        return [pool[key] for key in ordered_keys]

    def _rerank(self, question: str, docs: List[Document], cross_encoder) -> List[Document]:
        """CrossEncoder 精排，并按阈值丢弃不相关分片。"""
        if not docs:
            return []
        pairs = [(question, doc.page_content) for doc in docs]
        try:
            scores = [float(score) for score in cross_encoder.score(pairs)]
        except Exception as e:
            logger.error(f"Reranker打分失败，退回RRF融合排序: {e}")
            return docs

        scored = sorted(zip(docs, scores), key=lambda pair: pair[1], reverse=True)
        kept: List[Document] = []
        for doc, score in scored:
            doc.metadata["rerank_score"] = round(score, 4)
            doc.metadata["relevance_score"] = round(score, 4)
            if score >= settings.RERANK_SCORE_THRESHOLD:
                kept.append(doc)
        logger.info(
            f"重排完成: {len(docs)} -> {len(kept)} 个分片（阈值 {settings.RERANK_SCORE_THRESHOLD}）"
        )
        return kept

    @staticmethod
    def _trim(docs: List[Document], max_chars: int, top_k: int) -> List[Document]:
        """按相关度顺序取Top-K，并限制总字符数，避免prompt超出N_CTX被截断。"""
        final: List[Document] = []
        used = 0
        for doc in docs:
            if len(final) >= top_k:
                break
            length = len(doc.page_content)
            if final and used + length > max_chars:
                continue
            final.append(doc)
            used += length
        return final

    # ---------- 对外暴露的检索入口 ----------
    def build_retriever(self, vector_db):
        """返回一个Runnable：输入query字符串，输出最终用于作答的Document列表。"""
        vector_retriever = vector_db.as_retriever(
            search_kwargs={
                "k": settings.RETRIEVE_INIT_K,
                # 在Qdrant侧就按相似度过滤，不相关的分片直接不进候选
                "score_threshold": settings.SIMILARITY_THRESHOLD,
            }
        )
        logger.info(
            f"向量检索配置: k={settings.RETRIEVE_INIT_K}, "
            f"score_threshold={settings.SIMILARITY_THRESHOLD}"
        )
        if settings.USE_HYBRID_SEARCH and self._bm25_retriever is not None:
            logger.info(
                f"启用混合检索(BM25 + 向量, RRF融合)，权重: "
                f"bm25={settings.BM25_WEIGHT}, vector={settings.VECTOR_WEIGHT}"
            )
        else:
            logger.warning("BM25语料为空或已关闭混合检索，本次退化为纯向量检索")

        def _retrieve(question: str) -> List[Document]:
            return self.retrieve(question, vector_retriever)

        return RunnableLambda(_retrieve)

    def retrieve(self, question: str, vector_retriever) -> List[Document]:
        """完整检索流程：向量 + BM25 -> RRF融合 -> 重排 -> 阈值过滤 -> 截断。"""
        ranked_lists: List[Tuple[List[Document], float, str]] = []

        try:
            ranked_lists.append(
                (vector_retriever.invoke(question), settings.VECTOR_WEIGHT, "vector")
            )
        except Exception as e:
            logger.error(f"向量检索失败，本次仅使用BM25结果: {e}")

        if settings.USE_HYBRID_SEARCH and self._bm25_retriever is not None:
            try:
                ranked_lists.append(
                    (self._bm25_retriever.invoke(question), settings.BM25_WEIGHT, "bm25")
                )
            except Exception as e:
                logger.error(f"BM25检索失败: {e}")

        if not ranked_lists:
            logger.error("向量检索与BM25检索全部失败，返回空上下文")
            return []

        candidates = self._fuse(ranked_lists)
        logger.info(f"RRF融合候选: {len(candidates)} 个分片")

        if settings.USE_RERANKER:
            cross_encoder = self._get_cross_encoder()
            if cross_encoder is not None:
                candidates = self._rerank(question, candidates, cross_encoder)

        final_docs = self._trim(candidates, settings.MAX_CONTEXT_CHARS, settings.RETRIEVER_K)
        logger.info(
            f"最终进入上下文的正文: {len(final_docs)} 个分片, "
            f"共 {sum(len(doc.page_content) for doc in final_docs)} 字"
        )
        return final_docs


hybrid_retriever_manager = HybridRetrieverManager()

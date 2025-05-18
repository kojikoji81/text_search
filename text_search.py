from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
import faiss
import numpy as np

# TREC Microblog Track データセットの一部をロード
microblog_dataset = load_dataset(
    "trec", "microblog-2011", split="train[:10000]", trust_remote_code=True
)

# データ構造を確認
print(microblog_dataset[0])

# "text" キーを使ってツイート本文を抽出し、langchainでチャンク分解
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
documents = []
for i in range(len(microblog_dataset)):
    tweet = microblog_dataset[i]["text"]
    if tweet:
        chunks = text_splitter.split_text(tweet)
        documents.extend([f"tweet: {chunk}" for chunk in chunks])

# Hugging Faceの高次元埋め込みモデル
model = SentenceTransformer('intfloat/multilingual-e5-large')

# クエリ
query = "Japan"

# ベクトル検索用の埋め込み
doc_embeddings = model.encode(documents, convert_to_numpy=True)
query_embedding = model.encode(query, convert_to_numpy=True)

# FAISSによるベクトル検索
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)
D, I = index.search(np.array([query_embedding]), 10)  # 上位10件
faiss_results = I[0].tolist()

# 表層検索（BM25）
tokenized_docs = [doc.split() for doc in documents]
bm25 = BM25Okapi(tokenized_docs)
bm25_scores = bm25.get_scores(query.split())
top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:10]
surface_results = [i for i in top_bm25_indices if bm25_scores[i] > 0]

# RRF（Reciprocal Rank Fusion）でハイブリッド順位付け
def rrf_score(rank, k=60):
    return 1 / (k + rank)

rrf_dict = {}
# FAISS順位
for rank, idx in enumerate(faiss_results):
    rrf_dict[idx] = rrf_dict.get(idx, 0) + rrf_score(rank)
# BM25順位
for rank, idx in enumerate(surface_results):
    rrf_dict[idx] = rrf_dict.get(idx, 0) + rrf_score(rank)

# RRFスコアでソート
rrf_sorted = sorted(rrf_dict.items(), key=lambda x: x[1], reverse=True)[:5]

# 結果表示
print("検索クエリ:", query)
print("\n--- ハイブリッドサーチ結果（RRF融合） ---")
for rank, (idx, score) in enumerate(rrf_sorted, 1):
    print(f"{rank}. {documents[idx]} (RRFスコア: {score:.4f})")
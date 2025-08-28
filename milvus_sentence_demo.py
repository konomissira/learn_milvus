from typing import List
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

# Embeddings (384-d)
# Model: all-MiniLM-L6-v2 → 384 dims
from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384-d
model = SentenceTransformer(MODEL_NAME)

def embed(texts: List[str]) -> List[List[float]]:
    # Get embeddings and L2-normalize for COSINE metric
    vecs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    vecs = vecs / norms
    return vecs.astype(np.float32).tolist()

#  1) Connect to Milvus
connections.connect("default", host="127.0.0.1", port="19530")

# 2) Prepare data
docs = [
    "Milvus is an open-source vector database for AI applications.",
    "Vector databases store high-dimensional vectors for similarity search.",
    "Sentence embeddings map text into numerical vectors.",
    "PostgreSQL is a relational database.",
    "Cosine similarity is common for text embeddings.",
    "HNSW is a popular approximate nearest neighbor index.",
    "Docker compose makes it easy to run multi-container apps."
]
doc_vecs = embed(docs)
DIM = len(doc_vecs[0])  # should be 384 for the chosen model

# 3) Define schema (id auto, text + vector)
collection_name = "sent_demo_384"

if collection_name in utility.list_collections():
    utility.drop_collection(collection_name)

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=DIM),
]
schema = CollectionSchema(fields, description="Sentence embeddings demo (384-d)")

col = Collection(name=collection_name, schema=schema)
print(f"✅ Created collection: {collection_name} (dim={DIM})")

# 4) Insert data
# Order of columns must match non-auto fields in schema: text, vector
mr = col.insert([docs, doc_vecs])
print(f"✅ Inserted {len(docs)} rows")

# 5) Build index (COSINE for text embeddings)
index_params = {
    "index_type": "HNSW",         # or "AUTOINDEX"
    "metric_type": "COSINE",      # match search metric
    "params": {"M": 16, "efConstruction": 200},
}
col.create_index(field_name="vector", index_params=index_params)
print("✅ Index created")

# 6) Load to memory
col.load()

# 7) Top-k search
query_text = "What database helps with vector similarity search?"
qvec = embed([query_text])  # normalized to match COSINE

res = col.search(
    data=qvec,
    anns_field="vector",
    param={"metric_type": "COSINE", "params": {"ef": 64}},  # ef for HNSW
    limit=3,               # top-k
    output_fields=["text"] # return original text
)

print(f"\n🔎 Query: {query_text}\nTop-3 results:")
for i, hit in enumerate(res[0], 1):
    # For COSINE, a larger score (closer to 1.0) means more similar
    print(f"{i}. id={hit.id}, score={hit.distance:.4f}, text={hit.entity.get('text')}")

# 8) Count / housekeeping
print(f"\nTotal entities: {col.num_entities}")
# col.release()      # free memory
# utility.drop_collection(collection_name)  # clean up
